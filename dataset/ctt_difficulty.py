"""Instance-difficulty measures for ITC2007 Track 3 (CB-CTT) instances.

structural_features() reads only the parsed CTTProblem (no solver call), so
it is cheap enough to run on every real or generated instance.
solver_measures() solves the instance once with one of the two exact
solvers -- "cpsat" (dataset.ctt_solver, ortools) or "milp"
(dataset.ctt_solver_milp, gurobipy, needs a license) -- and reports how hard
it was for that solver. profile_problem()/profile_instance() combine both
into one flat row (the `solver` option picks none/cpsat/milp/both);
profile_parallel() does the same for many instances at once, as
single-threaded solver runs on up to `cpus` CPUs in parallel. analyze()
relates the structural features to solve time and optimality gap across many
rows. generate_instances() draws fresh instances from dataset.ctt_generator
at a chosen size, to profile alongside (or instead of) .ctt files.

    # structural features only (milliseconds per instance)
    python -m dataset.ctt_difficulty 'data/ITC2007/real/comp*.ctt'
    # + both solvers on all CPUs, then the analysis; the CSV can be re-analyzed later
    python -m dataset.ctt_difficulty 'data/ITC2007/real/comp*.ctt' --solver both --time-limit 60 --out difficulty.csv
    python -m dataset.ctt_difficulty --analyze difficulty.csv
    # 30 generated instances per size instead of files, CP-SAT only, 8 CPUs
    python -m dataset.ctt_difficulty --generate 30 --size small medium --solver cpsat --cpus 8 --out generated.csv
"""
import argparse
import csv
import glob
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset.ctt_generator import SIZES, generate_problem, write_ctt
from dataset.ctt_parser import CTTProblem, parse_ctt

FEATURES = ("n_lectures", "room_fill", "max_group_load", "curricula_per_course", "conflict_density",
            "min_availability", "mean_availability", "capacity_tightness")
SOLVERS = ("cpsat", "milp")
SOLVER_CHOICES = {"none": (), "cpsat": ("cpsat",), "milp": ("milp",), "both": SOLVERS}
# network_measures() columns, prefixed "nn_" in rows
NN_FEATURES = ("plausible_periods", "plausible_rooms", "clash_ratio", "best_guess_hard", "full_period_share",
               "random_guess_hard", "hidden_blocking", "tag_auc", "mean_local_load", "max_local_load",
               "repair_steps")

Lecture = Tuple[str, int]


def _conflict_groups(problem: CTTProblem) -> List[List[str]]:
    """Course groups whose lectures must all get distinct periods: every
    curriculum (H3a) and every teacher's courses (H3b)."""
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    return [q.course_ids for q in problem.curricula.values()] + list(teacher_courses.values())


def _unavailable_periods(problem: CTTProblem) -> Dict[str, set]:
    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * problem.periods_per_day + u.period)
    return unavailable


def structural_features(problem: CTTProblem) -> Dict[str, float]:
    """Structural measures of how constrained an instance is:

      n_lectures            size: total lectures to schedule.
      room_fill             n_lectures / (n_periods * n_rooms) -- share of
                            all (period, room) slots that must be used (H2).
      max_group_load        lectures of the busiest curriculum or teacher /
                            n_periods -- must be <= 1 for H3a/H3b to be
                            satisfiable.
      curricula_per_course  number of curricula a course belongs to,
                            averaged over courses (not the generator's
                            n_curricula / n_courses ratio).
      conflict_density      share of course pairs sharing a curriculum or a
                            teacher, i.e. pairs that may never overlap
                            (H3a/H3b).
      min_availability,     per course: periods it may use (H4) /
      mean_availability     n_lectures; min is the tightest course, and
                            < 1 means the instance is infeasible.
      capacity_tightness    room_fill of the tightest capacity tier: for each
                            room capacity c, lectures too large for every
                            smaller room / (n_periods * rooms with capacity
                            >= c), maxed over c. Equals room_fill when
                            capacities never bind; > 1 means S1
                            (RoomCapacity) penalty is unavoidable.
    """
    n_periods = problem.n_periods
    courses = problem.courses
    n_lectures = sum(c.n_lectures for c in courses.values())
    groups = _conflict_groups(problem)

    max_group_load = max(sum(courses[cid].n_lectures for cid in g) for g in groups) / n_periods

    curricula_per_course = sum(len(q.course_ids) for q in problem.curricula.values()) / len(courses)

    conflicting = set()
    for g in groups:
        conflicting.update(itertools.combinations(sorted(set(g)), 2))
    n_pairs = len(courses) * (len(courses) - 1) // 2
    conflict_density = len(conflicting) / n_pairs if n_pairs else 0.0

    unavailable = _unavailable_periods(problem)
    availability = [(n_periods - len(unavailable.get(cid, ()))) / c.n_lectures for cid, c in courses.items()]

    # Supply only changes at capacity values, and within a tier demand is
    # largest just above the next-smaller capacity -- so one check per
    # distinct capacity covers every threshold.
    capacity_tightness = 0.0
    prev_capacity = float("-inf")
    for capacity in sorted({r.capacity for r in problem.rooms.values()}):
        demand = sum(c.n_lectures for c in courses.values() if c.n_students > prev_capacity)
        supply = n_periods * sum(1 for r in problem.rooms.values() if r.capacity >= capacity)
        capacity_tightness = max(capacity_tightness, demand / supply)
        prev_capacity = capacity

    return dict(
        n_lectures=n_lectures,
        room_fill=n_lectures / (n_periods * len(problem.rooms)),
        max_group_load=max_group_load,
        curricula_per_course=curricula_per_course,
        conflict_density=conflict_density,
        min_availability=min(availability),
        mean_availability=sum(availability) / len(availability),
        capacity_tightness=capacity_tightness,
    )


def _gap(objective: Optional[float], best_bound: Optional[float], proven_optimal: bool) -> float:
    """(objective - best_bound) / objective: an upper bound on how far the
    best found solution is from the optimum. 1.0 without any solution
    (nothing found counts as maximally far)."""
    if objective is None:
        return 1.0
    if proven_optimal or objective <= 0:  # objective is a sum of penalties, so 0 is optimal
        return 0.0
    return max(0.0, objective - best_bound) / objective


def solver_measures(problem: CTTProblem, solver: str = "cpsat", time_limit: float = 60.0,
                    workers: int = 1) -> Dict[str, object]:
    """Solves the instance once with `solver` ("cpsat" or "milp"), using
    `workers` threads (1 by default, so runs can go side by side on separate
    CPUs -- see profile_parallel()), and reports:

      status      OPTIMAL / FEASIBLE / NO_SOLUTION (time limit hit before any
                  incumbent) / INFEASIBLE (proven).
      objective   best found objective (None without a solution).
      best_bound  the solver's lower bound on the optimum (None without a
                  solution). CP-SAT's is typically much weaker than Gurobi's.
      solve_time  seconds inside the solver itself (CP-SAT WallTime, Gurobi
                  optimize()) -- excludes Python-side model building, which
                  differs a lot between the two and isn't search effort.
      gap         see _gap(); None when INFEASIBLE.
    """
    if solver == "cpsat":
        from dataset.ctt_solver import solve  # ortools CP-SAT
        result = solve(problem, time_limit=time_limit, workers=workers)
        solve_time = result["solve_time"]
    elif solver == "milp":
        from dataset.ctt_solver_milp import solve  # gurobipy -- needs a license, see ctt_solver_milp.py
        result = solve(problem, time_limit=time_limit, workers=workers)
        solve_time = result["optimize_time"]
    else:
        raise ValueError(f"unknown solver {solver!r}, expected one of {SOLVERS}")

    objective, best_bound = result["objective"], result["best_bound"]
    if objective is not None:
        status = result["status"]
    else:
        # ctt_solver_milp reports a no-incumbent time-out as "FEASIBLE"; CP-SAT as "UNKNOWN"
        status = "INFEASIBLE" if result["status"] == "INFEASIBLE" else "NO_SOLUTION"
    gap = None if status == "INFEASIBLE" else _gap(objective, best_bound, status == "OPTIMAL")
    return dict(status=status, objective=objective, best_bound=best_bound, solve_time=solve_time, gap=gap)


def _lecture_conflicts(problem: CTTProblem) -> Tuple[List[Lecture], List[np.ndarray]]:
    """Lectures in the order ctt_encode/evaluate_assignment use, and per
    lecture the indices of every lecture that may not share its period: the
    other lectures of its course (H1) and all lectures of courses sharing a
    curriculum or teacher with it (H3a/H3b)."""
    lectures = [(cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)]
    by_course: Dict[str, List[int]] = {}
    for i, (cid, _li) in enumerate(lectures):
        by_course.setdefault(cid, []).append(i)
    linked = {cid: {cid} for cid in problem.courses}
    for g in _conflict_groups(problem):
        for cid in g:
            linked[cid].update(g)
    neighbors = [np.array(sorted(j for other in linked[cid] for j in by_course[other] if j != i), dtype=np.int64)
                 for i, (cid, _li) in enumerate(lectures)]
    return lectures, neighbors


def _available_periods(problem: CTTProblem, lectures: List[Lecture]) -> List[np.ndarray]:
    unavailable = _unavailable_periods(problem)
    return [np.array([p for p in range(problem.n_periods) if p not in unavailable.get(cid, ())])
            for cid, _li in lectures]


def _entropy(q: np.ndarray) -> float:
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def ambiguity_measures(problem: CTTProblem, solutions: Sequence[Dict[str, dict]]) -> Dict[str, float]:
    """How ambiguous the instance is as a training target, from a pool of
    equally good solutions (ctt_solver_milp.solve_pool()["solutions"]),
    treated as the distribution a network trained on random optima would
    learn. Lectures of a course are numbered in period order in every
    solution -- one labelling the Hungarian-matched period loss in
    trm_ctt.py lets the model adopt.

      plausible_periods  exp(entropy) of each lecture's period distribution
                         across the pool, averaged over lectures: how many
                         periods even a perfectly trained model has to
                         spread its probability over. 1 = one answer.
      plausible_rooms    the same for rooms.
      clash_ratio        expected number of conflicting lecture pairs that
                         share a period when each lecture draws its period
                         independently from the pool distribution, divided
                         by the same for a uniform draw over its available
                         periods. ~1 = the per-lecture distribution says
                         nothing about avoiding clashes.
      best_guess_hard    hard violations per lecture when every lecture
                         takes its most frequent period and room -- what a
                         perfectly calibrated model decodes to by argmax.
      full_period_share  share of lectures placed in a period whose rooms
                         are all in use, averaged over the pool: their
                         rooms form a matching that independent per-lecture
                         room choices can't coordinate.

    The pool is whatever the solver found first, not a uniform sample of
    all optima.
    """
    from dataset.ctt_solver import evaluate_assignment

    lectures, neighbors = _lecture_conflicts(problem)
    index = {lec: i for i, lec in enumerate(lectures)}
    room_ids = list(problem.rooms)
    room_index = {rid: k for k, rid in enumerate(room_ids)}
    period_freq = np.zeros((len(lectures), problem.n_periods))
    room_freq = np.zeros((len(lectures), len(room_ids)))
    full_period_share = 0.0
    for solution in solutions:
        for cid, c in problem.courses.items():
            slots = sorted((solution[f"{cid}#{li}"]["period"], solution[f"{cid}#{li}"]["room"])
                           for li in range(c.n_lectures))
            for li, (period, room) in enumerate(slots):
                period_freq[index[(cid, li)], period] += 1
                room_freq[index[(cid, li)], room_index[room]] += 1
        per_period = np.bincount([a["period"] for a in solution.values()], minlength=problem.n_periods)
        full_period_share += per_period[per_period >= len(room_ids)].sum() / len(lectures)
    period_freq /= len(solutions)
    room_freq /= len(solutions)

    uniform = np.zeros_like(period_freq)
    for i, options in enumerate(_available_periods(problem, lectures)):
        uniform[i, options] = 1 / len(options)
    pairs = np.array([(i, j) for i, nb in enumerate(neighbors) for j in nb if j > i], dtype=np.int64).reshape(-1, 2)
    if len(pairs):
        first, second = pairs[:, 0], pairs[:, 1]
        clash_ratio = (np.einsum("ij,ij->", period_freq[first], period_freq[second])
                       / np.einsum("ij,ij->", uniform[first], uniform[second]))
    else:
        clash_ratio = float("nan")

    period_of = {lec: int(period_freq[i].argmax()) for i, lec in enumerate(lectures)}
    room_of = {lec: room_ids[int(room_freq[i].argmax())] for i, lec in enumerate(lectures)}
    return dict(
        plausible_periods=float(np.mean([np.exp(_entropy(row)) for row in period_freq])),
        plausible_rooms=float(np.mean([np.exp(_entropy(row)) for row in room_freq])),
        clash_ratio=float(clash_ratio),
        best_guess_hard=evaluate_assignment(problem, period_of, room_of)["hard_violations"] / len(lectures),
        full_period_share=float(full_period_share / len(solutions)),
    )


def _min_conflicts_steps(available: List[np.ndarray], neighbors: List[np.ndarray], n_periods: int,
                         n_rooms: int, rng: np.random.Generator, cap: int, noise: float = 0.1) -> int:
    """Moves a min-conflicts local search needs from a random timetable to
    one without hard violations (`cap` if it never gets there). Each move
    re-places one clashing lecture at the available (period, room) with the
    fewest clashes -- or, with probability `noise`, at a random one. H4
    holds throughout, since only available periods are ever chosen."""
    n = len(available)
    period = np.array([rng.choice(options) for options in available])
    room = rng.integers(n_rooms, size=n)
    occupancy = np.zeros((n_periods, n_rooms), dtype=np.int64)
    np.add.at(occupancy, (period, room), 1)
    for step in range(cap):
        clashing = [i for i in range(n)
                    if occupancy[period[i], room[i]] > 1 or (period[neighbors[i]] == period[i]).any()]
        if not clashing:
            return step
        i = clashing[rng.integers(len(clashing))]
        occupancy[period[i], room[i]] -= 1
        options = available[i]
        if rng.random() < noise:
            period[i], room[i] = rng.choice(options), rng.integers(n_rooms)
        else:
            cost = np.bincount(period[neighbors[i]], minlength=n_periods)[options][:, None] + occupancy[options]
            best = np.flatnonzero(cost == cost.min())
            k = best[rng.integers(len(best))]
            period[i], room[i] = options[k // n_rooms], k % n_rooms
        occupancy[period[i], room[i]] += 1
    return cap


def input_measures(problem: CTTProblem, seed: int = 0, draws: int = 20, repair_runs: int = 5,
                   repair_cap: int = 2000) -> Dict[str, float]:
    """Network-specific measures that need no solver:

      random_guess_hard  hard violations per lecture of a uniform random
                         timetable (available period, any room), mean of
                         `draws` -- the untrained-model baseline.
      hidden_blocking    over pairs of conflicting lectures of different
                         courses, the mean share of periods blocked (H4)
                         for exactly one of the two. trm_ctt.py applies
                         availability only when it masks its output, so
                         this is coordination information neither lecture
                         can see while reasoning.
      tag_auc            how well the dot product of two courses' group tags
                         (ctt_encode.py: summed random curriculum + teacher
                         tags) separates conflicting from non-conflicting
                         course pairs, as ROC AUC averaged over `draws` tag
                         draws. 0.5 = chance, 1 = perfectly visible.
      mean_local_load,   per lecture: (1 + lectures it conflicts with) /
      max_local_load     its available periods -- how crowded its options
                         are; mean and max over lectures.
      repair_steps       moves of a min-conflicts local search from a random
                         timetable to one without hard violations, mean of
                         `repair_runs` runs, capped at `repair_cap` -- a
                         rough stand-in for how many refinement steps
                         placing the lectures jointly takes.
    """
    from scipy.stats import rankdata

    from dataset.ctt_encode import N_GROUP_TAG_DIMS
    from dataset.ctt_solver import evaluate_assignment

    rng = np.random.default_rng(seed)
    lectures, neighbors = _lecture_conflicts(problem)
    available = _available_periods(problem, lectures)
    room_ids = list(problem.rooms)

    random_guess_hard = 0.0
    for _ in range(draws):
        period_of = {lec: int(rng.choice(available[i])) for i, lec in enumerate(lectures)}
        room_of = {lec: room_ids[int(rng.integers(len(room_ids)))] for lec in lectures}
        random_guess_hard += evaluate_assignment(problem, period_of, room_of)["hard_violations"]

    unavailable = _unavailable_periods(problem)
    blocked_for_one = [len(unavailable.get(lectures[i][0], set()) ^ unavailable.get(lectures[j][0], set()))
                       for i, nb in enumerate(neighbors) for j in nb if j > i and lectures[j][0] != lectures[i][0]]
    hidden_blocking = np.mean(blocked_for_one) / problem.n_periods if blocked_for_one else 0.0

    course_ids = list(problem.courses)
    conflicting = set()
    for g in _conflict_groups(problem):
        conflicting.update(itertools.combinations(sorted(set(g)), 2))
    pairs = list(itertools.combinations(sorted(course_ids), 2))
    positive = np.array([pair in conflicting for pair in pairs])
    course_pos = {cid: k for k, cid in enumerate(sorted(course_ids))}
    first = np.array([course_pos[a] for a, _b in pairs], dtype=np.int64)
    second = np.array([course_pos[b] for _a, b in pairs], dtype=np.int64)
    if positive.any() and not positive.all():
        aucs = []
        for _ in range(draws):
            tag = {qid: rng.standard_normal(N_GROUP_TAG_DIMS) for qid in problem.curricula}
            tag.update({c.teacher_id: rng.standard_normal(N_GROUP_TAG_DIMS) for c in problem.courses.values()})
            course_tag = np.zeros((len(course_ids), N_GROUP_TAG_DIMS))
            for qid, q in problem.curricula.items():
                for cid in q.course_ids:
                    course_tag[course_pos[cid]] += tag[qid]
            for cid, c in problem.courses.items():
                course_tag[course_pos[cid]] += tag[c.teacher_id]
            ranks = rankdata(np.einsum("ij,ij->i", course_tag[first], course_tag[second]))
            n_pos, n_neg = positive.sum(), (~positive).sum()
            aucs.append((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        tag_auc = float(np.mean(aucs))
    else:
        tag_auc = float("nan")  # every pair, or no pair, conflicts: nothing to separate

    local_load = [(1 + len(nb)) / len(options) for nb, options in zip(neighbors, available)]
    repair_steps = np.mean([_min_conflicts_steps(available, neighbors, problem.n_periods, len(room_ids), rng, repair_cap)
                            for _ in range(repair_runs)])
    return dict(
        random_guess_hard=random_guess_hard / draws / len(lectures),
        hidden_blocking=float(hidden_blocking),
        tag_auc=tag_auc,
        mean_local_load=float(np.mean(local_load)),
        max_local_load=float(np.max(local_load)),
        repair_steps=float(repair_steps),
    )


def network_measures(problem: CTTProblem, solutions: Optional[Sequence[Dict[str, dict]]] = None,
                     seed: int = 0) -> Dict[str, float]:
    """input_measures(), plus ambiguity_measures() when a pool of equally
    good `solutions` is given (NaN otherwise) -- every NN_FEATURES key."""
    measures = dict.fromkeys(NN_FEATURES, float("nan"))
    measures.update(input_measures(problem, seed))
    if solutions:
        measures.update(ambiguity_measures(problem, solutions))
    return measures


def profile_problem(problem: CTTProblem, name: str, solver: str = "none", time_limit: float = 60.0,
                    workers: int = 1) -> Dict[str, object]:
    """One flat row per instance: `name`, structural_features(), and for
    each solver picked by `solver` ("none", "cpsat", "milp" or "both")
    solver_measures() with keys prefixed by the solver name (cpsat_gap,
    milp_solve_time, ...)."""
    if solver not in SOLVER_CHOICES:
        raise ValueError(f"unknown solver option {solver!r}, expected one of {list(SOLVER_CHOICES)}")
    row: Dict[str, object] = dict(name=name, **structural_features(problem))
    if SOLVER_CHOICES[solver]:
        row["time_limit"] = time_limit
    for s in SOLVER_CHOICES[solver]:
        row.update({f"{s}_{k}": v for k, v in solver_measures(problem, s, time_limit, workers).items()})
    return row


def instance_name(ctt_path: str) -> str:
    """Named by file (comp01, gen000, ...) -- real instances' own Name:
    field is e.g. "Fis0506-1"."""
    return os.path.splitext(os.path.basename(ctt_path))[0]


def profile_instance(ctt_path: str, solver: str = "none", time_limit: float = 60.0,
                     workers: int = 1) -> Dict[str, object]:
    """profile_problem() on a .ctt file, named by instance_name()."""
    return profile_problem(parse_ctt(ctt_path), instance_name(ctt_path), solver, time_limit, workers)


def _solve_single_threaded(problem: CTTProblem, solver: str, time_limit: float) -> Dict[str, object]:
    """profile_parallel()'s pool task (module-level so it pickles)."""
    return solver_measures(problem, solver, time_limit, workers=1)


def _network_single_threaded(problem: CTTProblem, time_limit: float, pool_solutions: int) -> Dict[str, object]:
    """profile_parallel()'s pool task for network=True: a single-threaded
    solve_pool() of up to `pool_solutions` equally good solutions (ties at
    the best objective found, optimal or not), then network_measures().
    n_optima is a lower bound unless optima_complete."""
    from dataset.ctt_solver_milp import solve_pool  # gurobipy -- needs a license, see ctt_solver_milp.py
    pool = solve_pool(problem, time_limit=time_limit, workers=1, pool_solutions=pool_solutions)
    measures = network_measures(problem, pool["solutions"])
    measures.update(n_optima=len(pool["solutions"]), optima_complete=pool["exhaustive"])
    return measures


def profile_parallel(named_problems: List[Tuple[str, CTTProblem]], solver: str = "both",
                     time_limit: float = 60.0, cpus: Optional[int] = None,
                     progress: bool = True, network: bool = False,
                     pool_solutions: int = 1000) -> List[Dict[str, object]]:
    """profile_problem() for many (name, problem) pairs at once: every
    (instance, solver) run is its own single-threaded process, up to `cpus`
    (default: all logical CPUs) at a time. Largest instances start first, so
    a long run doesn't start last while the other CPUs sit idle. Rows come
    back in input order. network=True adds the network_measures() columns
    (prefixed "nn_"), each instance's solution pool being one more
    single-threaded task (see _network_single_threaded()).

    Parallel runs share memory bandwidth, and on hybrid CPUs (performance +
    efficiency cores) land on cores of different speed, so solve_time is
    noisier than for one run at a time -- and a run on a slow core gets less
    search done within the same time_limit."""
    if solver not in SOLVER_CHOICES:
        raise ValueError(f"unknown solver option {solver!r}, expected one of {list(SOLVER_CHOICES)}")
    feature_rows = [profile_problem(problem, name) for name, problem in named_problems]  # milliseconds, in-process
    kinds = SOLVER_CHOICES[solver] + (("nn",) if network else ())
    if not kinds:
        return feature_rows

    tasks = sorted(((i, kind) for i in range(len(named_problems)) for kind in kinds),
                   key=lambda task: -feature_rows[task[0]]["n_lectures"])
    results: Dict[Tuple[int, str], Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=cpus or os.cpu_count()) as pool:
        futures = {}
        for i, kind in tasks:
            problem = named_problems[i][1]
            if kind == "nn":
                futures[pool.submit(_network_single_threaded, problem, time_limit, pool_solutions)] = (i, kind)
            else:
                futures[pool.submit(_solve_single_threaded, problem, kind, time_limit)] = (i, kind)
        for done, fut in enumerate(as_completed(futures), start=1):
            i, kind = futures[fut]
            measures = results[(i, kind)] = fut.result()
            if not progress:
                continue
            if kind == "nn":
                print(f"[{done}/{len(tasks)}] {feature_rows[i]['name']} nn: {measures['n_optima']} optima, "
                      f"plausible periods {measures['plausible_periods']:.1f}", flush=True)
            else:
                gap = "-" if measures["gap"] is None else f"{measures['gap']:.3f}"
                print(f"[{done}/{len(tasks)}] {feature_rows[i]['name']} {kind}: {measures['status']} "
                      f"{measures['solve_time']:.1f}s gap={gap}", flush=True)

    rows = []
    for i, row in enumerate(feature_rows):
        row = dict(row, time_limit=time_limit)
        for kind in kinds:
            row.update({f"{kind}_{k}": v for k, v in results[(i, kind)].items()})
        rows.append(row)
    return rows


def generate_instances(count: int, size: str = "small", seed: int = 0,
                       out_dir: Optional[str] = None) -> List[CTTProblem]:
    """`count` fresh instances from dataset.ctt_generator.generate_problem()
    at `size` ("small", "medium" or "large"), named f"{size}{i:03d}". Unlike
    the generator's own CLI nothing is solved here -- that's what
    profile_problem()'s `solver` option is for. Each attempt gets its own rng
    stream from (seed, size, attempt), so sizes don't share random draws;
    attempts the generator gives up on are skipped. out_dir: also write each
    instance there as <name>.ctt."""
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}, expected one of {list(SIZES)}")
    size_index = list(SIZES).index(size)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    problems: List[CTTProblem] = []
    for attempt in range(100 * count):
        if len(problems) == count:
            break
        name = f"{size}{len(problems):03d}"
        try:
            problem = generate_problem(np.random.default_rng([seed, size_index, attempt]), name, SIZES[size])
        except RuntimeError:  # generator's cheap feasibility pre-checks kept failing
            continue
        problems.append(problem)
        if out_dir:
            write_ctt(problem, os.path.join(out_dir, f"{name}.ctt"))
    if len(problems) < count:
        raise RuntimeError(f"generated only {len(problems)}/{count} {size} instances in {100 * count} attempts")
    return problems


def _num(row: Dict[str, object], key: str) -> Optional[float]:
    """Rows read back from CSV hold strings, with "" for None."""
    v = row.get(key)
    return None if v is None or v == "" else float(v)


def _time_and_gap(row: Dict[str, object], solvers: Sequence[str]) -> Tuple[float, float]:
    """Time to prove optimality (censored: an unproven run counts as the
    full time limit) and gap, for one solver -- or, given several, for their
    "virtual best": the fastest proof, and the best objective against the
    best bound any of them found."""
    limit = _num(row, "time_limit")
    times = [_num(row, f"{s}_solve_time") if row[f"{s}_status"] == "OPTIMAL" else limit for s in solvers]
    objectives = [o for o in (_num(row, f"{s}_objective") for s in solvers) if o is not None]
    bounds = [b for b in (_num(row, f"{s}_best_bound") for s in solvers) if b is not None]
    proven = any(row[f"{s}_status"] == "OPTIMAL" for s in solvers)
    gap = _gap(min(objectives) if objectives else None, max(bounds) if bounds else None, proven)
    return min(times), gap


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman correlation of x and y after regressing the ranks of both on
    the ranks of `control` -- i.e. what x says about y beyond `control`."""
    from scipy.stats import rankdata
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), rankdata(control)])
    residual_x = rankdata(x) - design @ np.linalg.lstsq(design, rankdata(x), rcond=None)[0]
    residual_y = rankdata(y) - design @ np.linalg.lstsq(design, rankdata(y), rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out R^2 of a linear fit (via the hat matrix) -- unlike plain
    R^2 it doesn't improve just because more features were added."""
    design = np.column_stack([np.ones(len(y)), X])
    if len(y) <= design.shape[1] + 1 or np.ptp(y) == 0:
        return float("nan")  # too few instances to fit and hold one out, or nothing to explain
    hat = design @ np.linalg.pinv(design)
    loo_residual = (y - hat @ y) / np.clip(1 - np.diag(hat), 1e-9, None)
    return float(1 - np.sum(loo_residual ** 2) / np.sum((y - y.mean()) ** 2))


def analyze(rows: List[Dict[str, object]]) -> None:
    """Prints, for each solver present in `rows` (plus their virtual best if
    both ran), how each structural feature relates to solve time and gap:

      rho          Spearman rank correlation with time / gap.
      rho|size     the same after removing what n_lectures already
                   explains (partial Spearman) -- the "beyond size" part.
      LOO R^2      how well a linear fit predicts log10(time) / gap from
                   log(n_lectures) alone vs. from all features.

    Time is time to *prove* optimality, censored at the time limit, so
    instances no solver finishes all tie at the limit. Proven-infeasible
    instances are excluded.
    """
    present = [s for s in SOLVERS if f"{s}_status" in rows[0]]
    if not present:
        print("no solver columns to analyze -- run with --solver cpsat/milp/both")
        return
    blocks = [(s, [s]) for s in present] + ([("virtual best (cpsat+milp)", present)] if len(present) > 1 else [])

    for label, solvers in blocks:
        kept = [r for r in rows if all(r[f"{s}_status"] != "INFEASIBLE" for s in solvers)]
        if len(kept) < 4:
            print(f"\n== {label}: only {len(kept)} usable instances, skipping")
            continue
        time, gap = map(np.array, zip(*(_time_and_gap(r, solvers) for r in kept)))
        X = np.array([[_num(r, f) for f in FEATURES] for r in kept])
        size = X[:, FEATURES.index("n_lectures")]
        X_log = X.copy()
        X_log[:, FEATURES.index("n_lectures")] = np.log(size)
        log_time = np.log10(np.clip(time, 1e-2, None))

        statuses = [r[f"{s}_status"] for r in kept for s in solvers]
        counts = ", ".join(f"{statuses.count(k)} {k.lower()}" for k in ("OPTIMAL", "FEASIBLE", "NO_SOLUTION"))
        limits = {_num(r, "time_limit") for r in kept}
        print(f"\n== {label}: {len(kept)} instances ({len(rows) - len(kept)} infeasible excluded); "
              f"runs: {counts}; time limit {', '.join(f'{t:g}s' for t in sorted(limits))}")
        print(f"   median time {np.median(time):.1f}s, median gap {np.median(gap):.3f}")
        print(f"   {'feature':22s} {'rho time':>9s} {'rho gap':>8s} {'rho|size time':>14s} {'rho|size gap':>13s}")
        for k, feature in enumerate(FEATURES):
            x = X[:, k]
            if feature == "n_lectures":
                partial = "--"
                print(f"   {feature:22s} {_spearman(x, time):+9.2f} {_spearman(x, gap):+8.2f} {partial:>14s} {partial:>13s}")
            else:
                print(f"   {feature:22s} {_spearman(x, time):+9.2f} {_spearman(x, gap):+8.2f} "
                      f"{_partial_spearman(x, time, size):+14.2f} {_partial_spearman(x, gap, size):+13.2f}")
        size_only = X_log[:, [FEATURES.index("n_lectures")]]
        print(f"   LOO R^2  log10(time): size only {_loo_r2(size_only, log_time):.2f}, "
              f"all features {_loo_r2(X_log, log_time):.2f}   |   "
              f"gap: size only {_loo_r2(size_only, gap):.2f}, all features {_loo_r2(X_log, gap):.2f}")


NETWORK_OUTCOMES = ("hard_per_lecture", "learning_curve", "period_loss", "recursion_gain")


def network_outcomes(log_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """How the network did on each (eval set, instance) of one
    pretrain_ctt.py training.instance_log:

      hard_per_lecture  hard violations per lecture at the last eval.
      learning_curve    hard violations per lecture averaged over every eval
                        after step 0 -- low = learned early and well.
      period_loss       period loss at the last eval.
      recursion_gain    hard violations per lecture after the first ACT step
                        minus after the last, at the last eval; > 0 means
                        the recursion helps on this instance.
    """
    by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in log_rows:
        by_key.setdefault((r["set"], r["instance"]), []).append(r)
    outcomes = {}
    for key, rs in by_key.items():
        rs.sort(key=lambda r: int(r["step"]))
        last, n = rs[-1], float(rs[-1]["n_lectures"])
        trained = [r for r in rs if int(r["step"]) > 0] or rs
        outcomes[key] = dict(
            hard_per_lecture=float(last["hard_violations"]) / n,
            learning_curve=float(np.mean([float(r["hard_violations"]) for r in trained])) / n,
            period_loss=float(last["period_loss"]),
            recursion_gain=(float(last["hard_step1"]) - float(last["hard_violations"])) / n,
        )
    return outcomes


def _critical_rho(n: int, tests: int = 1, alpha: float = 0.05) -> float:
    """|rho| needed for two-sided significance at alpha / tests (Bonferroni)."""
    from scipy.stats import t
    t_crit = t.ppf(1 - alpha / (2 * tests), n - 2)
    return float(t_crit / np.sqrt(n - 2 + t_crit ** 2))


def analyze_network(rows: List[Dict[str, object]], logs: List[List[Dict[str, str]]]) -> None:
    """Relates every measure in `rows` (structural features, MILP time and
    gap if present, nn_ columns) to how the network did per instance, from
    one or more pretrain_ctt.py instance logs (e.g. training seeds), per
    eval set. With several logs the outcome is their mean, and "seed
    agreement" is the Spearman correlation between the first two logs --
    the reproducibility no measure can beat. Prints rho, rho|size (* =
    significant at 5 %, ** = also after Bonferroni over all measures) and
    the leave-one-out R^2 of hard_per_lecture from log size alone vs size
    plus each group of measures."""
    by_name = {r["name"]: r for r in rows}
    groups = {"structural": [f for f in FEATURES if f != "n_lectures"]}
    if "milp_status" in rows[0]:
        for r in rows:
            r["milp_time"] = _time_and_gap(r, ["milp"])[0]  # censored at the time limit, as in analyze()
        groups["solver"] = ["milp_time", "milp_gap"]
    groups["network"] = [f"nn_{f}" for f in NN_FEATURES if f"nn_{f}" in rows[0]]
    predictors = [p for g in groups.values() for p in g]
    per_log = [network_outcomes(log) for log in logs]

    common = set.intersection(*(set(o) for o in per_log))
    for set_name in sorted({s for s, _name in common}):
        keys = sorted(k for k in common if k[0] == set_name and k[1] in by_name)
        if len(keys) < 10:
            continue
        size = np.array([_num(by_name[k[1]], "n_lectures") for k in keys])
        X = {p: np.array([np.nan if _num(by_name[k[1]], p) is None else _num(by_name[k[1]], p) for k in keys])
             for p in predictors}
        n = len(keys)
        print(f"\n== eval set {set_name!r}: {n} instances, {len(logs)} run(s); significant |rho| >= "
              f"{_critical_rho(n):.2f} (5 %), {_critical_rho(n, len(predictors) + 1):.2f} (Bonferroni)")
        if len(logs) > 1:
            print("   (outcomes are averaged over the runs; seed agreement caps how well any measure can do)")
        outcome = {}
        for name in NETWORK_OUTCOMES:
            runs = [np.array([o[k][name] for k in keys]) for o in per_log]
            outcome[name] = np.mean(runs, axis=0)
            agreement = f"seed agreement {_spearman(runs[0], runs[1]):+.2f}" if len(runs) > 1 else ""
            print(f"   {name:18s} median {np.median(outcome[name]):.3f}  {agreement}")

        critical, critical_bonferroni = _critical_rho(n), _critical_rho(n, len(predictors) + 1)

        def mark(rho: float) -> str:
            if np.isnan(rho) or abs(rho) < critical:
                return "  "
            return "**" if abs(rho) >= critical_bonferroni else "* "

        print(f"   {'measure':24s}" + "".join(f" {name[:16]:>16s} {'|size':>7s}" for name in NETWORK_OUTCOMES))
        for p in ["n_lectures"] + predictors:
            x = size if p == "n_lectures" else X[p]
            ok = ~np.isnan(x)
            cells = ""
            for name in NETWORK_OUTCOMES:
                rho = _spearman(x[ok], outcome[name][ok])
                partial = float("nan") if p == "n_lectures" else _partial_spearman(x[ok], outcome[name][ok], size[ok])
                cells += (f" {rho:+14.2f}{mark(rho)}"
                          f" {'--' if np.isnan(partial) else f'{partial:+.2f}':>5s}{mark(partial)}")
            print(f"   {p:24s}{cells}")

        y = outcome["hard_per_lecture"]
        base = np.log(size)[:, None]
        filled = {p: np.where(np.isnan(v), np.nanmean(v), v) for p, v in X.items()}
        r2 = {"size only": _loo_r2(base, y)}
        for g, members in groups.items():
            r2[f"+{g}"] = _loo_r2(np.column_stack([base] + [filled[p] for p in members]), y)
        r2["+all"] = _loo_r2(np.column_stack([base] + [filled[p] for p in predictors]), y)
        print("   LOO R^2 hard_per_lecture: " + ", ".join(f"{g} {v:.2f}" for g, v in r2.items()))


def _expand(patterns: List[str]) -> List[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no .ctt files matched {pattern!r}")
        paths.extend(matches)
    return paths


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Difficulty measures for .ctt instances: structural features, "
                                              "optionally solver runs, and an analysis relating the two.")
    cli.add_argument("ctt_paths", nargs="*", help=".ctt files or glob patterns, e.g. 'data/ITC2007/real/comp*.ctt'")
    cli.add_argument("--solver", choices=list(SOLVER_CHOICES), default="none",
                     help="none = structural features only; cpsat = dataset.ctt_solver (ortools); "
                          "milp = dataset.ctt_solver_milp (gurobipy, needs a Gurobi license); both = run both")
    cli.add_argument("--time-limit", type=float, default=60.0, help="per solver, per instance (s)")
    cli.add_argument("--cpus", type=int, default=None,
                     help=f"max single-threaded solver runs at once (default: all {os.cpu_count()} logical CPUs; "
                          f"more = faster, but noisier solve times)")
    cli.add_argument("--out", default=None, help="optional CSV path, one row per instance")
    cli.add_argument("--analyze", default=None, metavar="CSV",
                     help="analyze an existing --out CSV instead of profiling instances")
    cli.add_argument("--generate", type=int, default=0, metavar="N",
                     help="also profile N freshly generated instances per --size (dataset.ctt_generator)")
    cli.add_argument("--size", nargs="+", choices=list(SIZES), default=None,
                     help="generator size(s) for --generate (default: small)")
    cli.add_argument("--seed", type=int, default=0, help="generator seed for --generate")
    cli.add_argument("--gen-out-dir", default=None, help="also write generated instances here as .ctt files")
    cli.add_argument("--nn", action="store_true",
                     help="also compute the network-specific measures (network_measures(); needs gurobipy for "
                          "the solution pool, which gets --time-limit)")
    cli.add_argument("--pool-solutions", type=int, default=1000, help="pool size cap for --nn")
    cli.add_argument("--instance-log", nargs="+", default=None, metavar="LOG",
                     help="with --analyze: relate the measures to how the network did instead, from "
                          "pretrain_ctt.py training.instance_log CSV(s), e.g. one per training seed")
    args = cli.parse_args()

    if args.analyze:
        with open(args.analyze, newline="") as f:
            rows = list(csv.DictReader(f))
        if args.instance_log:
            logs = []
            for path in args.instance_log:
                with open(path, newline="") as f:
                    logs.append(list(csv.DictReader(f)))
            analyze_network(rows, logs)
        else:
            analyze(rows)
    else:
        if args.size and not args.generate:
            cli.error("--size only applies together with --generate N")
        if not args.ctt_paths and not args.generate:
            cli.error("give .ctt paths/patterns and/or --generate N, or --analyze CSV")
        named_problems = [(instance_name(path), parse_ctt(path)) for path in _expand(args.ctt_paths)]
        for size in args.size or ["small"]:
            if args.generate:
                named_problems += [(problem.name, problem)
                                   for problem in generate_instances(args.generate, size, args.seed, args.gen_out_dir)]
        rows = profile_parallel(named_problems, args.solver, args.time_limit, args.cpus,
                                network=args.nn, pool_solutions=args.pool_solutions)

        if not SOLVER_CHOICES[args.solver] and not args.nn:
            keys = list(rows[0].keys())
            print("  ".join(f"{k:>{len(k)}s}" for k in keys))
            for row in rows:
                print("  ".join(f"{v:>{len(k)}.3f}" if isinstance(v, float) else f"{v!s:>{len(k)}s}"
                                for k, v in row.items()))

        if args.out:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {len(rows)} rows to {args.out}")

        if SOLVER_CHOICES[args.solver]:
            analyze(rows)
