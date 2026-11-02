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

    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    groups = [q.course_ids for q in problem.curricula.values()] + list(teacher_courses.values())

    max_group_load = max(sum(courses[cid].n_lectures for cid in g) for g in groups) / n_periods

    curricula_per_course = sum(len(q.course_ids) for q in problem.curricula.values()) / len(courses)

    conflicting = set()
    for g in groups:
        conflicting.update(itertools.combinations(sorted(set(g)), 2))
    n_pairs = len(courses) * (len(courses) - 1) // 2
    conflict_density = len(conflicting) / n_pairs if n_pairs else 0.0

    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * problem.periods_per_day + u.period)
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


def profile_parallel(named_problems: List[Tuple[str, CTTProblem]], solver: str = "both",
                     time_limit: float = 60.0, cpus: Optional[int] = None,
                     progress: bool = True) -> List[Dict[str, object]]:
    """profile_problem() for many (name, problem) pairs at once: every
    (instance, solver) run is its own single-threaded process, up to `cpus`
    (default: all logical CPUs) at a time. Largest instances start first, so
    a long run doesn't start last while the other CPUs sit idle. Rows come
    back in input order.

    Parallel runs share memory bandwidth, and on hybrid CPUs (performance +
    efficiency cores) land on cores of different speed, so solve_time is
    noisier than for one run at a time -- and a run on a slow core gets less
    search done within the same time_limit."""
    if solver not in SOLVER_CHOICES:
        raise ValueError(f"unknown solver option {solver!r}, expected one of {list(SOLVER_CHOICES)}")
    feature_rows = [profile_problem(problem, name) for name, problem in named_problems]  # milliseconds, in-process
    if not SOLVER_CHOICES[solver]:
        return feature_rows

    tasks = sorted(((i, s) for i in range(len(named_problems)) for s in SOLVER_CHOICES[solver]),
                   key=lambda task: -feature_rows[task[0]]["n_lectures"])
    results: Dict[Tuple[int, str], Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=cpus or os.cpu_count()) as pool:
        futures = {pool.submit(_solve_single_threaded, named_problems[i][1], s, time_limit): (i, s) for i, s in tasks}
        for done, fut in enumerate(as_completed(futures), start=1):
            i, s = futures[fut]
            measures = results[(i, s)] = fut.result()
            if progress:
                gap = "-" if measures["gap"] is None else f"{measures['gap']:.3f}"
                print(f"[{done}/{len(tasks)}] {feature_rows[i]['name']} {s}: {measures['status']} "
                      f"{measures['solve_time']:.1f}s gap={gap}", flush=True)

    rows = []
    for i, row in enumerate(feature_rows):
        row = dict(row, time_limit=time_limit)
        for s in SOLVER_CHOICES[solver]:
            row.update({f"{s}_{k}": v for k, v in results[(i, s)].items()})
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
    args = cli.parse_args()

    if args.analyze:
        with open(args.analyze, newline="") as f:
            analyze(list(csv.DictReader(f)))
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
        rows = profile_parallel(named_problems, args.solver, args.time_limit, args.cpus)

        if not SOLVER_CHOICES[args.solver]:
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
