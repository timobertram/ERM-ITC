"""Fix-and-optimize matheuristic for ITC2007 Track 3 (CB-CTT), via gurobipy.

Reuses dataset.ctt_solver_milp.build_model() verbatim (same x[l,p,r] MILP,
same H1-H4/S1-S4 semantics) but instead of handing the whole model to Gurobi
in one solve, repeatedly fixes most lectures to their current incumbent
assignment and re-optimizes only a small free "neighborhood" of lectures at
a time. This is the core technique behind Mikkelsen, Holm, Sorensen &
Stidsen's winning entry to the 2019 International Timetabling Competition
(ITC2019) -- see their PhD thesis / "A Parallelized Matheuristic for the
International Timetabling Competition 2019". Their own numbers are the
reason this exists: on their largest instances the full MIP has up to 10
million constraints/variables and "cannot find any solutions... within 24
hours", while fix-and-optimize keeps making steady progress by only ever
solving small subproblems.

Adaptations from their setup to ours:
  - Their two variable-granularities for fixing were x_{c,t,r} (time+room)
    vs y_{c,t} (time only, room left free); they found x_{c,t,r} fixing
    "consistently better", so that's the only granularity implemented here
    -- a free lecture's x[l,p,r] is unfixed across all (p,r), a fixed
    lecture is pinned to exactly its incumbent (p,r).
  - Their "Common-Distribution-Constraint" and "Adjacent-Classes"
    neighborhood heuristics both grow the free set along problem structure
    (classes sharing a distribution constraint, or adjacent in a conflict
    graph). CB-CTT only has two such relations -- curriculum (H3a) and
    teacher (H3b) -- so those collapse into one "linked" neighborhood mode
    here instead of their three.
  - Not implemented: their solution-sharing across several parallel
    fix-and-optimize processes, and the 12-hour diversification/restart
    scheme. This is the single-process core of their method (adaptive
    neighborhood sizing + fixing, plus an optional final full-model bound
    check -- see bound_time_limit below), not their full parallelized
    system -- a reasonable next step, not reproduced here to keep this a
    single, readable solver.

This is fundamentally a heuristic: no sequence of restricted-neighborhood
subproblem solves can ever prove global optimality, since each one only
searches a small subset of the full assignment space -- there is always an
untried combination that could beat the incumbent. Two optional phases are
offered on top of the main loop, both borrowed from Mikkelsen & Holm's own
setup (they ran the full MIP in parallel purely for lower-bound info, and
Gurobi's Solution Pool is exactly the mechanism ctt_solver_milp.solve_pool()
already uses for exact ties):

  - bound_time_limit > 0: after the main loop, unfix everything and give
    the FULL model (warm-started from the heuristic's best incumbent) a
    time budget to search unrestricted. If it reaches GRB.OPTIMAL, that IS
    a genuine global optimality proof (status becomes "OPTIMAL"); if it
    only hits its time limit, its ObjBound is still a valid lower bound to
    report a gap against. This is the *only* way to get either -- and
    it's most likely to succeed precisely on the instances where you
    didn't need this matheuristic in the first place, since it's exactly
    the same full-model solve that's slow on the large ones.

  - pool_solutions > 0: after the main loop converges on best_obj, harvest
    up to pool_solutions distinct global assignments tied at that exact
    value. Rather than pooling the whole model (which is exactly what's
    intractable on the instances this file targets), each harvest round
    fixes everything but a small random neighborhood, adds a hard
    objective == best_obj constraint, and runs Gurobi's Solution Pool
    (PoolSearchMode=2) on just that small subproblem -- each alternate
    filling of the neighborhood, recombined with the fixed rest, is a
    genuinely different full assignment at the same objective.
"""
import argparse
import json
import random
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from dataset.ctt_parser import CTTProblem, parse_ctt
from dataset.ctt_solver_milp import build_model, _extract_assignment

Lecture = Tuple[str, int]


def _course_adjacency(problem: CTTProblem) -> Dict[str, set]:
    """Courses linked by a hard "must differ" relation (H3a/H3b) -- our
    analogue of Mikkelsen & Holm's distribution-constraint/conflict-graph
    adjacency, used by the "linked" neighborhood mode."""
    adjacency: Dict[str, set] = {cid: set() for cid in problem.courses}
    for q in problem.curricula.values():
        for c1 in q.course_ids:
            adjacency[c1].update(c2 for c2 in q.course_ids if c2 != c1)
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    for cids in teacher_courses.values():
        for c1 in cids:
            adjacency[c1].update(c2 for c2 in cids if c2 != c1)
    return adjacency


def _pick_free_lectures(lectures_by_course: Dict[str, List[Lecture]], target_count: int,
                         linked: bool, rng: random.Random, adjacency: Dict[str, set]) -> set:
    """Grows a free set course-by-course (mirrors Mikkelsen & Holm: they
    extract *all* classes of a chosen course, never a partial course) until
    target_count lectures are free. linked=True also pulls in every course
    adjacent to a chosen one (shares a curriculum or teacher), matching
    their Common-Distribution-Constraint / Adjacent-Classes heuristics."""
    course_ids = list(lectures_by_course.keys())
    rng.shuffle(course_ids)
    chosen: set = set()
    free: List[Lecture] = []
    for cid in course_ids:
        if len(free) >= target_count:
            break
        if cid in chosen:
            continue
        chosen.add(cid)
        free.extend(lectures_by_course[cid])
        if linked:
            for c2 in adjacency.get(cid, ()):
                if c2 not in chosen:
                    chosen.add(c2)
                    free.extend(lectures_by_course[c2])
    return set(free)


def _current_pr(vars_by_lecture: Dict[Lecture, List[Tuple[int, str, gp.Var]]],
                 value_attr: str = "X") -> Dict[Lecture, Tuple[int, str]]:
    """value_attr='X' reads the incumbent; 'Xn' reads whichever pool solution
    m.Params.SolutionNumber currently points at (see solve_pool() in
    ctt_solver_milp.py for the same pattern)."""
    pr = {}
    for lec, entries in vars_by_lecture.items():
        for p, rid, v in entries:
            if getattr(v, value_attr) > 0.5:
                pr[lec] = (p, rid)
                break
    return pr


def _apply_fixing(vars_by_lecture: Dict[Lecture, List[Tuple[int, str, gp.Var]]],
                   free_set: set, incumbent: Dict[Lecture, Tuple[int, str]]) -> None:
    for lec, entries in vars_by_lecture.items():
        if lec in free_set:
            for _, _, v in entries:
                v.LB, v.UB = 0, 1
        else:
            target = incumbent[lec]
            for p, rid, v in entries:
                on = (p, rid) == target
                v.LB = v.UB = 1 if on else 0


def _prove_or_bound(m: gp.Model, vars_by_lecture: Dict[Lecture, List[Tuple[int, str, gp.Var]]],
                     incumbent: Dict[Lecture, Tuple[int, str]], best_obj: float,
                     bound_time_limit: float, workers: int, log_progress: bool):
    """Unfixes every variable and gives the full model bound_time_limit
    seconds, warm-started from the heuristic's incumbent. Returns
    (best_obj, incumbent, best_bound, proved_optimal) -- best_obj/incumbent
    are updated in place if the unrestricted search finds something better
    (it might: this is a genuinely larger search than any single
    fix-and-optimize iteration)."""
    for entries in vars_by_lecture.values():
        for _, _, v in entries:
            v.LB, v.UB = 0, 1
    for lec, (p, rid) in incumbent.items():
        for pp, rr, v in vars_by_lecture[lec]:
            v.Start = 1.0 if (pp, rr) == (p, rid) else 0.0

    m.Params.TimeLimit = bound_time_limit
    m.Params.Threads = workers
    m.optimize()

    proved_optimal = m.Status == GRB.OPTIMAL
    best_bound = m.ObjBound if m.SolCount > 0 or proved_optimal else None

    if m.SolCount > 0 and m.ObjVal < best_obj - 1e-6:
        best_obj = m.ObjVal
        incumbent = _current_pr(vars_by_lecture, "X")
        if log_progress:
            print(f"[bound-check] unrestricted search found a better solution: {best_obj}")
    if log_progress:
        print(f"[bound-check] best_bound={best_bound} proved_optimal={proved_optimal}")

    return best_obj, incumbent, best_bound, proved_optimal


def _harvest_ties(m: gp.Model, vars_by_lecture: Dict[Lecture, List[Tuple[int, str, gp.Var]]],
                   lectures_by_course: Dict[str, List[Lecture]], neighborhood: str,
                   adjacency: Dict[str, set], incumbent: Dict[Lecture, Tuple[int, str]],
                   best_obj: float, pool_solutions: int, pool_time_limit: float,
                   iter_time_limit: float, workers: int, rng: random.Random,
                   log_progress: bool) -> List[Dict[Lecture, Tuple[int, str]]]:
    """Harvests up to pool_solutions distinct full assignments tied exactly
    at best_obj, by running Gurobi's Solution Pool on a sequence of small
    subproblems instead of the whole model (see module docstring)."""
    total_lectures = sum(len(lecs) for lecs in lectures_by_course.values())
    objective_expr = m.getObjective()
    tie_constr = m.addConstr(objective_expr == best_obj, name="pool_tie")
    m.Params.PoolSearchMode = 2
    m.Params.PoolGapAbs = 0.0
    m.Params.PoolSolutions = pool_solutions

    solutions: List[Dict[Lecture, Tuple[int, str]]] = [dict(incumbent)]
    seen = {tuple(sorted(incumbent.items()))}
    t0 = time.time()

    while len(solutions) < pool_solutions and time.time() - t0 < pool_time_limit:
        target_count = max(1, round(0.15 * total_lectures))
        free_lectures = _pick_free_lectures(lectures_by_course, target_count, neighborhood == "linked", rng, adjacency)
        _apply_fixing(vars_by_lecture, free_lectures, incumbent)

        remaining = pool_time_limit - (time.time() - t0)
        m.Params.TimeLimit = max(1.0, min(iter_time_limit, remaining))
        m.optimize()

        if m.SolCount == 0:
            continue
        for i in range(m.SolCount):
            m.Params.SolutionNumber = i
            if abs(m.PoolObjVal - best_obj) > 1e-6:
                continue
            snap = _current_pr(vars_by_lecture, "Xn")
            key = tuple(sorted(snap.items()))
            if key not in seen:
                seen.add(key)
                solutions.append(snap)
                if log_progress:
                    print(f"[harvest] found tied solution #{len(solutions)}")
                if len(solutions) >= pool_solutions:
                    break

    m.remove(tie_constr)
    m.Params.PoolSearchMode = 0
    m.Params.Threads = workers
    return solutions


def _assignment_from_pr(problem: CTTProblem, pr: Dict[Lecture, Tuple[int, str]]) -> dict:
    """Same JSON shape as ctt_solver_milp._extract_assignment, but built
    from an in-memory (period, room) snapshot instead of live Gurobi
    variables -- needed for harvested pool solutions, which are no longer
    the model's current incumbent by the time we're done collecting them."""
    assignment = {}
    for (cid, li), (p, rid) in pr.items():
        assignment[f"{cid}#{li}"] = {
            "day": p // problem.periods_per_day,
            "period_in_day": p % problem.periods_per_day,
            "period": p,
            "room": rid,
        }
    return assignment


def solve_fix_and_optimize(
    problem: CTTProblem,
    time_limit: float = 600.0,
    iter_time_limit: float = 20.0,
    workers: int = 8,
    initial_unfix_frac: float = 0.25,
    min_unfix_frac: float = 0.05,
    max_unfix_frac: float = 0.75,
    neighborhood: str = "linked",
    patience: int = 3,
    seed: int = 0,
    log_progress: bool = False,
    bound_time_limit: float = 0.0,
    pool_solutions: int = 0,
    pool_time_limit: float = 0.0,
) -> dict:
    """Fix-and-optimize: build the full model once, then repeatedly fix all
    but a random ~unfix_frac fraction of lectures to their incumbent
    (period, room) and re-solve just that subproblem. Grows unfix_frac when
    the search stalls (several iterations with no improvement), shrinks it
    when a subproblem doesn't even finish exploring within iter_time_limit
    -- the same two adaptation triggers Mikkelsen & Holm describe.

    bound_time_limit > 0 and pool_solutions > 0 enable the two optional
    phases described in the module docstring (optimality proof/bound, and
    harvesting tied-optimal solutions) -- both disabled by default, since
    both cost extra time on top of the main search.
    """
    assert neighborhood in ("standard", "linked")
    rng = random.Random(seed)
    t0 = time.time()

    m, lectures, x, room_ids = build_model(
        problem, time_limit=iter_time_limit, workers=workers, mip_gap=0.0, log_progress=log_progress,
    )

    lectures_by_course: Dict[str, List[Lecture]] = {}
    for lec in lectures:
        lectures_by_course.setdefault(lec[0], []).append(lec)
    adjacency = _course_adjacency(problem) if neighborhood == "linked" else {}

    vars_by_lecture: Dict[Lecture, List[Tuple[int, str, gp.Var]]] = {lec: [] for lec in lectures}
    for (lec, p, rid), v in x.items():
        vars_by_lecture[lec].append((p, rid, v))

    # Initial feasible solution: solve the full (unfixed) model once.
    m.optimize()
    if m.SolCount == 0:
        return {
            "status": "INFEASIBLE", "objective": None, "best_bound": None,
            "solve_time": time.time() - t0, "n_iterations": 0, "assignment": {},
        }
    incumbent = _current_pr(vars_by_lecture, "X")
    best_obj = m.ObjVal
    if log_progress:
        print(f"[fix-and-optimize] initial solution: objective={best_obj}")

    total_lectures = len(lectures)
    unfix_frac = initial_unfix_frac
    stall = 0
    n_iter = 0

    while time.time() - t0 < time_limit:
        n_iter += 1
        target_count = max(1, round(unfix_frac * total_lectures))
        free_lectures = _pick_free_lectures(lectures_by_course, target_count, neighborhood == "linked", rng, adjacency)
        _apply_fixing(vars_by_lecture, free_lectures, incumbent)

        remaining = time_limit - (time.time() - t0)
        m.Params.TimeLimit = max(1.0, min(iter_time_limit, remaining))
        m.optimize()

        finished_subproblem = m.Status == GRB.OPTIMAL
        improved = m.SolCount > 0 and m.ObjVal < best_obj - 1e-6

        if improved:
            best_obj = m.ObjVal
            incumbent = _current_pr(vars_by_lecture, "X")
            stall = 0
        else:
            stall += 1

        if not finished_subproblem:
            unfix_frac = max(min_unfix_frac, unfix_frac * 0.5)
        elif stall >= patience:
            unfix_frac = min(max_unfix_frac, unfix_frac * 1.5)
            stall = 0

        if log_progress:
            print(f"[fix-and-optimize] iter={n_iter} free={len(free_lectures)}/{total_lectures} "
                  f"({unfix_frac:.2%}) improved={improved} best={best_obj} "
                  f"finished_subproblem={finished_subproblem}")

    tied_solutions = None
    if pool_solutions > 0 and pool_time_limit > 0:
        if log_progress:
            print(f"[harvest] searching for up to {pool_solutions} solutions tied at {best_obj}")
        tied_solutions = _harvest_ties(
            m, vars_by_lecture, lectures_by_course, neighborhood, adjacency, incumbent, best_obj,
            pool_solutions, pool_time_limit, iter_time_limit, workers, rng, log_progress,
        )

    best_bound = None
    proved_optimal = False
    if bound_time_limit > 0:
        prev_best = best_obj
        best_obj, incumbent, best_bound, proved_optimal = _prove_or_bound(
            m, vars_by_lecture, incumbent, best_obj, bound_time_limit, workers, log_progress,
        )
        if tied_solutions is not None and best_obj < prev_best - 1e-6:
            # the bound-check phase found something strictly better -- the
            # harvested ties were for the old (now suboptimal) value and no
            # longer apply.
            if log_progress:
                print("[harvest] discarding tied solutions: bound-check found a strictly better objective")
            tied_solutions = None

    # The model's live .X may not currently reflect `incumbent` (the harvest/
    # bound-check phases mutate it further, or the main loop's last iteration
    # may not have been the improving one). Re-pin everything to the tracked
    # best incumbent and re-solve (trivial: everything fixed) so
    # _extract_assignment's .X read is guaranteed correct.
    _apply_fixing(vars_by_lecture, set(), incumbent)
    m.Params.TimeLimit = 1.0
    m.optimize()

    result = {
        "status": "OPTIMAL" if proved_optimal else "FEASIBLE",
        "objective": best_obj,
        "best_bound": best_bound,  # None unless bound_time_limit > 0 -- see module docstring
        "solve_time": time.time() - t0,
        "n_iterations": n_iter,
        "assignment": _extract_assignment(problem, lectures, x, "X"),
    }
    if tied_solutions is not None:
        result["n_solutions_found"] = len(tied_solutions)
        result["solutions"] = [_assignment_from_pr(problem, pr) for pr in tied_solutions]
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("ctt_path", nargs="?", default="data/ITC2007/real/comp01.ctt")
    cli.add_argument("--time-limit", type=float, default=600.0, help="total wall-clock budget (s)")
    cli.add_argument("--iter-time-limit", type=float, default=20.0, help="per-subproblem time limit (s)")
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--initial-unfix-frac", type=float, default=0.25)
    cli.add_argument("--min-unfix-frac", type=float, default=0.05)
    cli.add_argument("--max-unfix-frac", type=float, default=0.75)
    cli.add_argument("--neighborhood", choices=["standard", "linked"], default="linked")
    cli.add_argument("--patience", type=int, default=3)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--log-progress", action="store_true")
    cli.add_argument("--out", default=None)
    cli.add_argument("--bound-time-limit", type=float, default=0.0,
                      help="seconds to spend on a final unrestricted full-model solve, to get a lower "
                           "bound / optimality proof on top of the heuristic's incumbent (0 = disabled)")
    cli.add_argument("--pool-solutions", type=int, default=0,
                      help="harvest up to this many distinct solutions tied at the best objective (0 = disabled)")
    cli.add_argument("--pool-time-limit", type=float, default=60.0,
                      help="seconds to spend on the tie-harvesting phase, if --pool-solutions > 0")
    args = cli.parse_args()

    problem = parse_ctt(args.ctt_path)
    result = solve_fix_and_optimize(
        problem, time_limit=args.time_limit, iter_time_limit=args.iter_time_limit, workers=args.workers,
        initial_unfix_frac=args.initial_unfix_frac, min_unfix_frac=args.min_unfix_frac,
        max_unfix_frac=args.max_unfix_frac, neighborhood=args.neighborhood, patience=args.patience,
        seed=args.seed, log_progress=args.log_progress, bound_time_limit=args.bound_time_limit,
        pool_solutions=args.pool_solutions, pool_time_limit=args.pool_time_limit,
    )

    print(f"status={result['status']} objective={result['objective']} best_bound={result['best_bound']} "
          f"n_iterations={result['n_iterations']} solve_time={result['solve_time']:.1f}s")
    if "n_solutions_found" in result:
        print(f"n_solutions_found={result['n_solutions_found']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote solution to {args.out}")
