"""MILP solver for ITC2007 Track 3 (Curriculum-Based Course Timetabling),
via gurobipy. Same semantics as dataset/ctt_solver.py's CP-SAT model, and a
drop-in alternative to it -- same result-dict shape from solve().

Core decision variable: x[l, p, r] = 1 iff lecture l is held in period p,
room r. This joint (period, room) variable is what makes H2 (room
occupancy) a plain linear <=1-per-(p,r) constraint instead of a bilinear
one; H1/H3a/H3b (distinct periods per course/curriculum/teacher) are then
"<=1 per period" sums over the appropriate lecture group, and H4
(availability) is enforced by simply never creating x[l, p, r] for p in
that lecture's course's unavailable periods -- mirroring how ctt_solver.py
restricts each period_var's IntVar domain directly.

Soft constraints S1-S4 follow the same formulas/weights as ctt_solver.ALPHA.
S2 (day-used) and S3 (curriculum occupancy) auxiliaries only need an
upper-bound/equality link because the objective's own pressure pulls them
up to their true value; S4 (room-used) needs a lower-bound link instead,
since the objective would otherwise be incentivized to under-report room
usage. See inline comments at each block.

solve() returns the single incumbent, matching ctt_solver.solve()'s result
shape. solve_pool() instead returns every solution tied at the optimum
(Gurobi's Solution Pool, PoolSearchMode=2) -- use it to check whether an
instance's optimum is unique, or to get several equally-good schedules to
choose from; see its docstring for the exhaustive/uniqueness semantics.

Efficiency notes -- these only speed up Python-side model *construction*,
not the actual Gurobi solve (see S4's comment below for a case where a
row-count reduction actually made the solve slower, and was reverted): H1
is dropped for any course that belongs to a curriculum, since H3a's
AllDifferent over that curriculum's lectures already implies it -- a
provably redundant row, but presolve was already stripping it for ~free
either way, so this only saves build time, not solve time. H2 is built
from an inverted (period, room) index collected while x is created,
instead of rescanning every lecture per (period, room) pair. The
period-marginal helper px() is memoized instead of rebuilt on every H1/H3a/
H3b/S2/S3 call.
"""
import argparse
import json
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from dataset.ctt_parser import CTTProblem, parse_ctt

ALPHA = dict(S1=1, S2=5, S3=2, S4=1)

Lecture = Tuple[str, int]  # (course_id, lecture_index)


def build_model(problem: CTTProblem, time_limit: float = 60.0, workers: int = 4,
                 mip_gap: float = 0.0, log_progress: bool = False):
    n_periods = problem.n_periods
    n_days, periods_per_day = problem.n_days, problem.periods_per_day

    room_ids = list(problem.rooms.keys())
    n_rooms = len(room_ids)
    room_capacity = {rid: problem.rooms[rid].capacity for rid in room_ids}

    lectures: List[Lecture] = [
        (cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)
    ]

    # --- helper variables/functions used to build H4 (Availabilities) below ---
    # unavailable[cid]: the set of flattened period indices (day*periods_per_day
    # + period_in_day) that course cid's lectures are forbidden from. Built
    # once here from problem.unavailability so every later lookup is O(1).
    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * periods_per_day + u.period)
    _NONE_BLOCKED: set = set()  # shared empty-set sentinel for courses with no unavailability entries

    # is_allowed(cid, p): True iff course cid's lectures may be scheduled at
    # period p. This function *is* how H4 gets enforced -- see the x[l,p,r]
    # creation loop just below, which simply skips creating any variable for
    # a disallowed (course, period) combination instead of adding a
    # constraint that forbids it.
    def is_allowed(cid: str, p: int) -> bool:
        return p not in unavailable.get(cid, _NONE_BLOCKED)

    # allowed_periods(cid): the full list of periods course cid is permitted
    # to use, in period order. Used wherever a period range needs to be
    # pre-filtered up front (x-variable creation, H1).
    def allowed_periods(cid: str) -> List[int]:
        return [p for p in range(n_periods) if is_allowed(cid, p)]

    env = gp.Env(empty=True)
    if not log_progress:
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("ctt", env=env)
    m.Params.TimeLimit = time_limit
    m.Params.Threads = workers
    # m.Params.MIPGap = mip_gap

    # --- x: the core decision variable, and H4 (Availabilities) ---
    # x[l, p, r] (dict, keyed by (lecture, period, room)): binary Gurobi Var,
    # 1 iff lecture l is held in period p, room r. This is the ONLY decision
    # variable in the model -- every hard/soft constraint below is built out
    # of sums of these. H4 (Availabilities, official spec name) is enforced
    # here, not as a constraint: the inner loop only ever calls addVar() for
    # periods allowed_periods(cid) returns, so a variable for a forbidden
    # (course, period) combination is simply never created -- there is
    # nothing for the optimizer to ever set to 1, which is the same trick
    # ctt_solver.py uses via IntVarFromDomain on a restricted CP-SAT domain.
    #
    # by_period_room (dict, keyed by (period, room)): a list of every x
    # variable at that (period, room), collected in this same pass. It's an
    # inverted index -- a helper structure, not itself part of any
    # constraint's *meaning* -- that lets H2 below be built in one pass
    # instead of rescanning every lecture for every (period, room) pair.
    x: Dict[Tuple[Lecture, int, str], gp.Var] = {}
    by_period_room: Dict[Tuple[int, str], List[gp.Var]] = {}
    for lec in lectures:
        cid, _ = lec
        for p in allowed_periods(cid):
            for rid in room_ids:
                v = m.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{lec[1]}_{p}_{rid}")
                x[(lec, p, rid)] = v
                by_period_room.setdefault((p, rid), []).append(v)

    # px(lec, p) (helper function + _px_cache dict): the "period marginal" --
    # sum over every room of x[lec, p, r], i.e. "is lecture lec on at period
    # p, in any room". H1, H3a, H3b, S2, and S3 all need exactly this
    # quantity, repeatedly, for the same (lecture, period) pairs -- so
    # _px_cache memoizes each one the first time it's built instead of
    # reconstructing the same Gurobi linear expression from scratch on every
    # call. Every call site below only ever calls px() where
    # is_allowed(lec[0], p) is already known true, so no membership check is
    # needed inside px() itself.
    _px_cache: Dict[Tuple[Lecture, int], gp.LinExpr] = {}

    def px(lec, p):
        key = (lec, p)
        cached = _px_cache.get(key)
        if cached is None:
            cached = gp.quicksum(x[(lec, p, rid)] for rid in room_ids)
            _px_cache[key] = cached
        return cached

    # Assignment constraint: every lecture must get exactly one (period,
    # room). This is the "must be scheduled" half of the official spec's
    # "Lectures" hard constraint (the "distinct periods" half is H1, below).
    # It's not skippable/optional the way H1 sometimes is -- every lecture
    # needs exactly one row here, always.
    for lec in lectures:
        m.addConstr(gp.quicksum(v for (l2, _, _), v in x.items() if l2 == lec) == 1,
                    name=f"assign_{lec[0]}_{lec[1]}")

    # --- H1 (official spec name: "Lectures", the "distinct periods" half) ---
    # Requires: a course's own lectures must all land on different periods.
    # Encoded as one "<=1 lecture from this course per period" row, for each
    # period the course is allowed to use.
    #
    # courses_in_curricula (set, helper variable): every course id that
    # belongs to at least one curriculum. H1 is skipped entirely for any
    # such course, because H3a's constraint on that curriculum (below) is a
    # strict superset -- it already forces every lecture in the curriculum,
    # including this course's own, to be pairwise distinct. Generating H1
    # there would be a provably redundant row (Gurobi's own presolve was
    # already detecting and removing these; skipping them ourselves just
    # saves that presolve work).
    courses_in_curricula = {cid for q in problem.curricula.values() for cid in q.course_ids}
    for cid, c in problem.courses.items():
        if cid in courses_in_curricula:
            continue
        lecs = [(cid, li) for li in range(c.n_lectures)]
        if len(lecs) > 1:
            for p in allowed_periods(cid):
                m.addConstr(gp.quicksum(px(l, p) for l in lecs) <= 1, name=f"H1_{cid}_{p}")

    # --- H2 (official spec name: "RoomOccupancy") ---
    # Requires: no two lectures, from any courses, may share the same
    # (period, room). Encoded directly from by_period_room (see above): one
    # "<=1 lecture in this (period, room) slot" row per slot that has more
    # than one candidate lecture.
    for (p, rid), terms in by_period_room.items():
        if len(terms) > 1:
            m.addConstr(gp.quicksum(terms) <= 1, name=f"H2_{p}_{rid}")

    # --- H3a (the curriculum half of the official spec's "Conflicts") ---
    # Requires: every lecture of every course belonging to the same
    # curriculum must land on a different period from every other lecture
    # in that curriculum -- not just "no two courses clash", but every
    # single lecture instance across the whole curriculum needs its own
    # period. lecs below is exactly that pooled set (every course's every
    # lecture, unioned across the curriculum's members). One "<=1 lecture
    # from this pooled set per period" row per period, skipping periods none
    # of the curriculum's lectures could ever use anyway.
    for qid, q in problem.curricula.items():
        lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        if len(lecs) > 1:
            for p in range(n_periods):
                terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
                if terms:
                    m.addConstr(gp.quicksum(terms) <= 1, name=f"H3a_{qid}_{p}")

    # --- H3b (the teacher half of the official spec's "Conflicts") ---
    # Requires: every lecture of every course taught by the same teacher
    # must land on a different period -- same shape as H3a, just grouped by
    # teacher instead of curriculum, since a teacher obviously can't be in
    # two lectures at once regardless of which of their courses they belong
    # to.
    #
    # teacher_courses (dict, helper variable): maps each teacher id to the
    # list of course ids they teach, built once here by scanning every
    # course's teacher_id field. Needed because, unlike curricula, the
    # problem data doesn't already group courses by teacher anywhere.
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    for tid, cids in teacher_courses.items():
        lecs = [(cid, li) for cid in cids for li in range(problem.courses[cid].n_lectures)]
        if len(lecs) > 1:
            for p in range(n_periods):
                terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
                if terms:
                    m.addConstr(gp.quicksum(terms) <= 1, name=f"H3b_{tid}_{p}")

    objective = gp.LinExpr()

    # S1: RoomCapacity -- 1 point per student over the chosen room's capacity.
    # Purely a per-(l,r) cost coefficient, no auxiliary variables needed.
    for (lec, p, rid), v in x.items():
        cid, _ = lec
        excess = max(0, problem.courses[cid].n_students - room_capacity[rid])
        if excess:
            objective += ALPHA["S1"] * excess * v

    # S2: MinimumWorkingDays -- 5 points per day short of the course's minimum.
    # day_used only needs an upper bound: minimizing shortfall pushes
    # days_count (and so day_used) up, and the upper bound caps it exactly
    # at the true "was any lecture held that day" indicator.
    day_used: Dict[Tuple[str, int], gp.Var] = {}
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        for d in range(n_days):
            day_used[(cid, d)] = m.addVar(vtype=GRB.BINARY, name=f"dayused_{cid}_{d}")
            day_terms = [px(l, p) for l in lecs
                         for p in range(d * periods_per_day, (d + 1) * periods_per_day)
                         if is_allowed(l[0], p)]
            if day_terms:
                m.addConstr(day_used[(cid, d)] <= gp.quicksum(day_terms), name=f"dayused_ub_{cid}_{d}")
            else:
                m.addConstr(day_used[(cid, d)] == 0, name=f"dayused_zero_{cid}_{d}")
        days_count = gp.quicksum(day_used[(cid, d)] for d in range(n_days))
        shortfall = m.addVar(lb=0, ub=c.min_working_days, vtype=GRB.INTEGER, name=f"shortfall_{cid}")
        m.addConstr(shortfall >= c.min_working_days - days_count, name=f"shortfall_{cid}")
        objective += ALPHA["S2"] * shortfall

    # S3: CurriculumCompactness -- 2 points per lecture with no same-curriculum
    # neighbor in the adjacent slot(s) of the same day. occ must equal true
    # occupancy exactly (used both as "self" and as a neighbor's negation, so
    # it could be gamed in either direction with only a one-sided bound).
    for qid, q in problem.curricula.items():
        lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        occ: Dict[int, object] = {}  # gp.Var, or the literal 0 when no lecture can ever land here
        for d in range(n_days):
            for s in range(periods_per_day):
                p = d * periods_per_day + s
                terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
                if terms:
                    occ_p = m.addVar(vtype=GRB.BINARY, name=f"occ_{qid}_{p}")
                    m.addConstr(occ_p == gp.quicksum(terms), name=f"occ_eq_{qid}_{p}")
                    occ[p] = occ_p
                else:
                    occ[p] = 0  # no course in this curriculum is ever allowed at p
        for d in range(n_days):
            for s in range(periods_per_day):
                p = d * periods_per_day + s
                # occ[p] is either the int sentinel 0 (curriculum can never occupy p,
                # so provably not "isolated" there either) or a gp.Var -- checked via
                # isinstance, not `== 0`, since gp.Var.__eq__ builds a TempConstr for
                # addConstr() rather than doing a value comparison.
                if isinstance(occ[p], int):
                    continue
                terms = [occ[p]]
                if s > 0:
                    terms.append(1 - occ[p - 1])
                if s < periods_per_day - 1:
                    terms.append(1 - occ[p + 1])
                isolated = m.addVar(vtype=GRB.BINARY, name=f"isolated_{qid}_{p}")
                for t in terms:
                    m.addConstr(isolated <= t, name=f"isolated_ub_{qid}_{p}")
                m.addConstr(isolated >= gp.quicksum(terms) - (len(terms) - 1), name=f"isolated_lb_{qid}_{p}")
                objective += ALPHA["S3"] * isolated

    # S4: RoomStability -- 1 point per distinct room beyond the first, per
    # course. room_used needs a *lower* bound: minimizing "extra rooms"
    # would otherwise incentivize under-reporting usage, so we force it up
    # whenever any lecture of the course actually uses that room.
    #
    # Tried replacing this disaggregated per-(lecture,period) linking with a
    # single addGenConstrOr(ru, terms) per (course, room) -- fewer rows
    # pre-presolve, but empirically SLOWER: Gurobi's presolve was already
    # stripping the extra disaggregated rows for ~free (0.26s on a comp01
    # run), while addGenConstrOr's internal linearization produced a denser
    # presolved matrix (+17% nonzeros) and needed ~2.5x more B&B nodes to
    # prove optimality despite an identical root bound. Reverted -- the
    # disaggregated linear form is the tighter/faster formulation in
    # practice, not just in theory.
    allowed_period_list: Dict[str, List[int]] = {cid: allowed_periods(cid) for cid in problem.courses}
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        room_used = []
        for rid in room_ids:
            ru = m.addVar(vtype=GRB.BINARY, name=f"roomused_{cid}_{rid}")
            terms = [x[(lec, p, rid)] for lec in lecs for p in allowed_period_list[cid]]
            for t in terms:
                m.addConstr(ru >= t, name=f"roomused_lb_{cid}_{rid}")
            if not terms:
                m.addConstr(ru == 0, name=f"roomused_zero_{cid}_{rid}")
            room_used.append(ru)
        distinct_rooms = gp.quicksum(room_used)
        extra = m.addVar(lb=0, ub=max(0, n_rooms - 1), vtype=GRB.INTEGER, name=f"extra_{cid}")
        m.addConstr(extra >= distinct_rooms - 1, name=f"extra_{cid}")
        objective += ALPHA["S4"] * extra

    m.setObjective(objective, GRB.MINIMIZE)
    return m, lectures, x, room_ids


_STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "FEASIBLE",
    GRB.SUBOPTIMAL: "FEASIBLE",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INFEASIBLE",
}


def _extract_assignment(problem: CTTProblem, lectures: List[Lecture],
                         x: Dict[Tuple[Lecture, int, str], gp.Var], value_attr: str = "X") -> dict:
    """Reads one solution out of x -- value_attr='X' for the incumbent,
    'Xn' for whichever pool solution SolutionNumber currently points at."""
    assignment = {}
    for lec in lectures:
        cid, li = lec
        for (l2, p, rid), v in x.items():
            if l2 == lec and getattr(v, value_attr) > 0.5:
                assignment[f"{cid}#{li}"] = {
                    "day": p // problem.periods_per_day,
                    "period_in_day": p % problem.periods_per_day,
                    "period": p,
                    "room": rid,
                }
                break
    return assignment


def solve(problem: CTTProblem, time_limit: float = 60.0, workers: int = 8,
          mip_gap: float = 0.0, log_progress: bool = False) -> dict:
    t0 = time.time()
    m, lectures, x, room_ids = build_model(
        problem, time_limit=time_limit, workers=workers, mip_gap=mip_gap, log_progress=log_progress,
    )
    build_time = time.time() - t0
    t1 = time.time()
    m.optimize()
    optimize_time = time.time() - t1

    status_name = _STATUS_MAP.get(m.Status, f"UNKNOWN({m.Status})")
    feasible = m.SolCount > 0

    result = {
        "status": status_name,
        "objective": m.ObjVal if feasible else None,
        "best_bound": m.ObjBound if feasible else None,
        "build_time": build_time,  # Python-side model construction (addVar/addConstr calls)
        "optimize_time": optimize_time,  # Gurobi's own m.optimize() call
        "solve_time": build_time + optimize_time,  # kept for backward compatibility -- the old combined figure
        "assignment": _extract_assignment(problem, lectures, x, "X") if feasible else {},
    }
    return result


def solve_pool(problem: CTTProblem, time_limit: float = 60.0, workers: int = 8,
               mip_gap: float = 0.0, pool_solutions: int = 10, pool_gap_abs: float = 0.0,
               log_progress: bool = False) -> dict:
    """Like solve(), but returns every solution Gurobi's Solution Pool finds
    tied (within pool_gap_abs) at the best objective, instead of just the
    incumbent -- i.e. the set of (equally-)optimal solutions, capped at
    pool_solutions.

    Uses PoolSearchMode=2 ("find the pool_solutions best solutions"), so a
    single optimize() call both finds the optimum and searches for ties.
    `exhaustive=True` in the result means the search finished (proved
    optimality) without hitting the pool_solutions cap -- i.e. `solutions`
    is the *complete* set of optima, so len(solutions) == 1 there proves the
    optimum is unique. exhaustive=False (hit the cap, or hit time_limit
    before proving optimality) only means "at least this many" were found.
    """
    t0 = time.time()
    m, lectures, x, room_ids = build_model(
        problem, time_limit=time_limit, workers=workers, mip_gap=mip_gap, log_progress=log_progress,
    )
    build_time = time.time() - t0
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = pool_solutions
    m.Params.PoolGapAbs = pool_gap_abs
    t1 = time.time()
    m.optimize()
    optimize_time = time.time() - t1

    status_name = _STATUS_MAP.get(m.Status, f"UNKNOWN({m.Status})")
    feasible = m.SolCount > 0

    solutions = []
    if feasible:
        best = m.ObjVal
        for i in range(m.SolCount):
            m.Params.SolutionNumber = i
            if abs(m.PoolObjVal - best) <= pool_gap_abs + 1e-6:
                solutions.append(_extract_assignment(problem, lectures, x, "Xn"))

    exhaustive = feasible and m.Status == GRB.OPTIMAL and len(solutions) < pool_solutions

    return {
        "status": status_name,
        "objective": m.ObjVal if feasible else None,
        "best_bound": m.ObjBound if feasible else None,
        "build_time": build_time,
        "optimize_time": optimize_time,
        "solve_time": build_time + optimize_time,  # kept for backward compatibility
        "n_solutions_found": len(solutions),
        "exhaustive": exhaustive,  # True => `solutions` is provably complete (len==1 proves uniqueness)
        "solutions": solutions,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("ctt_path", nargs="?", default="data/ITC2007/real/comp01.ctt")
    cli.add_argument("--time-limit", type=float, default=1200.0)
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--mip-gap", type=float, default=0.0)
    cli.add_argument("--log-progress", action="store_true")
    cli.add_argument("--out", default=None)
    cli.add_argument("--pool", action="store_true",
                      help="search for all optimal solutions (Gurobi Solution Pool) instead of just one")
    cli.add_argument("--pool-solutions", type=int, default=10, help="cap on --pool solutions returned")
    cli.add_argument("--pool-gap-abs", type=float, default=0.0,
                      help="keep --pool solutions within this absolute gap of the best objective")
    args = cli.parse_args()

    problem = parse_ctt(args.ctt_path)

    if args.pool:
        result = solve_pool(
            problem, time_limit=args.time_limit, workers=args.workers, mip_gap=args.mip_gap,
            pool_solutions=args.pool_solutions, pool_gap_abs=args.pool_gap_abs,
            log_progress=args.log_progress,
        )
        print(f"status={result['status']} objective={result['objective']} bound={result['best_bound']} "
              f"build_time={result['build_time']:.1f}s optimize_time={result['optimize_time']:.1f}s "
              f"n_solutions_found={result['n_solutions_found']} exhaustive={result['exhaustive']}")
        if result["exhaustive"] and result["n_solutions_found"] == 1:
            print("-> optimum is unique (exhaustive search found exactly one tied solution)")
        elif result["exhaustive"]:
            print(f"-> exhaustive search: exactly {result['n_solutions_found']} solutions tied at the optimum")
        else:
            print(f"-> non-exhaustive: at least {result['n_solutions_found']} tied solutions exist "
                  f"(hit --pool-solutions cap or time limit before proving completeness)")
    else:
        result = solve(
            problem, time_limit=args.time_limit, workers=args.workers,
            mip_gap=args.mip_gap, log_progress=args.log_progress,
        )
        print(f"status={result['status']} objective={result['objective']} bound={result['best_bound']} "
              f"build_time={result['build_time']:.1f}s optimize_time={result['optimize_time']:.1f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote solution to {args.out}")
