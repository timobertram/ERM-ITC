"""MILP solver for ITC2007 Track 3 (Curriculum-Based Course Timetabling),
via gurobipy. Same semantics as dataset/ctt_solver.py's CP-SAT model, and a
drop-in alternative to it -- same result-dict shape from solve().

=====================================================================
CHANGED (this pass): core decision variable reformulated from
x[lecture, period, room] to y[course, period, room].
=====================================================================
Previously each of a course's n_lectures lectures had its own labeled copy
of variables (lecture 0, lecture 1, ...), which are, per the official spec,
"freely interchangeable" -- nothing in H1-H4/S1-S4 ever cared WHICH
lecture-index occupied which period, only which periods/rooms/days a
course used in aggregate. That meant every real schedule had n_lectures!
redundant relabelings sitting in the model as distinct, equally-costed
points -- confirmed empirically via solve_pool() (comp01's pool collapsed
10 raw entries to 3 real timetables; a synthetic instance collapsed 1000 to
581). The previous fix (see the OLD build_model(), commented out below)
added explicit symmetry-breaking constraints to prune the search over
those redundant copies. This version goes further: it removes the
redundant dimension from the model's variable set entirely, rather than
pruning a search over it. y[c, p, r] = 1 iff course c has *a* lecture at
(p, r), with one count constraint sum_{p,r} y[c,p,r] == n_lectures(c)
replacing the old per-lecture assignment constraint. Every H1-H4/S1-S4
constraint translates over directly (see each block's comment) because, as
above, none of them ever depended on lecture identity to begin with. Net
effect: variable count drops from lectures*periods*rooms to
courses*periods*rooms (a real reduction, not just fewer explored
duplicates), the explicit symmetry-breaking constraints become unnecessary
and are removed, and solve_pool() duplication from this cause is
eliminated at the source (there is no lecture-index left to relabel).

The OLD x[lecture,period,room]-based build_model() and _extract_assignment()
are preserved below, commented out in full, for easy revert -- look for the
"OLD (commented out)" banners. solve()/solve_pool()/the CLI are otherwise
unaffected: same call signatures, same result-dict shape.

Core decision variable: y[c, p, r] = 1 iff course c has a lecture in period
p, room r. This joint (period, room) variable is what makes H2 (room
occupancy) a plain linear <=1-per-(p,r) constraint instead of a bilinear
one; H1/H3a/H3b (distinct periods per course/curriculum/teacher) are then
"<=1 per period" sums over the appropriate course group, and H4
(availability) is enforced by simply never creating y[c, p, r] for p in
that course's unavailable periods -- mirroring how ctt_solver.py restricts
each period_var's IntVar domain directly.

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
AllDifferent over that curriculum's courses already implies it -- a
provably redundant row, but presolve was already stripping it for ~free
either way, so this only saves build time, not solve time. H2 is built
from an inverted (period, room) index collected while y is created,
instead of rescanning every course per (period, room) pair. The
period-marginal helper py() is memoized instead of rebuilt on every H1/H3a/
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

Lecture = Tuple[str, int]  # (course_id, lecture_index) -- still used for _extract_assignment's output keys


# =====================================================================
# OLD (commented out): x[lecture, period, room]-based build_model(), with
# explicit symmetry-breaking constraints to prune the resulting redundant
# lecture-index relabelings. Kept for easy revert -- see the new
# y[course, period, room]-based build_model() below, which removes that
# redundant dimension from the model instead of pruning a search over it.
# =====================================================================
# def build_model(problem: CTTProblem, time_limit: float = 60.0, workers: int = 4,
#                  mip_gap: float = 0.0, log_progress: bool = False,
#                  mip_focus: int = 0, heuristics: float = None,
#                  no_rel_heur_time: float = 0.0, pump_passes: int = None):
#     n_periods = problem.n_periods
#     n_days, periods_per_day = problem.n_days, problem.periods_per_day
#
#     room_ids = list(problem.rooms.keys())
#     n_rooms = len(room_ids)
#     room_capacity = {rid: problem.rooms[rid].capacity for rid in room_ids}
#
#     lectures: List[Lecture] = [
#         (cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)
#     ]
#
#     unavailable: Dict[str, set] = {}
#     for u in problem.unavailability:
#         unavailable.setdefault(u.course_id, set()).add(u.day * periods_per_day + u.period)
#     _NONE_BLOCKED: set = set()
#
#     def is_allowed(cid: str, p: int) -> bool:
#         return p not in unavailable.get(cid, _NONE_BLOCKED)
#
#     def allowed_periods(cid: str) -> List[int]:
#         return [p for p in range(n_periods) if is_allowed(cid, p)]
#
#     env = gp.Env(empty=True)
#     if not log_progress:
#         env.setParam("OutputFlag", 0)
#     env.start()
#     m = gp.Model("ctt", env=env)
#     m.Params.TimeLimit = time_limit
#     m.Params.Threads = workers
#     # m.Params.MIPGap = mip_gap
#
#     if mip_focus:
#         m.Params.MIPFocus = mip_focus
#     if heuristics is not None:
#         m.Params.Heuristics = heuristics
#     if no_rel_heur_time:
#         m.Params.NoRelHeurTime = no_rel_heur_time
#     if pump_passes is not None:
#         m.Params.PumpPasses = pump_passes
#
#     x: Dict[Tuple[Lecture, int, str], gp.Var] = {}
#     by_period_room: Dict[Tuple[int, str], List[gp.Var]] = {}
#     for lec in lectures:
#         cid, _ = lec
#         for p in allowed_periods(cid):
#             for rid in room_ids:
#                 v = m.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{lec[1]}_{p}_{rid}")
#                 x[(lec, p, rid)] = v
#                 by_period_room.setdefault((p, rid), []).append(v)
#
#     _px_cache: Dict[Tuple[Lecture, int], gp.LinExpr] = {}
#
#     def px(lec, p):
#         key = (lec, p)
#         cached = _px_cache.get(key)
#         if cached is None:
#             cached = gp.quicksum(x[(lec, p, rid)] for rid in room_ids)
#             _px_cache[key] = cached
#         return cached
#
#     for lec in lectures:
#         m.addConstr(gp.quicksum(v for (l2, _, _), v in x.items() if l2 == lec) == 1,
#                     name=f"assign_{lec[0]}_{lec[1]}")
#
#     courses_in_curricula = {cid for q in problem.curricula.values() for cid in q.course_ids}
#     for cid, c in problem.courses.items():
#         if cid in courses_in_curricula:
#             continue
#         lecs = [(cid, li) for li in range(c.n_lectures)]
#         if len(lecs) > 1:
#             for p in allowed_periods(cid):
#                 m.addConstr(gp.quicksum(px(l, p) for l in lecs) <= 1, name=f"H1_{cid}_{p}")
#
#     for (p, rid), terms in by_period_room.items():
#         if len(terms) > 1:
#             m.addConstr(gp.quicksum(terms) <= 1, name=f"H2_{p}_{rid}")
#
#     for qid, q in problem.curricula.items():
#         lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
#         if len(lecs) > 1:
#             for p in range(n_periods):
#                 terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
#                 if terms:
#                     m.addConstr(gp.quicksum(terms) <= 1, name=f"H3a_{qid}_{p}")
#
#     teacher_courses: Dict[str, List[str]] = {}
#     for cid, c in problem.courses.items():
#         teacher_courses.setdefault(c.teacher_id, []).append(cid)
#     for tid, cids in teacher_courses.items():
#         lecs = [(cid, li) for cid in cids for li in range(problem.courses[cid].n_lectures)]
#         if len(lecs) > 1:
#             for p in range(n_periods):
#                 terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
#                 if terms:
#                     m.addConstr(gp.quicksum(terms) <= 1, name=f"H3b_{tid}_{p}")
#
#     # Symmetry breaking: canonical lecture-index ordering. No longer needed
#     # in the y[course,period,room] formulation below -- there is no
#     # lecture-index dimension left to have symmetry in.
#     for cid, c in problem.courses.items():
#         lecs = [(cid, li) for li in range(c.n_lectures)]
#         if len(lecs) > 1:
#             period_of = {l: gp.quicksum(p * px(l, p) for p in allowed_periods(cid)) for l in lecs}
#             for i in range(len(lecs) - 1):
#                 m.addConstr(period_of[lecs[i]] <= period_of[lecs[i + 1]] - 1, name=f"symbreak_{cid}_{i}")
#
#     objective = gp.LinExpr()
#
#     for (lec, p, rid), v in x.items():
#         cid, _ = lec
#         excess = max(0, problem.courses[cid].n_students - room_capacity[rid])
#         if excess:
#             objective += ALPHA["S1"] * excess * v
#
#     day_used: Dict[Tuple[str, int], gp.Var] = {}
#     for cid, c in problem.courses.items():
#         lecs = [(cid, li) for li in range(c.n_lectures)]
#         for d in range(n_days):
#             day_used[(cid, d)] = m.addVar(vtype=GRB.BINARY, name=f"dayused_{cid}_{d}")
#             day_terms = [px(l, p) for l in lecs
#                          for p in range(d * periods_per_day, (d + 1) * periods_per_day)
#                          if is_allowed(l[0], p)]
#             if day_terms:
#                 m.addConstr(day_used[(cid, d)] <= gp.quicksum(day_terms), name=f"dayused_ub_{cid}_{d}")
#             else:
#                 m.addConstr(day_used[(cid, d)] == 0, name=f"dayused_zero_{cid}_{d}")
#         days_count = gp.quicksum(day_used[(cid, d)] for d in range(n_days))
#         shortfall = m.addVar(lb=0, ub=c.min_working_days, vtype=GRB.INTEGER, name=f"shortfall_{cid}")
#         m.addConstr(shortfall >= c.min_working_days - days_count, name=f"shortfall_{cid}")
#         objective += ALPHA["S2"] * shortfall
#
#     for qid, q in problem.curricula.items():
#         lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
#         occ: Dict[int, object] = {}
#         for d in range(n_days):
#             for s in range(periods_per_day):
#                 p = d * periods_per_day + s
#                 terms = [px(l, p) for l in lecs if is_allowed(l[0], p)]
#                 if terms:
#                     occ_p = m.addVar(vtype=GRB.BINARY, name=f"occ_{qid}_{p}")
#                     m.addConstr(occ_p == gp.quicksum(terms), name=f"occ_eq_{qid}_{p}")
#                     occ[p] = occ_p
#                 else:
#                     occ[p] = 0
#         for d in range(n_days):
#             for s in range(periods_per_day):
#                 p = d * periods_per_day + s
#                 if isinstance(occ[p], int):
#                     continue
#                 terms = [occ[p]]
#                 if s > 0:
#                     terms.append(1 - occ[p - 1])
#                 if s < periods_per_day - 1:
#                     terms.append(1 - occ[p + 1])
#                 isolated = m.addVar(vtype=GRB.BINARY, name=f"isolated_{qid}_{p}")
#                 for t in terms:
#                     m.addConstr(isolated <= t, name=f"isolated_ub_{qid}_{p}")
#                 m.addConstr(isolated >= gp.quicksum(terms) - (len(terms) - 1), name=f"isolated_lb_{qid}_{p}")
#                 objective += ALPHA["S3"] * isolated
#
#     allowed_period_list: Dict[str, List[int]] = {cid: allowed_periods(cid) for cid in problem.courses}
#     for cid, c in problem.courses.items():
#         lecs = [(cid, li) for li in range(c.n_lectures)]
#         room_used = []
#         for rid in room_ids:
#             ru = m.addVar(vtype=GRB.BINARY, name=f"roomused_{cid}_{rid}")
#             terms = [x[(lec, p, rid)] for lec in lecs for p in allowed_period_list[cid]]
#             for t in terms:
#                 m.addConstr(ru >= t, name=f"roomused_lb_{cid}_{rid}")
#             if not terms:
#                 m.addConstr(ru == 0, name=f"roomused_zero_{cid}_{rid}")
#             room_used.append(ru)
#         distinct_rooms = gp.quicksum(room_used)
#         extra = m.addVar(lb=0, ub=max(0, n_rooms - 1), vtype=GRB.INTEGER, name=f"extra_{cid}")
#         m.addConstr(extra >= distinct_rooms - 1, name=f"extra_{cid}")
#         objective += ALPHA["S4"] * extra
#
#     m.setObjective(objective, GRB.MINIMIZE)
#     return m, lectures, x, room_ids


# =====================================================================
# NEW: y[course, period, room]-based build_model(). Same public name/
# signature/return arity as before (m, <something>, <something>, room_ids)
# except the second element is now `y` (course-indexed) instead of
# `lectures` (the lecture list is no longer needed inside build_model()
# itself -- see _extract_assignment() below for where per-lecture labels
# get reconstructed, arbitrarily, purely for output-format compatibility).
# =====================================================================
def build_model(problem: CTTProblem, time_limit: float = 60.0, workers: int = 4,
                 mip_gap: float = 0.0, log_progress: bool = False,
                 mip_focus: int = 0, heuristics: float = None,
                 no_rel_heur_time: float = 0.0, pump_passes: int = None,
                 exact_day_used: bool = False):
    n_periods = problem.n_periods
    n_days, periods_per_day = problem.n_days, problem.periods_per_day

    room_ids = list(problem.rooms.keys())
    n_rooms = len(room_ids)
    room_capacity = {rid: problem.rooms[rid].capacity for rid in room_ids}

    # --- helper variables/functions used to build H4 (Availabilities) below ---
    # unavailable[cid]: the set of flattened period indices (day*periods_per_day
    # + period_in_day) that course cid is forbidden from. Built once here
    # from problem.unavailability so every later lookup is O(1).
    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * periods_per_day + u.period)
    _NONE_BLOCKED: set = set()  # shared empty-set sentinel for courses with no unavailability entries

    # is_allowed(cid, p): True iff course cid may be scheduled at period p.
    # This function *is* how H4 gets enforced -- see the y[c,p,r] creation
    # loop just below, which simply skips creating any variable for a
    # disallowed (course, period) combination instead of adding a
    # constraint that forbids it.
    def is_allowed(cid: str, p: int) -> bool:
        return p not in unavailable.get(cid, _NONE_BLOCKED)

    # allowed_periods(cid): the full list of periods course cid is permitted
    # to use, in period order.
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

    # Feasibility-finding knobs -- all off/default unless explicitly passed,
    # so existing behavior on instances that already find an incumbent fine
    # is unchanged. Aimed at large instances where finding ANY feasible
    # solution is the actual bottleneck, not proving optimality:
    #   mip_focus=1     bias the whole search toward finding feasible
    #                    solutions rather than improving/proving the bound.
    #   heuristics       fraction of runtime spent in Gurobi's internal
    #                    heuristics (default 0.05) -- raise it to spend more
    #                    effort hunting incumbents instead of growing the tree.
    #   no_rel_heur_time seconds spent searching for a feasible solution
    #                    BEFORE even solving the root LP relaxation.
    #   pump_passes      enables/tunes the Feasibility Pump, a heuristic
    #                    purpose-built for constructing a first feasible
    #                    integer solution to a hard combinatorial problem.
    if mip_focus:
        m.Params.MIPFocus = mip_focus
    if heuristics is not None:
        m.Params.Heuristics = heuristics
    if no_rel_heur_time:
        m.Params.NoRelHeurTime = no_rel_heur_time
    if pump_passes is not None:
        m.Params.PumpPasses = pump_passes

    # --- y: the core decision variable, and H4 (Availabilities) ---
    # CHANGED: y[c, p, r] (dict, keyed by (course_id, period, room)): binary
    # Gurobi Var, 1 iff course c has *a* lecture in period p, room r --
    # there is no longer a separate copy per lecture-index. This is the
    # ONLY decision variable in the model -- every hard/soft constraint
    # below is built out of sums of these. H4 (Availabilities, official
    # spec name) is enforced here, not as a constraint: the inner loop only
    # ever calls addVar() for periods allowed_periods(cid) returns.
    #
    # by_period_room (dict, keyed by (period, room)): a list of every y
    # variable at that (period, room), collected in this same pass -- an
    # inverted index that lets H2 below be built in one pass instead of
    # rescanning every course for every (period, room) pair.
    y: Dict[Tuple[str, int, str], gp.Var] = {}
    by_period_room: Dict[Tuple[int, str], List[gp.Var]] = {}
    for cid in problem.courses:
        for p in allowed_periods(cid):
            for rid in room_ids:
                v = m.addVar(vtype=GRB.BINARY, name=f"y_{cid}_{p}_{rid}")
                y[(cid, p, rid)] = v
                by_period_room.setdefault((p, rid), []).append(v)

    # CHANGED: py(cid, p) (renamed from px; course-indexed instead of
    # lecture-indexed) -- sum over every room of y[cid, p, r], i.e. "does
    # course cid have a lecture on at period p, in any room". H1, H3a, H3b,
    # S2, and S3 all need exactly this quantity, repeatedly, for the same
    # (course, period) pairs -- _py_cache memoizes each one the first time
    # it's built. Every call site below only ever calls py() where
    # is_allowed(cid, p) is already known true, so no membership check is
    # needed inside py() itself.
    _py_cache: Dict[Tuple[str, int], gp.LinExpr] = {}

    def py(cid, p):
        key = (cid, p)
        cached = _py_cache.get(key)
        if cached is None:
            cached = gp.quicksum(y[(cid, p, rid)] for rid in room_ids)
            _py_cache[key] = cached
        return cached

    # CHANGED: count constraint, replacing the old per-lecture assignment
    # constraint -- every course must use exactly n_lectures periods (each
    # contributing exactly one room, enforced by y's own binary domain plus
    # H1 below capping each period's contribution at 1). This is the "must
    # be scheduled" half of the official spec's "Lectures" hard constraint
    # (the "distinct periods" half is H1, below).
    for cid, c in problem.courses.items():
        m.addConstr(gp.quicksum(py(cid, p) for p in allowed_periods(cid)) == c.n_lectures,
                    name=f"count_{cid}")

    # --- H1 (official spec name: "Lectures", the "distinct periods" half) ---
    # Requires: a course's own lectures must all land on different periods.
    # CHANGED: this is no longer implied "for free" by the count constraint
    # alone (nothing else stops 2+ of a course's n_lectures budget being
    # spent on the SAME period across different rooms) -- still needed
    # explicitly, just directly on py(cid, p) now instead of pooling a
    # lecs list. Still skipped for any course in a curriculum, since H3a's
    # constraint on that curriculum (below) is a strict superset -- exactly
    # the same redundancy argument as before, re-verified for this
    # formulation: if py(cid, p) could exceed 1, the curriculum-level H3a
    # sum (which includes py(cid, p) as one of its terms) would already be
    # violated, so H3a alone still forces py(cid, p) <= 1 for any
    # curriculum-covered course.
    courses_in_curricula = {cid for q in problem.curricula.values() for cid in q.course_ids}
    for cid, c in problem.courses.items():
        if cid in courses_in_curricula:
            continue
        if c.n_lectures > 1:
            for p in allowed_periods(cid):
                m.addConstr(py(cid, p) <= 1, name=f"H1_{cid}_{p}")

    # --- H2 (official spec name: "RoomOccupancy") ---
    # Requires: no two lectures, from any courses, may share the same
    # (period, room). Encoded directly from by_period_room (see above): one
    # "<=1 lecture in this (period, room) slot" row per slot that has more
    # than one candidate course.
    for (p, rid), terms in by_period_room.items():
        if len(terms) > 1:
            m.addConstr(gp.quicksum(terms) <= 1, name=f"H2_{p}_{rid}")

    # --- H3a (the curriculum half of the official spec's "Conflicts") ---
    # Requires: every lecture of every course belonging to the same
    # curriculum must land on a different period from every other lecture
    # in that curriculum -- not just "no two courses clash", but every
    # single lecture across the whole curriculum needs its own period.
    # CHANGED: summed directly over the curriculum's courses' py() values
    # instead of a pooled per-lecture list -- mathematically identical
    # (py(cid, p) is already "how many of cid's lectures land at p", capped
    # at 1 by H1/this-constraint-itself, so summing it across courses is
    # exactly "how many lectures from any course in the curriculum land at
    # p"). total_lectures uses n_lectures directly (not len(course_ids))
    # so single-course, multi-lecture curricula (H3a subsuming H1 for that
    # one course) are still correctly covered.
    for qid, q in problem.curricula.items():
        total_lectures = sum(problem.courses[cid].n_lectures for cid in q.course_ids)
        if total_lectures > 1:
            for p in range(n_periods):
                terms = [py(cid, p) for cid in q.course_ids if is_allowed(cid, p)]
                if terms:
                    m.addConstr(gp.quicksum(terms) <= 1, name=f"H3a_{qid}_{p}")

    # --- H3b (the teacher half of the official spec's "Conflicts") ---
    # Requires: every lecture of every course taught by the same teacher
    # must land on a different period -- same shape as H3a, grouped by
    # teacher instead of curriculum. Same total_lectures fix as H3a, same
    # reasoning.
    #
    # teacher_courses (dict, helper variable): maps each teacher id to the
    # list of course ids they teach, built once here by scanning every
    # course's teacher_id field.
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    for tid, cids in teacher_courses.items():
        total_lectures = sum(problem.courses[cid].n_lectures for cid in cids)
        if total_lectures > 1:
            for p in range(n_periods):
                terms = [py(cid, p) for cid in cids if is_allowed(cid, p)]
                if terms:
                    m.addConstr(gp.quicksum(terms) <= 1, name=f"H3b_{tid}_{p}")

    # CHANGED: no symmetry-breaking block here -- see the module docstring.
    # There is no lecture-index dimension left in this formulation, so
    # there is nothing to break symmetry over; the old block (pruning a
    # search over n_lectures! redundant relabelings) is now unnecessary by
    # construction rather than handled by an added constraint.

    objective = gp.LinExpr()

    # S1: RoomCapacity -- 1 point per student over the chosen room's capacity.
    # Purely a per-(c,r) cost coefficient, no auxiliary variables needed.
    # CHANGED: iterates y.items() (course-keyed) instead of x.items()
    # (lecture-keyed) -- excess never depended on lecture identity anyway.
    for (cid, p, rid), v in y.items():
        excess = max(0, problem.courses[cid].n_students - room_capacity[rid])
        if excess:
            objective += ALPHA["S1"] * excess * v

    # S2: MinimumWorkingDays -- 5 points per day short of the course's minimum.
    # day_used only needs an upper bound: minimizing shortfall pushes
    # days_count (and so day_used) up, and the upper bound caps it exactly
    # at the true "was any lecture held that day" indicator.
    # That pins the objective, not day_used itself: once a course already
    # meets min_working_days, a used day's day_used is free to be 0 or 1 at
    # equal cost, so a solution pool lists every timetable once per such
    # combination (confirmed: the only differing variables between repeated
    # pool entries). exact_day_used=True -- set by solve_pool() -- adds the
    # matching lower bounds (day_used >= every period's py of that day);
    # solve() leaves them out, since the optimum doesn't need them.
    # CHANGED: day_terms reads py(cid, p) directly (one term per period)
    # instead of pooling px(l, p) over a lecs list (many terms per period,
    # only ever 0 or 1 truthy under H1 anyway) -- same result, simpler.
    day_used: Dict[Tuple[str, int], gp.Var] = {}
    for cid, c in problem.courses.items():
        for d in range(n_days):
            day_used[(cid, d)] = m.addVar(vtype=GRB.BINARY, name=f"dayused_{cid}_{d}")
            day_terms = [py(cid, p)
                         for p in range(d * periods_per_day, (d + 1) * periods_per_day)
                         if is_allowed(cid, p)]
            if day_terms:
                m.addConstr(day_used[(cid, d)] <= gp.quicksum(day_terms), name=f"dayused_ub_{cid}_{d}")
                if exact_day_used:
                    for k, term in enumerate(day_terms):
                        m.addConstr(day_used[(cid, d)] >= term, name=f"dayused_lb_{cid}_{d}_{k}")
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
    # CHANGED: terms sums py(cid, p) over the curriculum's courses directly
    # instead of pooling px(l, p) over a per-lecture list.
    for qid, q in problem.curricula.items():
        occ: Dict[int, object] = {}  # gp.Var, or the literal 0 when no lecture can ever land here
        for d in range(n_days):
            for s in range(periods_per_day):
                p = d * periods_per_day + s
                terms = [py(cid, p) for cid in q.course_ids if is_allowed(cid, p)]
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
                # commented out because this is already inforced by the objective function (currently testing it)
                # for t in terms:
                #     m.addConstr(isolated <= t, name=f"isolated_ub_{qid}_{p}")
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
    # CHANGED: terms reads y[(cid,p,rid)] directly per allowed period --
    # the old per-lecture lecs list is no longer needed since y is already
    # course-scoped (one variable per (course,period,room), not per
    # (lecture,period,room)).
    allowed_period_list: Dict[str, List[int]] = {cid: allowed_periods(cid) for cid in problem.courses}
    for cid, c in problem.courses.items():
        room_used = []
        for rid in room_ids:
            ru = m.addVar(vtype=GRB.BINARY, name=f"roomused_{cid}_{rid}")
            terms = [y[(cid, p, rid)] for p in allowed_period_list[cid]]
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
    return m, y, room_ids


_STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "FEASIBLE",
    GRB.SUBOPTIMAL: "FEASIBLE",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INFEASIBLE",
}


# =====================================================================
# OLD (commented out): _extract_assignment() for the x[lecture,period,room]
# formulation -- reads a solution by looking up each already-labeled
# lecture's variable directly.
# =====================================================================
# def _extract_assignment(problem: CTTProblem, lectures: List[Lecture],
#                          x: Dict[Tuple[Lecture, int, str], gp.Var], value_attr: str = "X") -> dict:
#     """Reads one solution out of x -- value_attr='X' for the incumbent,
#     'Xn' for whichever pool solution SolutionNumber currently points at."""
#     assignment = {}
#     for lec in lectures:
#         cid, li = lec
#         for (l2, p, rid), v in x.items():
#             if l2 == lec and getattr(v, value_attr) > 0.5:
#                 assignment[f"{cid}#{li}"] = {
#                     "day": p // problem.periods_per_day,
#                     "period_in_day": p % problem.periods_per_day,
#                     "period": p,
#                     "room": rid,
#                 }
#                 break
#     return assignment


# CHANGED: new y-based _extract_assignment(). Since y no longer carries a
# lecture-index label, this collects each course's selected (period, room)
# slots (there are exactly c.n_lectures of them, guaranteed by the count
# constraint) and assigns them arbitrary lecture-index labels 0..n-1 (sorted
# by period for determinism) purely to keep the output dict's "cid#li" key
# format compatible with the rest of the pipeline (ctt_encode.py, the
# official ITC2007 output format itself doesn't have a lecture-index at all
# -- see the module docstring). Which physical lecture gets which index is
# arbitrary and irrelevant, exactly because lectures are interchangeable --
# ctt_encode.py's downstream training already handles this via a
# Hungarian-matched loss (see README), so this arbitrary relabeling doesn't
# introduce any new inconsistency. Also more efficient than the old version:
# one O(len(y)) pass building an index, instead of an O(lectures * len(x))
# nested scan.
def _extract_assignment(problem: CTTProblem, y: Dict[Tuple[str, int, str], gp.Var],
                         value_attr: str = "X") -> dict:
    """Reads one solution out of y -- value_attr='X' for the incumbent,
    'Xn' for whichever pool solution SolutionNumber currently points at."""
    by_course: Dict[str, List[Tuple[int, str]]] = {}
    for (cid, p, rid), v in y.items():
        if getattr(v, value_attr) > 0.5:
            by_course.setdefault(cid, []).append((p, rid))

    assignment = {}
    for cid, slots in by_course.items():
        for li, (p, rid) in enumerate(sorted(slots)):
            assignment[f"{cid}#{li}"] = {
                "day": p // problem.periods_per_day,
                "period_in_day": p % problem.periods_per_day,
                "period": p,
                "room": rid,
            }
    return assignment


def solve(problem: CTTProblem, time_limit: float = 60.0, workers: int = 8,
          mip_gap: float = 0.0, log_progress: bool = False,
          mip_focus: int = 0, heuristics: float = None,
          no_rel_heur_time: float = 0.0, pump_passes: int = None) -> dict:
    t0 = time.time()
    # CHANGED: build_model() now returns y instead of lectures as the
    # second element.
    m, y, room_ids = build_model(
        problem, time_limit=time_limit, workers=workers, mip_gap=mip_gap, log_progress=log_progress,
        mip_focus=mip_focus, heuristics=heuristics,
        no_rel_heur_time=no_rel_heur_time, pump_passes=pump_passes,
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
        "assignment": _extract_assignment(problem, y, "X") if feasible else {},
    }
    return result


def solve_pool(problem: CTTProblem, time_limit: float = 60.0, workers: int = 8,
               mip_gap: float = 0.0, pool_solutions: int = 10, pool_gap_abs: float = 0.0,
               log_progress: bool = False,
               mip_focus: int = 0, heuristics: float = None,
               no_rel_heur_time: float = 0.0, pump_passes: int = None) -> dict:
    """Like solve(), but returns every distinct timetable Gurobi's Solution
    Pool finds tied (within pool_gap_abs) at the best objective, instead of
    just the incumbent -- i.e. the set of (equally-)optimal solutions,
    capped at pool_solutions. `assignment` is still the incumbent, as in
    solve(), so the result works anywhere a solve() result does (e.g.
    dataset/ctt_encode.py).

    Uses PoolSearchMode=2 ("find the pool_solutions best solutions"), so a
    single optimize() call both finds the optimum and searches for ties.
    `exhaustive=True` in the result means the search finished (proved
    optimality) without the pool filling up -- i.e. `solutions` is the
    *complete* set of optima, so len(solutions) == 1 there proves the
    optimum is unique. exhaustive=False (pool full, or hit time_limit
    before proving optimality) only means "at least this many" were found.

    Pool entries are distinct in *all* model variables, so a timetable
    whose helper variables can take several values at equal cost would be
    listed several times and fill the pool early. exact_day_used=True (see
    build_model()'s S2 block) removes the source found in practice; any
    remaining repeats -- e.g. S3/S4 helpers set higher than needed, within
    pool_gap_abs > 0 -- are dropped here, keeping each timetable's first
    (best-objective) entry. `n_pool_entries` counts entries before that.
    """
    t0 = time.time()
    m, y, room_ids = build_model(
        problem, time_limit=time_limit, workers=workers, mip_gap=mip_gap, log_progress=log_progress,
        mip_focus=mip_focus, heuristics=heuristics,
        no_rel_heur_time=no_rel_heur_time, pump_passes=pump_passes, exact_day_used=True,
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
    seen = set()
    n_pool_entries = 0
    if feasible:
        best = m.ObjVal
        for i in range(m.SolCount):  # pool is sorted best-first
            m.Params.SolutionNumber = i
            if abs(m.PoolObjVal - best) > pool_gap_abs + 1e-6:
                continue
            n_pool_entries += 1
            assignment = _extract_assignment(problem, y, "Xn")
            timetable = tuple(sorted((k, a["period"], a["room"]) for k, a in assignment.items()))
            if timetable not in seen:
                seen.add(timetable)
                solutions.append(assignment)

    # judged on the raw pool: repeats removed above still took up pool slots
    exhaustive = feasible and m.Status == GRB.OPTIMAL and m.SolCount < pool_solutions

    return {
        "status": status_name,
        "objective": m.ObjVal if feasible else None,
        "best_bound": m.ObjBound if feasible else None,
        "build_time": build_time,
        "optimize_time": optimize_time,
        "solve_time": build_time + optimize_time,  # kept for backward compatibility
        "n_solutions_found": len(solutions),
        "n_pool_entries": n_pool_entries,  # before dropping repeated timetables
        "exhaustive": exhaustive,  # True => `solutions` is provably complete (len==1 proves uniqueness)
        "assignment": _extract_assignment(problem, y, "X") if feasible else {},  # the incumbent, as in solve()
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
    cli.add_argument("--pool-solutions", type=int, default=100, help="cap on --pool solutions returned")
    cli.add_argument("--pool-gap-abs", type=float, default=0.0,
                      help="keep --pool solutions within this absolute gap of the best objective")
    cli.add_argument("--mip-focus", type=int, default=0, choices=[0, 1, 2, 3],
                      help="Gurobi MIPFocus -- 1 biases the search toward finding feasible solutions "
                           "quickly rather than proving optimality (0 = Gurobi's balanced default)")
    cli.add_argument("--heuristics", type=float, default=None,
                      help="fraction of runtime spent in Gurobi's internal heuristics (default: Gurobi's own 0.05)")
    cli.add_argument("--no-rel-heur-time", type=float, default=0.0,
                      help="seconds spent searching for a feasible solution before solving the root LP relaxation")
    cli.add_argument("--pump-passes", type=int, default=None,
                      help="enables/tunes Gurobi's Feasibility Pump heuristic")
    args = cli.parse_args()

    problem = parse_ctt(args.ctt_path)
    _feas_kwargs = dict(
        mip_focus=args.mip_focus, heuristics=args.heuristics,
        no_rel_heur_time=args.no_rel_heur_time, pump_passes=args.pump_passes,
    )

    if args.pool:
        result = solve_pool(
            problem, time_limit=args.time_limit, workers=args.workers, mip_gap=args.mip_gap,
            pool_solutions=args.pool_solutions, pool_gap_abs=args.pool_gap_abs,
            log_progress=args.log_progress, **_feas_kwargs,
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
            mip_gap=args.mip_gap, log_progress=args.log_progress, **_feas_kwargs,
        )
        print(f"status={result['status']} objective={result['objective']} bound={result['best_bound']} "
              f"build_time={result['build_time']:.1f}s optimize_time={result['optimize_time']:.1f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote solution to {args.out}")
