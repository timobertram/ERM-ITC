"""CP-SAT solver for ITC2007 Track 3 (Curriculum-Based Course Timetabling).

  H1/H3a/H3b (distinct periods per course/curriculum/teacher): AddAllDifferent
  over period_var. H2 (room occupancy): AddAllDifferent over
  period_var*n_rooms + room_var. H4 (availability): excluded from each
  lecture's period_var domain directly.

Soft constraints S1-S4 follow the official spec's formulas/weights.
"""
import argparse
import json
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from dataset.ctt_parser import CTTProblem, parse_ctt

ALPHA = dict(S1=1, S2=5, S3=2, S4=1)

Lecture = Tuple[str, int]  # (course_id, lecture_index)


def build_model(problem: CTTProblem):
    model = cp_model.CpModel()
    n_periods = problem.n_periods
    n_days, periods_per_day = problem.n_days, problem.periods_per_day

    room_ids = list(problem.rooms.keys())
    n_rooms = len(room_ids)
    room_capacity = [problem.rooms[rid].capacity for rid in room_ids]

    lectures: List[Lecture] = [
        (cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)
    ]

    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * periods_per_day + u.period)

    period_var: Dict[Lecture, cp_model.IntVar] = {}
    room_var: Dict[Lecture, cp_model.IntVar] = {}
    for lec in lectures:
        cid, li = lec
        allowed = [p for p in range(n_periods) if p not in unavailable.get(cid, set())]
        period_var[lec] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(allowed), f"period_{cid}_{li}")
        room_var[lec] = model.NewIntVar(0, n_rooms - 1, f"room_{cid}_{li}")

    # H1(b): lectures of the same course must land on distinct periods
    for cid, c in problem.courses.items():
        # lecs = [(cid, li) for li in range(c.n_lectures)]
        # if len(lecs) > 1:
        #     #model.AddAllDifferent([period_var[l] for l in lecs])
        for l in range(c.n_lectures-1):
            model.Add(period_var[(cid,l)] < period_var[(cid,l+1)]) # symmetry breaking: order lessons for courses

    # H2: room occupancy -- no two lectures (any course) share (period, room)
    model.AddAllDifferent([period_var[l] * n_rooms + room_var[l] for l in lectures])

    # H3a: courses in the same curriculum must all land on distinct periods
    for q in problem.curricula.values():
        lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        if len(lecs) > 1:
            model.AddAllDifferent([period_var[l] for l in lecs])

    # H3b: courses taught by the same teacher must all land on distinct periods
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    for cids in teacher_courses.values():
        lecs = [(cid, li) for cid in cids for li in range(problem.courses[cid].n_lectures)]
        if len(lecs) > 1:
            model.AddAllDifferent([period_var[l] for l in lecs])

    objective_terms = []

    # S1: RoomCapacity -- 1 point per student over the chosen room's capacity
    for lec in lectures:
        cid, _ = lec
        n_students = problem.courses[cid].n_students
        capacity = model.NewIntVar(min(room_capacity), max(room_capacity), f"cap_{cid}_{lec[1]}")
        model.AddElement(room_var[lec], room_capacity, capacity)
        excess = model.NewIntVar(0, max(0, n_students - min(room_capacity)), f"excess_{cid}_{lec[1]}")
        model.Add(excess >= n_students - capacity)
        objective_terms.append(ALPHA["S1"] * excess)

    # day_var, needed for both S2 and S3
    day_var: Dict[Lecture, cp_model.IntVar] = {}
    for lec in lectures:
        d = model.NewIntVar(0, n_days - 1, f"day_{lec[0]}_{lec[1]}")
        model.AddDivisionEquality(d, period_var[lec], periods_per_day)
        day_var[lec] = d

    # S2: MinimumWorkingDays -- 5 points per day short of the course's minimum
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        day_used = []
        for d in range(n_days):
            on_day = []
            for lec in lecs:
                b = model.NewBoolVar(f"onday_{cid}_{lec[1]}_{d}")
                model.Add(day_var[lec] == d).OnlyEnforceIf(b)
                model.Add(day_var[lec] != d).OnlyEnforceIf(b.Not())
                on_day.append(b)
            used = model.NewBoolVar(f"dayused_{cid}_{d}")
            model.AddMaxEquality(used, on_day)
            day_used.append(used)
        days_count = model.NewIntVar(0, n_days, f"dayscount_{cid}")
        model.Add(days_count == sum(day_used))
        shortfall = model.NewIntVar(0, c.min_working_days, f"shortfall_{cid}")
        model.Add(shortfall >= c.min_working_days - days_count)
        objective_terms.append(ALPHA["S2"] * shortfall)

    # S3: CurriculumCompactness -- 2 points per lecture with no same-curriculum neighbor that day
    for qid, q in problem.curricula.items():
        lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        occ = {}
        for d in range(n_days):
            for s in range(periods_per_day):
                p = d * periods_per_day + s
                at_slot = []
                for lec in lecs:
                    b = model.NewBoolVar(f"occ_{qid}_{lec[0]}_{lec[1]}_{p}")
                    model.Add(period_var[lec] == p).OnlyEnforceIf(b)
                    model.Add(period_var[lec] != p).OnlyEnforceIf(b.Not())
                    at_slot.append(b)
                o = model.NewBoolVar(f"qocc_{qid}_{p}")
                if at_slot:
                    model.AddMaxEquality(o, at_slot)
                else:
                    model.Add(o == 0)
                occ[(d, s)] = o
        for d in range(n_days):
            for s in range(periods_per_day):
                neighbors = []
                if s > 0:
                    neighbors.append(occ[(d, s - 1)])
                if s < periods_per_day - 1:
                    neighbors.append(occ[(d, s + 1)])
                isolated = model.NewBoolVar(f"isolated_{qid}_{d}_{s}")
                model.AddMinEquality(isolated, [occ[(d, s)]] + [n.Not() for n in neighbors])
                objective_terms.append(ALPHA["S3"] * isolated)

    # S4: RoomStability -- 1 point per distinct room beyond the first, per course
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        room_used = []
        for ri in range(n_rooms):
            on_room = []
            for lec in lecs:
                b = model.NewBoolVar(f"onroom_{cid}_{lec[1]}_{ri}")
                model.Add(room_var[lec] == ri).OnlyEnforceIf(b)
                model.Add(room_var[lec] != ri).OnlyEnforceIf(b.Not())
                on_room.append(b)
            used = model.NewBoolVar(f"roomused_{cid}_{ri}")
            model.AddMaxEquality(used, on_room)
            room_used.append(used)
        distinct_rooms = model.NewIntVar(0, n_rooms, f"distinctrooms_{cid}")
        model.Add(distinct_rooms == sum(room_used))
        extra = model.NewIntVar(0, max(0, n_rooms - 1), f"extra_{cid}")
        model.Add(extra >= distinct_rooms - 1)
        objective_terms.append(ALPHA["S4"] * extra)

    # Create an explicit objective variable so we can add constraints like
    # "objective <= value" between solves when sampling.
    total_obj = model.NewIntVar(0, 10 ** 9, "total_objective")
    model.Add(total_obj == sum(objective_terms))
    return model, lectures, period_var, room_var, room_ids, total_obj


def extract_assignment(solver, lectures, period_var, room_var, room_ids, periods_per_day) -> dict:
    """Extracts assignment out of solver into dictionary per course lesson."""

    assignment = {}
    for lec in lectures:
                cid, li = lec
                p = solver.Value(period_var[lec])
                assignment[f"{cid}#{li}"] = {
                    "day": p // periods_per_day,
                    "period_in_day": p % periods_per_day,
                    "period": p,
                    "room": room_ids[solver.Value(room_var[lec])]}
    return assignment


def solve(problem: CTTProblem, time_limit: float = 60.0, workers: int = 8, log_progress: bool = False) -> dict:
    model, lectures, period_var, room_var, room_ids, total_obj = build_model(problem)
    model.minimize(total_obj)  # turn on optimization mode

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.log_search_progress = log_progress

    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    result = {
        "status": solver.StatusName(status),
        "objective": solver.Value(total_obj) if feasible else None,
        "best_bound": solver.BestObjectiveBound() if feasible else None,
        "solve_time": solver.WallTime(),
        "assignment": {},
    }
    if feasible:
        result["assignment"] = extract_assignment(solver, lectures, period_var, room_var, room_ids, problem.periods_per_day)
    return result


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects fixed number of different solutions found by solver."""

    def __init__(self, lectures, period_var, room_var, room_ids, periods_per_day, limit: int):
        cp_model.CpSolverSolutionCallback.__init__(self)

        self.__solution_count = 0
        self.__solution_limit = limit
        self.__solutions = {}  # collect solutions in output format
        self.lectures = lectures
        self.period_var = period_var
        self.room_var = room_var
        self.room_ids = room_ids
        self.periods_per_day = periods_per_day
        self._seen = set()

    def on_solution_callback(self) -> None:
        # extract current solution assignment
        assignment = extract_assignment(self, self.lectures, self.period_var, self.room_var, self.room_ids, self.periods_per_day)
        # canonical representation to detect duplicates
        rep = tuple((k, assignment[k]["day"], assignment[k]["period_in_day"], assignment[k]["period"], assignment[k]["room"]) for k in sorted(assignment.keys()))
        # ignore duplicate solution
        if rep in self._seen:
            return
        self._seen.add(rep) # new unique solution
        self.__solution_count += 1
        self.__solutions[f'assignment{self.__solution_count}'] = assignment
        print(f'Found {self.__solution_count}')
        if self.__solution_count >= self.__solution_limit:
            self.stop_search()
            print(f"Stopped search after {self.__solution_count} solutions")

    def solution_count(self) -> int:
        return self.__solution_count

    def solutions(self) -> dict:
        return self.__solutions


def sample_solutions(problem: CTTProblem, max_solutions: int=10, timeout: float=30) -> dict:
    """Samples given number of optimal solution. Overall time limit is timeout*(max_solutions+1)."""

    model, lectures, period_var, room_var, room_ids, total_obj = build_model(problem)
    opt_model = model.clone()
    opt_model.minimize(total_obj)  # add objective to minimize

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(opt_model)  # solve in optimization mode
    opt = solver.Value(total_obj)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError('No feasible solution found!')
    elif status != cp_model.OPTIMAL:
        print('No optimal solution found within time limit. Continuing with best one found.')
    else:
        print('Optimal solution found. Start sampling.')

    solutions = {"Initial status": solver.StatusName(status),
                "Initial objective": opt,
                "Initial best_bound": solver.BestObjectiveBound(),
                "Initial solve_time": solver.WallTime()}
    init_sol = extract_assignment(solver, lectures, period_var, room_var, room_ids, problem.periods_per_day)

    # enter sampling mode
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = max_solutions * timeout

    # warm start using initial solution as hint
    for i, v in enumerate(model.proto.variables):
        v_ = model.get_int_var_from_proto_index(i)
        assert v.name == v_.name, "Variable names should match"
        model.add_hint(v_, solver.value(v_))

    model.Add(total_obj <= opt)  # add optimality constraint for sampling mode

    solution_collector = SolutionCollector(lectures, period_var, room_var, room_ids, problem.periods_per_day, max_solutions)
    solver.SearchForAllSolutions(model, solution_collector)  # sample at most max_solutions many solutions
    samples = solution_collector.solutions()

    solutions["Sampling time"] = solver.WallTime()
    solutions["Number of samples"] = len(samples)

    # compute average pairwise differences between sampled assignments
    solutions["Avg pairwise differences"] = avg_differences(samples)

    solutions["Assignments"] = samples

    return solutions


def avg_differences(samples):
    """Compute differences between sampled solutions. Different period and room count as 2."""
    sample_list = [samples[k] for k in sorted(samples.keys())]
    n = len(sample_list)
    if n < 2:
        avg_diff = 0.0
    else:
        total_diff = 0
        pairs = 0
        for i in range(n):
            a = sample_list[i]
            for j in range(i + 1, n):
                b = sample_list[j]
                diff = 0
                for lec in sorted(a.keys()):
                    if a[lec]["period"] != b[lec]["period"]:
                        diff += 1
                    if a[lec]["room"] != b[lec]["room"]:
                        diff += 1
                total_diff += diff
                pairs += 1
        avg_diff = float(total_diff) / pairs if pairs else 0.0
    return avg_diff


def evaluate_assignment(problem: CTTProblem, period_of: Dict[Lecture, int], room_of: Dict[Lecture, str]) -> dict:
    """Scores an arbitrary (period, room) assignment against the H1-H4/S1-S4
    semantics build_model() encodes, without invoking the solver. period_of/
    room_of use solve()'s raw encoding (period=day*periods_per_day+period_in_day,
    room=room id). Hard violations count as group size minus distinct
    periods/rooms used, not just broken/not-broken."""
    periods_per_day = problem.periods_per_day
    lectures: List[Lecture] = [(cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)]

    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * periods_per_day + u.period)

    def _distinct_violations(groups: List[List[Lecture]]) -> int:
        total = 0
        for lecs in groups:
            if len(lecs) <= 1:
                continue
            periods = [period_of[l] for l in lecs]
            total += len(periods) - len(set(periods))
        return total

    h1 = _distinct_violations([[(cid, li) for li in range(c.n_lectures)] for cid, c in problem.courses.items()])
    h3a = _distinct_violations([
        [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        for q in problem.curricula.values()
    ])
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    h3b = _distinct_violations([
        [(cid, li) for cid in cids for li in range(problem.courses[cid].n_lectures)]
        for cids in teacher_courses.values()
    ])

    occ_count: Dict[Tuple[int, str], int] = {}
    for lec in lectures:
        key = (period_of[lec], room_of[lec])
        occ_count[key] = occ_count.get(key, 0) + 1
    h2 = sum(n - 1 for n in occ_count.values() if n > 1)

    h4 = sum(1 for lec in lectures if period_of[lec] in unavailable.get(lec[0], ()))

    hard_violations = h1 + h2 + h3a + h3b + h4

    s1 = sum(
        max(0, problem.courses[cid].n_students - problem.rooms[room_of[(cid, li)]].capacity)
        for cid, c in problem.courses.items() for li in range(c.n_lectures)
    )

    s2 = 0
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        days_used = {period_of[l] // periods_per_day for l in lecs}
        s2 += max(0, c.min_working_days - len(days_used))

    s3 = 0
    for q in problem.curricula.values():
        lecs = [(cid, li) for cid in q.course_ids for li in range(problem.courses[cid].n_lectures)]
        occ_slots = {(period_of[l] // periods_per_day, period_of[l] % periods_per_day) for l in lecs}
        for d, s in occ_slots:
            if (d, s - 1) not in occ_slots and (d, s + 1) not in occ_slots:
                s3 += 1

    s4 = 0
    for cid, c in problem.courses.items():
        lecs = [(cid, li) for li in range(c.n_lectures)]
        s4 += max(0, len({room_of[l] for l in lecs}) - 1)

    soft_objective = ALPHA["S1"] * s1 + ALPHA["S2"] * s2 + ALPHA["S3"] * s3 + ALPHA["S4"] * s4

    return dict(
        h1=h1, h2=h2, h3a=h3a, h3b=h3b, h4=h4, hard_violations=hard_violations,
        s1=s1, s2=s2, s3=s3, s4=s4, soft_objective=soft_objective,
    )


def solution_metrics(period_prob, room_prob, ctt_paths: List[str], problem_cache: dict) -> Dict[str, float]:
    """Averages evaluate_assignment() over a mini-batch (argmax predictions).
    ctt_paths[row] is the .ctt file that row belongs to; problem_cache
    memoizes parsed problems across calls."""
    from dataset.ctt_encode import CANONICAL_MAX_PERIODS_PER_DAY

    pred_period = period_prob.argmax(dim=-1).detach().cpu().numpy()
    pred_room = room_prob.argmax(dim=-1).detach().cpu().numpy()

    keys = ("h1", "h2", "h3a", "h3b", "h4", "hard_violations", "s1", "s2", "s3", "s4", "soft_objective")
    totals = {k: 0.0 for k in keys}

    for row, ctt_path in enumerate(ctt_paths):
        if ctt_path not in problem_cache:
            problem = parse_ctt(ctt_path)
            lectures = [(cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)]
            room_ids = list(problem.rooms.keys())
            problem_cache[ctt_path] = (problem, lectures, room_ids)
        problem, lectures, room_ids = problem_cache[ctt_path]

        period_of, room_of = {}, {}
        for i, lec in enumerate(lectures):
            day, slot = divmod(int(pred_period[row, i]), CANONICAL_MAX_PERIODS_PER_DAY)
            period_of[lec] = day * problem.periods_per_day + slot
            room_of[lec] = room_ids[int(pred_room[row, i])]

        result = evaluate_assignment(problem, period_of, room_of)
        for k in keys:
            totals[k] += result[k]

    n = len(ctt_paths)
    return {k: v / n for k, v in totals.items()} if n else totals


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("ctt_path", nargs="?", default="data/ITC2007/real/comp01.ctt")
    cli.add_argument("--time-limit", type=float, default=60.0)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--log-progress", action="store_true")
    cli.add_argument("--out", default=None)
    cli.add_argument("--samples", type=int, default=1, help="If >1, sample up to this many solutions.")
    args = cli.parse_args()

    problem = parse_ctt(args.ctt_path)

    if args.samples is None or args.samples <= 1:
        result = solve(problem, time_limit=args.time_limit, workers=args.workers, log_progress=args.log_progress)
        print(f"status={result['status']} objective={result['objective']} bound={result['best_bound']} "
              f"solve_time={result['solve_time']:.1f}s")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"wrote solution to {args.out}")

    else:
        sols = sample_solutions(problem, max_solutions=args.samples, timeout=args.time_limit)
        print(f"sampled {len(sols["Assignments"])} solutions (requested {args.samples})")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(sols, f, indent=2)
            print(f"wrote solutions to {args.out}")
