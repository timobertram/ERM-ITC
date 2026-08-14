"""Generates synthetic CB-CTT instances (same CTTProblem/.ctt format as the
real data).

Cheap necessary-condition checks reject obviously-infeasible instances
before calling the solver; dataset.ctt_solver is the final feasibility
oracle.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from dataset.ctt_parser import (
    CTTProblem, Course, Room, Curriculum, UnavailabilityConstraint,
)


@dataclass
class SizeRange:
    n_days: List[int]
    periods_per_day: List[int]
    n_courses: range
    n_rooms: range
    curricula_per_course: tuple      # (min, max) ratio of n_curricula to n_courses
    curriculum_size: range          # courses per curriculum
    lectures_per_course: List[int]  # sampled with replacement
    n_students: range
    room_capacity: range
    unavailability_frac: tuple       # (min, max) fraction of a course's periods blocked


# Roughly calibrated against comp01-21's real stats.
SMALL = SizeRange(
    n_days=[5, 5, 5, 6], periods_per_day=[5, 5, 5, 6],
    n_courses=range(4, 11), n_rooms=range(2, 5),
    curricula_per_course=(0.5, 1.2),
    curriculum_size=range(1, 4),
    lectures_per_course=[2, 3, 3, 3, 4, 5],
    n_students=range(10, 60), room_capacity=range(20, 100),
    unavailability_frac=(0.0, 0.15),
)

MEDIUM = SizeRange(
    n_days=[5, 5, 5, 6], periods_per_day=[5, 5, 5, 6, 6],
    n_courses=range(20, 60), n_rooms=range(6, 14),
    curricula_per_course=(0.5, 1.2),
    curriculum_size=range(1, 5),
    lectures_per_course=[2, 3, 3, 3, 3, 4, 5, 6],
    n_students=range(15, 120), room_capacity=range(20, 150),
    unavailability_frac=(0.05, 0.30),
)

# extends room_capacity/n_students to real comp01-21's actual range (up to 450/440)
LARGE = SizeRange(
    n_days=[5, 5, 5, 6], periods_per_day=[5, 5, 6, 6, 9],
    n_courses=range(50, 140), n_rooms=range(10, 21),
    curricula_per_course=(0.5, 1.2),
    curriculum_size=range(1, 6),
    lectures_per_course=[2, 3, 3, 3, 3, 4, 5, 6],
    n_students=range(10, 320), room_capacity=range(30, 460),
    unavailability_frac=(0.1, 0.35),
)

SIZES = {"small": SMALL, "medium": MEDIUM, "large": LARGE}


def _feasible_necessary_conditions(problem: CTTProblem) -> bool:
    """Checks for H1/H3a/H3b (all AllDifferent-over-periods,
    so exact and free). """
    n_periods = problem.n_periods
    unavailable_count = {cid: 0 for cid in problem.courses}
    for u in problem.unavailability:
        unavailable_count[u.course_id] += 1

    for cid, c in problem.courses.items():
        if n_periods - unavailable_count[cid] < c.n_lectures:
            return False  # H1

    for q in problem.curricula.values():
        if sum(problem.courses[cid].n_lectures for cid in q.course_ids) > n_periods:
            return False  # H3a

    teacher_lectures: dict = {}
    for c in problem.courses.values():
        teacher_lectures[c.teacher_id] = teacher_lectures.get(c.teacher_id, 0) + c.n_lectures
    if any(total > n_periods for total in teacher_lectures.values()):
        return False  # H3b

    total_lecture_demand = sum(c.n_lectures for c in problem.courses.values())
    total_room_period_supply = len(problem.rooms) * n_periods
    return total_lecture_demand <= total_room_period_supply


def generate_problem(rng: np.random.Generator, name: str, size: SizeRange, max_attempts: int = 20) -> CTTProblem:
    for _attempt in range(max_attempts):
        n_days = int(rng.choice(size.n_days))
        periods_per_day = int(rng.choice(size.periods_per_day))
        n_periods = n_days * periods_per_day

        n_courses = int(rng.choice(size.n_courses))
        n_rooms = int(rng.choice(size.n_rooms))
        n_teachers = max(1, int(n_courses * rng.uniform(0.5, 0.85)))  # some teachers teach >1 course

        course_ids = [f"c{i}" for i in range(n_courses)]
        courses = {}
        for i, cid in enumerate(course_ids):
            n_lectures = int(rng.choice(size.lectures_per_course))
            courses[cid] = Course(
                id=cid,
                teacher_id=f"t{int(rng.integers(0, n_teachers))}",
                n_lectures=n_lectures,
                # must not exceed n_lectures, else S2's shortfall is unavoidable
                min_working_days=int(rng.integers(1, min(n_lectures, n_days) + 1)),
                n_students=int(rng.choice(size.n_students)),
            )

        rooms = {
            f"r{i}": Room(id=f"r{i}", capacity=int(rng.choice(size.room_capacity)))
            for i in range(n_rooms)
        }

        # scaled relative to n_courses (real ratio is 0.43-2.57, median 0.72)
        lo, hi = size.curricula_per_course
        n_curricula = max(1, int(rng.integers(max(1, round(lo * n_courses)), round(hi * n_courses) + 2)))
        curricula = {}
        for i in range(n_curricula):
            qsize = min(n_courses, int(rng.choice(size.curriculum_size)))
            members = list(rng.choice(course_ids, size=qsize, replace=False))
            curricula[f"q{i}"] = Curriculum(id=f"q{i}", course_ids=members)

        unavailability = []
        for cid in course_ids:
            frac = rng.uniform(*size.unavailability_frac)
            n_blocked = int(round(frac * n_periods))
            if n_blocked == 0:
                continue
            blocked_periods = rng.choice(n_periods, size=n_blocked, replace=False)
            for p in blocked_periods:
                unavailability.append(UnavailabilityConstraint(
                    course_id=cid, day=int(p) // periods_per_day, period=int(p) % periods_per_day,
                ))

        problem = CTTProblem(
            name=name, n_days=n_days, periods_per_day=periods_per_day,
            courses=courses, rooms=rooms, curricula=curricula, unavailability=unavailability,
        )
        if _feasible_necessary_conditions(problem):
            return problem

    raise RuntimeError(f"could not generate a plausibly-feasible instance for {name} in {max_attempts} attempts")


def write_ctt(problem: CTTProblem, path: str) -> None:
    lines = [
        f"Name: {problem.name}",
        f"Courses: {len(problem.courses)}",
        f"Rooms: {len(problem.rooms)}",
        f"Days: {problem.n_days}",
        f"Periods_per_day: {problem.periods_per_day}",
        f"Curricula: {len(problem.curricula)}",
        f"Constraints: {len(problem.unavailability)}",
        "",
        "COURSES:",
    ]
    for c in problem.courses.values():
        lines.append(f"{c.id} {c.teacher_id} {c.n_lectures} {c.min_working_days} {c.n_students}")
    lines += ["", "ROOMS:"]
    for r in problem.rooms.values():
        lines.append(f"{r.id} {r.capacity}")
    lines += ["", "CURRICULA:"]
    for q in problem.curricula.values():
        lines.append(f"{q.id} {len(q.course_ids)} " + " ".join(q.course_ids))
    lines += ["", "UNAVAILABILITY_CONSTRAINTS:"]
    for u in problem.unavailability:
        lines.append(f"{u.course_id} {u.day} {u.period}")
    lines += ["", "END."]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _generate_and_solve_one(task):
    """Picklable worker for parallel generation: each call seeds its own rng,
    generates one instance, solves it. Returns None on infeasible/unsolved
    (including generate_problem giving up on its feasibility pre-checks)."""
    seed, name, size, time_limit, solver_workers = task
    rng = np.random.default_rng(seed)
    try:
        problem = generate_problem(rng, name, size)
    except RuntimeError:
        return None
    from dataset.ctt_solver import solve
    result = solve(problem, time_limit=time_limit, workers=solver_workers, log_progress=False)
    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        return None
    return problem, result


if __name__ == "__main__":
    import argparse
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    from dataset.ctt_parser import parse_ctt

    cli = argparse.ArgumentParser()
    cli.add_argument("--count", type=int, default=5)
    cli.add_argument("--size", choices=list(SIZES), default="small")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--out-dir", default="data/ITC2007/synth")
    cli.add_argument("--time-limit", type=float, default=20.0)
    cli.add_argument("--prefix", default="gen")
    cli.add_argument("--jobs", type=int, default=1, help="parallel instance-solves in flight at once")
    cli.add_argument("--solver-workers", type=int, default=8,
                      help="OR-tools search workers per instance -- lower when --jobs > 1")
    args = cli.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    size = SIZES[args.size]
    name_width = max(3, len(str(args.count - 1)))

    def _write_result(name, problem, result):
        ctt_path = os.path.join(args.out_dir, f"{name}.ctt")
        with open(os.path.join(args.out_dir, f"{name}_solution.json"), "w") as f:
            json.dump(result, f, indent=2)
        write_ctt(problem, ctt_path)
        # round-trip check: re-parsing our own .ctt must reproduce the same problem
        reparsed = parse_ctt(ctt_path)
        assert reparsed == problem, f"round-trip mismatch for {name}: {reparsed} != {problem}"

    n_written = n_rejected = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, tqdm(total=args.count, unit="instance") as pbar:
        in_flight = {}
        seed_box = [args.seed]  # mutable cell -- module-level code has no function scope for `nonlocal`

        def _submit_next():
            name = f"{args.prefix}{(n_written + len(in_flight)):0{name_width}d}"
            fut = pool.submit(_generate_and_solve_one, (seed_box[0], name, size, args.time_limit, args.solver_workers))
            in_flight[fut] = name
            seed_box[0] += 1

        for _ in range(min(args.jobs * 2, args.count)):
            _submit_next()

        while n_written < args.count:
            fut = next(as_completed(in_flight))
            del in_flight[fut]
            outcome = fut.result()

            if outcome is None:
                n_rejected += 1
                pbar.set_postfix(rejected=n_rejected)
            else:
                problem, result = outcome
                final_name = f"{args.prefix}{n_written:0{name_width}d}"
                problem.name = final_name
                _write_result(final_name, problem, result)
                n_written += 1
                pbar.set_postfix(rejected=n_rejected, solve_time=f"{result['solve_time']:.1f}s")
                pbar.update(1)

            if n_written < args.count:
                _submit_next()

    print(f"wrote {n_written} instances to {args.out_dir} ({n_rejected} rejected as infeasible/unsolved)")
