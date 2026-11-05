import zlib
from typing import Dict

import numpy as np

from dataset.ctt_parser import CTTProblem

MAX_DAYS = 6
MAX_PERIODS_PER_DAY = 9
MAX_LECTURES = 16
MAX_COURSES = 256
N_RANDOM = 8

COUNT_LOG_MEAN, COUNT_LOG_STD = 3.94, 1.02
MWD_LOG_MEAN, MWD_LOG_STD = 1.1689, 0.3620
LEC_LOG_MEAN, LEC_LOG_STD = 1.569, 0.2604


def _scale(x, mean, std):
    """log1p then fixed normalization."""
    return (np.log1p(np.asarray(x, dtype=np.float32)) - mean) / std


def encode(problem: CTTProblem, seed: int = 0) -> Dict[str, np.ndarray]:
    """CTTProblem -> lecture, room and membership arrays."""
    if problem.n_days > MAX_DAYS or problem.periods_per_day > MAX_PERIODS_PER_DAY:
        raise ValueError(f"{problem.name} exceeds the {MAX_DAYS}x{MAX_PERIODS_PER_DAY} period grid")
    if len(problem.courses) > MAX_COURSES:
        raise ValueError(f"{problem.name} has more than {MAX_COURSES} courses")
    if max(c.n_lectures for c in problem.courses.values()) > MAX_LECTURES:
        raise ValueError(f"{problem.name} has a course with more than {MAX_LECTURES} lectures")
    rng = np.random.default_rng(zlib.crc32(f"{problem.name}:{seed}".encode()))

    courses = list(problem.courses.values())
    course_idx = {c.id: i for i, c in enumerate(courses)}
    teacher_idx = {t: i for i, t in enumerate(dict.fromkeys(c.teacher_id for c in courses))}
    rooms = list(problem.rooms.values())
    lectures = [c for c in courses for _ in range(c.n_lectures)]
    lecture_index = np.array([li for c in courses for li in range(c.n_lectures)], dtype=np.int64)
    n_lec = len(lectures)

    unavailable = {(u.course_id, u.day, u.period) for u in problem.unavailability}
    period_mask = np.zeros((n_lec, MAX_DAYS, MAX_PERIODS_PER_DAY), dtype=bool)
    course_m = np.zeros((n_lec, len(courses)), dtype=bool)
    curriculum_m = np.zeros((n_lec, len(problem.curricula)), dtype=bool)
    teacher_m = np.zeros((n_lec, len(teacher_idx)), dtype=bool)
    curricula_of = {c.id: [] for c in courses}
    for qi, q in enumerate(problem.curricula.values()):
        for cid in q.course_ids:
            curricula_of[cid].append(qi)

    for i, c in enumerate(lectures):
        for d in range(problem.n_days):
            for s in range(problem.periods_per_day):
                period_mask[i, d, s] = (c.id, d, s) not in unavailable
        course_m[i, course_idx[c.id]] = True
        curriculum_m[i, curricula_of[c.id]] = True
        teacher_m[i, teacher_idx[c.teacher_id]] = True

    n_students = np.array([c.n_students for c in lectures], dtype=np.float32)
    min_working_days = np.array([c.min_working_days for c in courses], dtype=np.float32)
    room_capacity = np.array([r.capacity for r in rooms], dtype=np.float32)

    lecture_features = np.concatenate([
        _scale(n_students, COUNT_LOG_MEAN, COUNT_LOG_STD)[:, None],
        _scale([c.min_working_days for c in lectures], MWD_LOG_MEAN, MWD_LOG_STD)[:, None],
        _scale([c.n_lectures for c in lectures], LEC_LOG_MEAN, LEC_LOG_STD)[:, None],
    ], axis=1)
    room_features = np.concatenate([
        _scale(room_capacity, COUNT_LOG_MEAN, COUNT_LOG_STD)[:, None],
        rng.standard_normal((len(rooms), N_RANDOM)).astype(np.float32),
    ], axis=1)

    return dict(
        lecture_features=lecture_features,
        lecture_index=lecture_index,
        room_features=room_features,
        period_mask=period_mask.reshape(n_lec, -1),
        course_membership=course_m,
        curriculum_membership=curriculum_m,
        teacher_membership=teacher_m,
        n_students=n_students,
        min_working_days=min_working_days,
        room_capacity=room_capacity,
    )


if __name__ == "__main__":
    import argparse

    from dataset.ctt_parser import parse_ctt

    cli = argparse.ArgumentParser()
    cli.add_argument("ctt_paths", nargs="+")
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args()

    for path in args.ctt_paths:
        np.savez(path.removesuffix(".ctt") + "_encoded.npz", **encode(parse_ctt(path), args.seed))
    print(f"encoded {len(args.ctt_paths)} instances")
