"""Turns a parsed CTTProblem + its solved assignment into dense, padded
tensors for models/recursive_reasoning/trm_ctt.py.

One row per lecture. Curriculum/teacher relations are baked into each
lecture's feature vector as random per-group tags (see `lecture_group_tag`
below).. Periods use a shared global vocab
(fixed classification head); rooms are per-instance, scored via dot product
against a small per-instance room table instead.
"""
from typing import Dict, List, Tuple
import numpy as np

from dataset.ctt_parser import CTTProblem

N_RANDOM_ID_DIMS = 8  # random per-lecture symmetry-breaker width
N_GROUP_TAG_DIMS = 8  # random per-curriculum/per-teacher relatedness-tag width
N_ROOM_RANDOM_ID_DIMS = 8  # random per-room symmetry-breaker width

Lecture = Tuple[str, int]


def _standardize(x: np.ndarray) -> np.ndarray:
    """z-score per feature column -- keeps lecture/room identity from
    collapsing before the model even runs."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + 1e-4)


# fixed (not per-instance) log1p scale for room capacity + class size, same units
CAPACITY_LOG_MEAN = 3.94
CAPACITY_LOG_STD = 1.02


def _scale_capacity(x) -> np.ndarray:
    return (np.log1p(x) - CAPACITY_LOG_MEAN) / CAPACITY_LOG_STD


# Fixed bounds covering every comp01-21 instance, so any subset can be
# encoded independently and still share one vocab.
CANONICAL_MAX_DAYS = 6
CANONICAL_MAX_PERIODS_PER_DAY = 9


def global_period_dims(problems: List[CTTProblem]) -> Tuple[int, int]:
    """Returns the canonical (max_days, max_periods_per_day); raises if any
    problem exceeds it."""
    for p in problems:
        if p.n_days > CANONICAL_MAX_DAYS or p.periods_per_day > CANONICAL_MAX_PERIODS_PER_DAY:
            raise ValueError(
                f"instance exceeds canonical period vocab (n_days={p.n_days}, "
                f"periods_per_day={p.periods_per_day} vs canonical "
                f"{CANONICAL_MAX_DAYS}x{CANONICAL_MAX_PERIODS_PER_DAY})"
            )
    return CANONICAL_MAX_DAYS, CANONICAL_MAX_PERIODS_PER_DAY


def encode_instance(problem: CTTProblem, solution: dict, max_days: int, max_periods_per_day: int) -> dict:
    lectures: List[Lecture] = [
        (cid, li) for cid, c in problem.courses.items() for li in range(c.n_lectures)
    ]
    n_lectures = len(lectures)
    room_ids = list(problem.rooms.keys())
    n_rooms = len(room_ids)

    curriculum_degree = {cid: 0 for cid in problem.courses}
    for q in problem.curricula.values():
        for cid in q.course_ids:
            curriculum_degree[cid] += len(q.course_ids) - 1
    teacher_courses: Dict[str, List[str]] = {}
    for cid, c in problem.courses.items():
        teacher_courses.setdefault(c.teacher_id, []).append(cid)
    teacher_degree = {cid: len(teacher_courses[c.teacher_id]) - 1 for cid, c in problem.courses.items()}

    # one random tag per curriculum/teacher; a course's tag is the sum over
    # its curricula (it may be in several) plus its one teacher's tag
    curriculum_tag = {qid: np.random.randn(N_GROUP_TAG_DIMS).astype(np.float32) for qid in problem.curricula}
    teacher_tag = {tid: np.random.randn(N_GROUP_TAG_DIMS).astype(np.float32) for tid in teacher_courses}
    course_group_tag = {cid: np.zeros(N_GROUP_TAG_DIMS, dtype=np.float32) for cid in problem.courses}
    for qid, q in problem.curricula.items():
        for cid in q.course_ids:
            course_group_tag[cid] = course_group_tag[cid] + curriculum_tag[qid]
    for cid, c in problem.courses.items():
        course_group_tag[cid] = course_group_tag[cid] + teacher_tag[c.teacher_id]

    course_features = np.zeros((n_lectures, 4), dtype=np.float32)
    for i, (cid, _li) in enumerate(lectures):
        c = problem.courses[cid]
        course_features[i] = [
            _scale_capacity(c.n_students), np.log1p(c.min_working_days),
            np.log1p(curriculum_degree[cid]), np.log1p(teacher_degree[cid]),
        ]
    course_features[:, 1:] = _standardize(course_features[:, 1:])  # col 0 (n_students) already fixed-scale
    # kept separate from course_features (own projection in the model) since
    # it's already N(0,1) by construction
    random_id = np.random.randn(n_lectures, N_RANDOM_ID_DIMS).astype(np.float32)
    group_tag = np.stack([course_group_tag[cid] for cid, _li in lectures])  # [n_lec, N_GROUP_TAG_DIMS]

    course_ids_list = list(problem.courses.keys())
    course_to_idx = {cid: i for i, cid in enumerate(course_ids_list)}
    course_index = np.array([course_to_idx[cid] for cid, _li in lectures], dtype=np.int64)

    room_features = np.array(
        [[_scale_capacity(problem.rooms[rid].capacity)] for rid in room_ids], dtype=np.float32
    )
    # capacity alone is a single scalar (rank-1), so a random per-room tag is
    # added too, same fix as the lecture symmetry-breaker above
    room_random_id = np.random.randn(n_rooms, N_ROOM_RANDOM_ID_DIMS).astype(np.float32)

    unavailable: Dict[str, set] = {}
    for u in problem.unavailability:
        unavailable.setdefault(u.course_id, set()).add(u.day * problem.periods_per_day + u.period)

    vocab_size = max_days * max_periods_per_day
    period_candidate_mask = np.zeros((n_lectures, vocab_size), dtype=bool)
    period_label = np.zeros(n_lectures, dtype=np.int64)
    room_label = np.zeros(n_lectures, dtype=np.int64)

    for i, (cid, li) in enumerate(lectures):
        blocked = unavailable.get(cid, ())
        for p in range(problem.n_periods):
            if p in blocked:
                continue
            day, slot = p // problem.periods_per_day, p % problem.periods_per_day
            period_candidate_mask[i, day * max_periods_per_day + slot] = True

        entry = solution["assignment"][f"{cid}#{li}"]
        true_p = entry["period"]
        day, slot = true_p // problem.periods_per_day, true_p % problem.periods_per_day
        period_label[i] = day * max_periods_per_day + slot
        room_label[i] = room_ids.index(entry["room"])

    return dict(
        lecture_course_features=course_features,  # [n_lec, 4]
        lecture_random_id=random_id,               # [n_lec, 8]
        lecture_group_tag=group_tag,                # [n_lec, 8]
        course_index=course_index,                # [n_lec]
        period_candidate_mask=period_candidate_mask,  # [n_lec, vocab_size]
        period_label=period_label,                 # [n_lec]
        room_features=room_features,               # [n_room, 1]
        room_random_id=room_random_id,              # [n_room, 8]
        room_label=room_label,                      # [n_lec]
        max_days=np.int64(max_days),
        max_periods_per_day=np.int64(max_periods_per_day),
    )


if __name__ == "__main__":
    import argparse
    import glob
    import json
    import os

    from dataset.ctt_parser import parse_ctt

    cli = argparse.ArgumentParser(description="Encode one or more .ctt instances against a SHARED period vocab.")
    cli.add_argument("names", nargs="+", help="instance names, e.g. comp01 comp02 (expects <data-dir>/<name>.ctt "
                                                "and <data-dir>/<name>_solution.json) -- also accepts glob patterns "
                                                "like 'comp*' instead of listing every name individually")
    cli.add_argument("--data-dir", default="data/ITC2007/real")
    args = cli.parse_args()

    names = []
    for pattern in args.names:
        matches = sorted(glob.glob(os.path.join(args.data_dir, f"{pattern}.ctt")))
        if not matches:
            raise FileNotFoundError(f"no .ctt files matched {pattern!r} in {args.data_dir}")
        names.extend(os.path.splitext(os.path.relpath(m, args.data_dir))[0] for m in matches)

    problems = [parse_ctt(os.path.join(args.data_dir, f"{name}.ctt")) for name in names]
    max_days, max_periods_per_day = global_period_dims(problems)
    print(f"shared vocab: max_days={max_days} max_periods_per_day={max_periods_per_day} "
          f"(vocab_size={max_days * max_periods_per_day}) across {len(names)} instances")

    for name, problem in zip(names, problems):
        with open(os.path.join(args.data_dir, f"{name}_solution.json")) as f:
            solution = json.load(f)
        data = encode_instance(problem, solution, max_days, max_periods_per_day)
        out_path = os.path.join(args.data_dir, f"{name}_encoded.npz")
        np.savez(out_path, **data)
        print(f"{name:8s} lectures={data['lecture_course_features'].shape[0]:4d} "
              f"rooms={data['room_features'].shape[0]:3d}  saved to {out_path}")
