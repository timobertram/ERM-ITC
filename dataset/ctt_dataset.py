"""Batch loader for CB-CTT instances encoded by dataset.ctt_encode.

Pads the lecture and room axes to the batch max; the period-vocab axis is
already uniform across instances (shared/global by construction).
"""
import glob
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


def expand_paths(npz_paths: List[str]) -> List[str]:
    """Expands glob patterns; a literal path with no wildcard matches itself."""
    expanded = []
    for pattern in npz_paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no files matched {pattern!r}")
        expanded.extend(matches)
    return expanded


@dataclass
class CTTExample:
    lecture_course_features: np.ndarray  # [N, F] -- real per-lecture features
    lecture_random_id: np.ndarray        # [N, R] -- symmetry-breaker, own projection (see trm_ctt.py)
    lecture_group_tag: np.ndarray         # [N, G] -- curriculum/teacher relatedness tag, own projection
    course_index: np.ndarray            # [N]
    period_candidate_mask: np.ndarray    # [N, V]
    period_label: np.ndarray             # [N]
    room_features: np.ndarray            # [R, 1]
    room_random_id: np.ndarray            # [R, G] -- symmetry-breaker, own projection (see trm_ctt.py)
    room_label: np.ndarray               # [N]


def load_examples(npz_paths: List[str]) -> List[CTTExample]:
    examples = []
    max_days = max_periods_per_day = None
    for path in expand_paths(npz_paths):
        data = np.load(path)
        if max_days is None:
            max_days, max_periods_per_day = int(data["max_days"]), int(data["max_periods_per_day"])
        elif (int(data["max_days"]), int(data["max_periods_per_day"])) != (max_days, max_periods_per_day):
            raise ValueError(
                f"{path} was encoded against a different shared vocab than {npz_paths[0]} -- "
                f"re-encode all instances together (see ctt_encode.py's CLI)."
            )
        examples.append(CTTExample(
            lecture_course_features=data["lecture_course_features"],
            lecture_random_id=data["lecture_random_id"],
            lecture_group_tag=data["lecture_group_tag"],
            course_index=data["course_index"],
            period_candidate_mask=data["period_candidate_mask"],
            period_label=data["period_label"],
            room_features=data["room_features"],
            room_random_id=data["room_random_id"],
            room_label=data["room_label"],
        ))
    return examples


def collate_batch(examples: List[CTTExample]) -> Dict[str, torch.Tensor]:
    max_lec = max(ex.lecture_course_features.shape[0] for ex in examples)
    max_room = max(ex.room_features.shape[0] for ex in examples)
    n_course_features = examples[0].lecture_course_features.shape[1]
    n_random_dims = examples[0].lecture_random_id.shape[1]
    n_group_dims = examples[0].lecture_group_tag.shape[1]
    vocab_size = examples[0].period_candidate_mask.shape[1]
    batch_size = len(examples)

    lecture_course_features = np.zeros((batch_size, max_lec, n_course_features), dtype=np.float32)
    lecture_random_id = np.zeros((batch_size, max_lec, n_random_dims), dtype=np.float32)
    lecture_group_tag = np.zeros((batch_size, max_lec, n_group_dims), dtype=np.float32)
    course_index = np.zeros((batch_size, max_lec), dtype=np.int64)
    period_candidate_mask = np.zeros((batch_size, max_lec, vocab_size), dtype=bool)
    period_label = np.zeros((batch_size, max_lec), dtype=np.int64)
    room_label = np.zeros((batch_size, max_lec), dtype=np.int64)
    node_mask = np.zeros((batch_size, max_lec), dtype=bool)

    n_room_random_dims = examples[0].room_random_id.shape[1]
    room_features = np.zeros((batch_size, max_room, 1), dtype=np.float32)
    room_random_id = np.zeros((batch_size, max_room, n_room_random_dims), dtype=np.float32)
    room_mask = np.zeros((batch_size, max_room), dtype=bool)

    for row, ex in enumerate(examples):
        n_lec = ex.lecture_course_features.shape[0]
        n_room = ex.room_features.shape[0]
        lecture_course_features[row, :n_lec] = ex.lecture_course_features
        lecture_random_id[row, :n_lec] = ex.lecture_random_id
        lecture_group_tag[row, :n_lec] = ex.lecture_group_tag
        course_index[row, :n_lec] = ex.course_index
        period_candidate_mask[row, :n_lec] = ex.period_candidate_mask
        period_label[row, :n_lec] = ex.period_label
        room_label[row, :n_lec] = ex.room_label
        node_mask[row, :n_lec] = True

        room_features[row, :n_room] = ex.room_features
        room_random_id[row, :n_room] = ex.room_random_id
        room_mask[row, :n_room] = True

    return dict(
        lecture_course_features=torch.from_numpy(lecture_course_features),
        lecture_random_id=torch.from_numpy(lecture_random_id),
        lecture_group_tag=torch.from_numpy(lecture_group_tag),
        course_index=torch.from_numpy(course_index),
        period_candidate_mask=torch.from_numpy(period_candidate_mask),
        period_label=torch.from_numpy(period_label),
        room_label=torch.from_numpy(room_label),
        node_mask=torch.from_numpy(node_mask),
        room_features=torch.from_numpy(room_features),
        room_random_id=torch.from_numpy(room_random_id),
        room_mask=torch.from_numpy(room_mask),
    )


def load_batch(npz_paths: List[str]) -> Dict[str, torch.Tensor]:
    return collate_batch(load_examples(npz_paths))


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] or ["data/ITC2007/real/comp01_encoded.npz"]
    batch = load_batch(paths)
    for k, v in batch.items():
        print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
    print("lectures per instance (node_mask sum):", batch["node_mask"].sum(dim=1).tolist())
    print("rooms per instance (room_mask sum):", batch["room_mask"].sum(dim=1).tolist())
