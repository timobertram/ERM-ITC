import glob
from typing import Dict, List

import numpy as np
import torch

TAG_DIM = 8


def expand_paths(patterns: List[str]) -> List[str]:
    """Sorted glob matches; errors on an empty pattern."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no files matched {pattern!r}")
        paths += matches
    return paths


def _pad_stack(arrays: List[np.ndarray]) -> np.ndarray:
    """Zero-pad arrays to a common shape and stack."""
    shape = np.max([a.shape for a in arrays], axis=0)
    out = np.zeros((len(arrays), *shape), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[(i, *map(slice, a.shape))] = a
    return out


def _length_mask(lengths: List[int]) -> np.ndarray:
    """Bool mask of valid positions per row."""
    return np.arange(max(lengths))[None, :] < np.array(lengths)[:, None]


def load_examples(paths: List[str]) -> List[Dict[str, np.ndarray]]:
    """Load encoded .npz files."""
    return [dict(np.load(p)) for p in paths]


def _lecture_tag(batch: Dict[str, np.ndarray], rng: np.random.Generator, mode: str) -> np.ndarray:
    """Random identity tag per course, per lecture, or none."""
    course = batch["course_membership"].astype(np.float32)
    B, N, C = course.shape
    if mode == "course":
        return course @ rng.standard_normal((B, C, TAG_DIM), dtype=np.float32)
    if mode == "lecture":
        return rng.standard_normal((B, N, TAG_DIM), dtype=np.float32) * batch["node_mask"][..., None]
    if mode == "none":
        return np.zeros((B, N, TAG_DIM), dtype=np.float32)
    raise ValueError(f"unknown lecture_tag {mode!r}")


def collate(examples: List[Dict[str, np.ndarray]], rng: np.random.Generator, lecture_tag: str = "course") -> Dict[str, torch.Tensor]:
    """Pad examples into a batch with node/room masks."""
    batch = {k: _pad_stack([ex[k] for ex in examples]) for k in examples[0]}
    batch["node_mask"] = _length_mask([len(ex["n_students"]) for ex in examples])
    batch["room_mask"] = _length_mask([len(ex["room_capacity"]) for ex in examples])
    batch["lecture_tag"] = _lecture_tag(batch, rng, lecture_tag)
    return {k: torch.from_numpy(v) for k, v in batch.items()}
