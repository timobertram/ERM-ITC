from typing import Dict, List
from dataclasses import dataclass
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from models.common import trunc_normal_init_
from models.layers import rms_norm, CastedLinear, SwiGLU, Attention

# TRM for CB-CTT (ITC2007): one recursive latent per lecture, refined by
# dense self-attention over the whole lecture sequence. Same-curriculum/
# same-teacher relations are baked into each lecture's feature vector at
# encode time rather than modeled as graph edges. Periods decode via a
# fixed linear head (global vocab); rooms decode via a dot product against
# that instance's own room table (per-instance, so no fixed head). Lectures
# within a course are interchangeable, so training uses a Hungarian-matched
# period loss.

LOGIT_MASK_VALUE = -1e9  # not literal -inf: avoids NaN gradients through log(0)


@dataclass
class TRMCTTInnerCarry:
    z: torch.Tensor  # [B, max_lec, hidden_size] -- single latent, see module docstring


@dataclass
class TRMCTTCarry:
    inner_carry: TRMCTTInnerCarry
    current_data: Dict[str, torch.Tensor]  # which instance each ACT "slot" is currently reasoning about
    steps: torch.Tensor    # [B]
    halted: torch.Tensor   # [B]


class TRMCTTConfig(BaseModel):
    hidden_size: int
    expansion: float
    num_heads: int
    head_dim: int

    course_feature_dim: int  # width of dataset.ctt_encode's real per-lecture feature vector
    random_id_dim: int        # width of the per-lecture symmetry-breaker vector
    group_tag_dim: int         # width of the per-curriculum/per-teacher relatedness tag
    room_random_id_dim: int     # width of the per-room symmetry-breaker vector
    period_vocab_size: int   # max_days * max_periods_per_day, shared across instances

    L_layers: int
    H_cycles: int  # T in the TRM paper: outer cycles, T-1 run under no_grad
    L_cycles: int  # n in the TRM paper: z-refinement steps per outer cycle

    halt_max_steps: int
    halt_exploration_prob: float
    halt_max_prob: float

    rms_norm_eps: float = 1e-5
    forward_dtype: str = "float32"


class TRMCTTBlock(nn.Module):
    def __init__(self, config: TRMCTTConfig) -> None:
        super().__init__()
        self.attn = Attention(
            hidden_size=config.hidden_size, head_dim=config.head_dim,
            num_heads=config.num_heads, num_key_value_heads=config.num_heads, causal=False,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        # padding lectures must never be attended *to*; fine if padding
        # rows attend everywhere since their own output is discarded downstream
        attn_mask = node_mask.view(B, 1, 1, N)

        hidden_states = rms_norm(
            hidden_states + self.attn(hidden_states=hidden_states, mask=attn_mask),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class TRMCTTReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMCTTBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMCTT_Inner(nn.Module):
    def __init__(self, config: TRMCTTConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.feature_proj = CastedLinear(config.course_feature_dim, config.hidden_size, bias=False)
        self.random_proj = CastedLinear(config.random_id_dim, config.hidden_size, bias=False)
        # curriculum/teacher relatedness tag (see dataset/ctt_encode.py)
        self.group_proj = CastedLinear(config.group_tag_dim, config.hidden_size, bias=False)
        self.room_proj = CastedLinear(1, config.hidden_size, bias=False)
        # capacity alone is rank-1; a random per-room tag breaks the symmetry
        self.room_random_proj = CastedLinear(config.room_random_id_dim, config.hidden_size, bias=False)

        # period head: shared global vocab -> one learned embedding table,
        # scored via query/key dot product (same pattern as the room head)
        self.period_embed = nn.Parameter(
            trunc_normal_init_(torch.empty(config.period_vocab_size, config.hidden_size, dtype=self.forward_dtype), std=1)
        )
        self.period_query_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.period_key_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        # room head: per-instance table (no shared vocab), scored the same way
        self.room_query_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.room_key_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        self.L_level = TRMCTTReasoningModule(
            layers=[TRMCTTBlock(config) for _ in range(config.L_layers)]
        )

        self.Z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

    def empty_carry(self, batch_size: int, max_lec: int, device) -> TRMCTTInnerCarry:
        return TRMCTTInnerCarry(
            z=torch.empty(batch_size, max_lec, self.config.hidden_size, dtype=self.forward_dtype, device=device)
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMCTTInnerCarry) -> TRMCTTInnerCarry:
        return TRMCTTInnerCarry(
            z=torch.where(reset_flag.view(-1, 1, 1), self.Z_init, carry.z),
        )

    def forward(self, carry: TRMCTTInnerCarry, batch: Dict[str, torch.Tensor]):
        course_features = batch["lecture_course_features"].to(self.forward_dtype)  # [B, N, course_feature_dim]
        random_id = batch["lecture_random_id"].to(self.forward_dtype)              # [B, N, random_id_dim]
        group_tag = batch["lecture_group_tag"].to(self.forward_dtype)              # [B, N, group_tag_dim]
        node_mask = batch["node_mask"]                                        # [B, N] bool
        period_candidate_mask = batch["period_candidate_mask"]                # [B, N, V] bool
        room_features = batch["room_features"].to(self.forward_dtype)         # [B, R, 1]
        room_random_id = batch["room_random_id"].to(self.forward_dtype)       # [B, R, room_random_id_dim]
        room_mask = batch["room_mask"]                                        # [B, R] bool

        input_embed = self.feature_proj(course_features) + self.random_proj(random_id) + self.group_proj(group_tag)
        room_embed = self.room_proj(room_features) + self.room_random_proj(room_random_id)  # [B, R, hidden]

        z = carry.z
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z = self.L_level(z, input_embed, node_mask=node_mask)
        for _ in range(self.config.L_cycles):
            z = self.L_level(z, input_embed, node_mask=node_mask)

        new_carry = TRMCTTInnerCarry(z=z.detach())  # New carry, no grad

        has_candidate = period_candidate_mask.any(dim=-1, keepdim=True)  # [B, N, 1]

        period_scale = self.config.hidden_size ** -0.5
        period_query = self.period_query_proj(z)  # [B, N, hidden]
        period_key = self.period_key_proj(self.period_embed.to(z.dtype))  # [V, hidden]
        period_logits = torch.matmul(period_query, period_key.t()) * period_scale  # [B, N, V]
        period_logits = period_logits.masked_fill(~(period_candidate_mask | ~has_candidate), LOGIT_MASK_VALUE)
        period_prob = F.softmax(period_logits.to(torch.float32), dim=-1)

        room_scale = self.config.hidden_size ** -0.5
        room_query = self.room_query_proj(z)  # [B, N, hidden]
        room_key = self.room_key_proj(room_embed)  # [B, R, hidden]
        room_scores = torch.bmm(room_query, room_key.transpose(-2, -1)) * room_scale  # [B, N, R]
        room_scores = room_scores.masked_fill(~room_mask.unsqueeze(1), LOGIT_MASK_VALUE)
        room_prob = F.softmax(room_scores.to(torch.float32), dim=-1)

        with torch.no_grad():
            max_period_prob = period_prob.amax(dim=-1)  # [B, N]
            max_room_prob = room_prob.amax(dim=-1)       # [B, N]
            lecture_confidence = max_period_prob * max_room_prob
            confidence = (
                torch.where(node_mask, lecture_confidence, torch.zeros_like(lecture_confidence)).sum(dim=-1)
                / node_mask.sum(dim=-1).clamp_min(1)
            )

        return new_carry, period_prob, room_prob, confidence


class TRMCTT(nn.Module):
    """ACT wrapper. Each slot tracks its own current_data, swapped in only
    when that slot's `halted` flag fires, so training can offer a fresh
    candidate batch every call while a slot still mid-reasoning keeps
    working on its existing instance."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMCTTConfig(**config_dict)
        self.inner = TRMCTT_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCTTCarry:
        B, max_lec = batch["lecture_course_features"].shape[:2]
        device = batch["lecture_course_features"].device
        return TRMCTTCarry(
            inner_carry=self.inner.empty_carry(B, max_lec, device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            steps=torch.zeros(B, dtype=torch.int32, device=device),
            halted=torch.ones(B, dtype=torch.bool, device=device),  # forces a full swap-in on the first call
        )

    def forward(self, carry: TRMCTTCarry, batch: Dict[str, torch.Tensor]):
        # `batch` is a freshly-sampled candidate per slot; only halted slots adopt it
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (v.ndim - 1)), v, carry.current_data[k])
            for k, v in batch.items()
        }

        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        new_inner_carry, period_prob, room_prob, confidence = self.inner(new_inner_carry, new_current_data)
        outputs = dict(period_prob=period_prob, room_prob=room_prob)

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = new_steps >= self.config.halt_max_steps
            if self.training and self.config.halt_max_steps > 1:
                rand_stop = torch.rand_like(confidence) < self.config.halt_exploration_prob
                if self.config.halt_max_prob > 0:
                    halted = halted | (confidence > self.config.halt_max_prob) | rand_stop
                else:
                    halted = halted | rand_stop

        return TRMCTTCarry(new_inner_carry, new_current_data, new_steps, halted), outputs


def _group_id(course_index: torch.Tensor) -> torch.Tensor:
    """batch-row * n_courses + course_index -- disambiguates courses from
    different rows in the same padded batch."""
    B, N = course_index.shape
    b_idx = torch.arange(B, device=course_index.device).unsqueeze(1).expand(B, N)
    n_courses = int(course_index.max().item()) + 1
    return b_idx * n_courses + course_index


def _course_set_accuracy(period_prob: torch.Tensor, period_label: torch.Tensor,
                          node_mask: torch.Tensor, course_index: torch.Tensor) -> float:
    """Fraction of each course's periods predicted correctly as a set (order
    within a course doesn't matter). Vectorized over groups, bucketed by
    group size `k`, instead of a per-group python loop."""
    pred_period = period_prob.argmax(dim=-1)  # [B, N]
    group_id = _group_id(course_index)

    flat_mask = node_mask.reshape(-1).cpu().numpy()
    flat_group = group_id.reshape(-1).cpu().numpy()[flat_mask]
    flat_true = period_label.reshape(-1).cpu().numpy()[flat_mask]
    flat_pred = pred_period.reshape(-1).cpu().numpy()[flat_mask]

    order = np.argsort(flat_group, kind="stable")
    flat_group, flat_true, flat_pred = flat_group[order], flat_true[order], flat_pred[order]
    _, start_idx = np.unique(flat_group, return_index=True)
    boundaries = np.concatenate([start_idx, [len(flat_group)]])
    sizes = np.diff(boundaries)
    starts = boundaries[:-1]

    total = len(flat_group)
    matched = 0

    # k=1 groups: trivial single-element comparison, fully vectorized
    starts_1 = starts[sizes == 1]
    if len(starts_1) > 0:
        matched += int((flat_true[starts_1] == flat_pred[starts_1]).sum())

    for k in np.unique(sizes[sizes > 1]):
        k = int(k)
        starts_k = starts[sizes == k]
        offsets = starts_k[:, None] + np.arange(k)[None, :]  # [G_k, k]
        true_k = flat_true[offsets]  # [G_k, k]
        pred_k = flat_pred[offsets]  # [G_k, k]
        # true_k values are already distinct within a group (H1), so
        # counting "does true[i] appear anywhere in pred[g]" IS |true ∩ pred|
        any_match = (true_k[:, :, None] == pred_k[:, None, :]).any(axis=2)  # [G_k, k]
        matched += int(any_match.sum())

    return matched / total


def _period_fully_matched_per_instance(period_prob: torch.Tensor, period_label: torch.Tensor,
                                        node_mask: torch.Tensor, course_index: torch.Tensor) -> torch.Tensor:
    """Per-instance (batch row) bool: True iff EVERY course in that instance
    has its full true period set present among its predictions -- same
    grouping as `_course_set_accuracy`, but all-or-nothing per course
    instead of counting partial credit."""
    pred_period = period_prob.argmax(dim=-1)  # [B, N]
    group_id = _group_id(course_index)
    B, N = period_label.shape
    b_idx = np.arange(B)[:, None].repeat(N, axis=1).reshape(-1)

    flat_mask = node_mask.reshape(-1).cpu().numpy()
    flat_b = b_idx[flat_mask]
    flat_group = group_id.reshape(-1).cpu().numpy()[flat_mask]
    flat_true = period_label.reshape(-1).cpu().numpy()[flat_mask]
    flat_pred = pred_period.reshape(-1).cpu().numpy()[flat_mask]

    order = np.argsort(flat_group, kind="stable")
    flat_group, flat_b, flat_true, flat_pred = flat_group[order], flat_b[order], flat_true[order], flat_pred[order]
    _, start_idx = np.unique(flat_group, return_index=True)
    boundaries = np.concatenate([start_idx, [len(flat_group)]])
    sizes = np.diff(boundaries)
    starts = boundaries[:-1]
    group_batch = flat_b[starts]  # [G] batch row owning each group

    solved = np.ones(len(sizes), dtype=bool)

    starts_1 = starts[sizes == 1]
    if len(starts_1) > 0:
        solved[sizes == 1] = flat_true[starts_1] == flat_pred[starts_1]

    for k in np.unique(sizes[sizes > 1]):
        k = int(k)
        mask_k = sizes == k
        starts_k = starts[mask_k]
        offsets = starts_k[:, None] + np.arange(k)[None, :]  # [G_k, k]
        true_k = flat_true[offsets]
        pred_k = flat_pred[offsets]
        any_match = (true_k[:, :, None] == pred_k[:, None, :]).any(axis=2)  # [G_k, k]
        solved[mask_k] = any_match.all(axis=1)

    per_instance = np.ones(B, dtype=bool)
    np.logical_and.at(per_instance, group_batch, solved)  # AND across every course in a row
    return torch.from_numpy(per_instance)


_PERM_CACHE: Dict[int, np.ndarray] = {}


def _all_perms(k: int) -> np.ndarray:
    """[k!, k] array of every permutation of range(k), cached per k."""
    if k not in _PERM_CACHE:
        _PERM_CACHE[k] = np.array(list(itertools.permutations(range(k))))
    return _PERM_CACHE[k]


# groups larger than this fall back to a per-group scipy call (k! blows up)
_BRUTE_FORCE_MAX_K = 8


def _match_period_targets(period_prob: torch.Tensor, period_label: torch.Tensor,
                           node_mask: torch.Tensor, course_index: torch.Tensor) -> torch.Tensor:
    """DETR-style Hungarian matching: lectures within a course are
    interchangeable, so the target period for each is whichever pairing to
    the course's true periods is currently cheapest under `period_prob`.
    Returns a [B, N] LongTensor of matched target vocab indices.

    Vectorized over course-groups bucketed by size `k` (brute-force
    permutation search per bucket) instead of a per-group scipy call."""
    device = period_label.device
    detached_prob_np = period_prob.detach().cpu().numpy()  # [B, N, V], ONE sync
    period_label_np = period_label.cpu().numpy()
    matched_target_np = period_label_np.copy()
    group_id_np = _group_id(course_index).cpu().numpy()

    B, N = period_label_np.shape
    b_idx = np.arange(B)[:, None].repeat(N, axis=1).reshape(-1)
    n_idx = np.arange(N)[None, :].repeat(B, axis=0).reshape(-1)
    flat_mask = node_mask.reshape(-1).cpu().numpy()
    flat_group = group_id_np.reshape(-1)[flat_mask]
    flat_b = b_idx[flat_mask]
    flat_n = n_idx[flat_mask]
    flat_true = period_label_np.reshape(-1)[flat_mask]

    order = np.argsort(flat_group, kind="stable")
    flat_group, flat_b, flat_n, flat_true = flat_group[order], flat_b[order], flat_n[order], flat_true[order]
    _, start_idx = np.unique(flat_group, return_index=True)
    boundaries = np.concatenate([start_idx, [len(flat_group)]])
    sizes = np.diff(boundaries)   # [G] size of each group, vectorized (no python loop)
    starts = boundaries[:-1]      # [G] start offset of each group

    # bucket course-groups by size k (excluding k=1, which needs no matching)
    # entirely via boolean masks -- no per-group python loop anywhere below
    for k in np.unique(sizes[sizes > 1]):
        k = int(k)
        starts_k = starts[sizes == k]  # [G_k]
        offsets = starts_k[:, None] + np.arange(k)[None, :]  # [G_k, k]
        b0_arr = flat_b[starts_k]            # [G_k]
        ns_arr = flat_n[offsets]             # [G_k, k]
        true_periods_arr = flat_true[offsets]  # [G_k, k]

        if k > _BRUTE_FORCE_MAX_K:
            # extremely rare fallback: per-group scipy call
            for g in range(len(starts_k)):
                probs = detached_prob_np[b0_arr[g]][ns_arr[g][:, None], true_periods_arr[g][None, :]]
                cost = -np.log(np.clip(probs, 1e-8, None))
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_target_np[b0_arr[g], ns_arr[g][row_ind]] = true_periods_arr[g][col_ind]
            continue

        # probs_all[g, i, j] = P(lecture i of group g gets true period j of group g)
        probs_all = detached_prob_np[b0_arr[:, None, None], ns_arr[:, :, None], true_periods_arr[:, None, :]]
        cost_all = -np.log(np.clip(probs_all, 1e-8, None))  # [G, k, k]

        perms = _all_perms(k)  # [P, k]
        # perm_costs[g, p] = sum_i cost_all[g, i, perms[p, i]]
        perm_costs = cost_all[:, np.arange(k)[None, :], perms].sum(axis=-1)  # [G, P]
        best_perm_idx = perm_costs.argmin(axis=-1)  # [G]
        chosen_cols = perms[best_perm_idx]  # [G, k]
        chosen_periods = np.take_along_axis(true_periods_arr, chosen_cols, axis=1)  # [G, k]

        matched_target_np[b0_arr[:, None], ns_arr] = chosen_periods

    return torch.from_numpy(matched_target_np).to(device)


def _collision_loss(period_prob: torch.Tensor, room_prob: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Soft H2 (room occupancy) penalty: period and room are decoded
    per-lecture independently, so this penalizes the expected probability
    mass any two lectures place on colliding on the same (period, room).
    Computed via two [N,N] Gram matrices instead of a pairwise loop;
    checkpointed since those intermediates can be large."""
    def _compute(period_prob, room_prob, mask):
        period_masked = period_prob * mask
        room_masked = room_prob * mask

        period_overlap = torch.bmm(period_masked, period_masked.transpose(-2, -1))  # [B, N, N]
        room_overlap = torch.bmm(room_masked, room_masked.transpose(-2, -1))        # [B, N, N]
        collision = period_overlap * room_overlap

        N = collision.shape[-1]
        eye = torch.eye(N, device=collision.device, dtype=torch.bool)
        collision = collision.masked_fill(eye, 0.0)  # exclude i==j
        return collision.sum() / 2  # each unordered pair counted twice

    mask = node_mask.to(period_prob.dtype).unsqueeze(-1)  # [B, N, 1]
    total = checkpoint(_compute, period_prob, room_prob, mask, use_reentrant=False)
    n_real = node_mask.sum(dim=1).to(period_prob.dtype)  # [B]
    n_pairs = (n_real * (n_real - 1) / 2).sum().clamp_min(1.0)
    return total / n_pairs


class TRMCTTLossHead(nn.Module):
    def __init__(self, model: nn.Module, collision_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.collision_weight = collision_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, carry: TRMCTTCarry, batch: Dict[str, torch.Tensor]):
        new_carry, outputs = self.model(carry, batch)
        period_prob, room_prob = outputs["period_prob"], outputs["room_prob"]

        data = new_carry.current_data  # post swap-on-halt, not the raw candidate batch
        node_mask = data["node_mask"]
        course_index = data["course_index"]
        period_label = data["period_label"]
        room_label = data["room_label"]

        with torch.no_grad():
            matched_target_period = _match_period_targets(period_prob, period_label, node_mask, course_index)

        target_period_prob = period_prob.gather(-1, matched_target_period.unsqueeze(-1)).squeeze(-1)
        target_room_prob = room_prob.gather(-1, room_label.unsqueeze(-1)).squeeze(-1)

        eps = 1e-8
        period_loss = -torch.log(target_period_prob[node_mask].clamp_min(eps)).mean()
        room_loss = -torch.log(target_room_prob[node_mask].clamp_min(eps)).mean()
        collision_loss = _collision_loss(period_prob, room_prob, node_mask)
        loss = period_loss + room_loss + self.collision_weight * collision_loss

        with torch.no_grad():
            pred_room = room_prob.argmax(dim=-1)
            room_correct = (pred_room == room_label)
            period_set_accuracy = _course_set_accuracy(period_prob, period_label, node_mask, course_index)

            # exact_accuracy: fraction of INSTANCES fully solved -- every
            # course's period set correct AND every room correct, all at once
            period_fully_correct = _period_fully_matched_per_instance(
                period_prob, period_label, node_mask, course_index
            ).to(room_correct.device)
            room_fully_correct = (room_correct | ~node_mask).all(dim=1)
            exact_accuracy = (period_fully_correct & room_fully_correct).float().mean()

            metrics = dict(
                room_accuracy=room_correct[node_mask].float().mean(),
                exact_accuracy=exact_accuracy,
                period_set_accuracy=torch.tensor(period_set_accuracy),
                period_loss=period_loss.detach(),
                room_loss=room_loss.detach(),
                collision_loss=collision_loss.detach(),
                count=node_mask.sum().float(),
            )
        return new_carry, loss, metrics, outputs, new_carry.halted.all()
