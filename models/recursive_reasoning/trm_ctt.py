from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import Attention, Linear, SwiGLU, rms_norm
from dataset.ctt_dataset import TAG_DIM
from dataset.ctt_encode import MAX_COURSES, MAX_DAYS, MAX_LECTURES, MAX_PERIODS_PER_DAY

MASK_VALUE = -1e9
RELATION_SCALE = (1.0, 2.0, 1.0)  # course, curriculum, teacher


@dataclass
class TRMCTTConfig:
    """Model sizes, recursion depth and options."""
    lecture_dim: int
    room_dim: int
    period_vocab: int
    hidden_size: int = 256
    expansion: float = 4.0
    num_heads: int = 4
    head_dim: int = 64
    l_layers: int = 2
    h_cycles: int = 2
    l_cycles: int = 2
    sup_steps: int = 16
    logit_scale: float = 4.0
    course_embed: bool = True
    zero_init_residual: bool = False
    feedback: bool = False
    feedback_counts: bool = True


class Block(nn.Module):
    """Lecture self/cross attention and slot cross attention, each with an MLP."""
    def __init__(self, config: TRMCTTConfig):
        """Three attentions and two MLPs."""
        super().__init__()
        attn = lambda: Attention(config.hidden_size, config.num_heads, config.head_dim)
        self.lecture_self, self.lecture_cross, self.slot_cross = attn(), attn(), attn()
        self.lecture_mlp = SwiGLU(config.hidden_size, config.expansion)
        self.slot_mlp = SwiGLU(config.hidden_size, config.expansion)
        if config.zero_init_residual:
            for m in (self.lecture_self, self.lecture_cross, self.slot_cross):
                nn.init.zeros_(m.out.weight)
            for m in (self.lecture_mlp, self.slot_mlp):
                nn.init.zeros_(m.down.weight)

    def forward(self, zl, zs, masks):
        """Lectures attend to lectures and slots, then slots attend to lectures."""
        self_mask, to_slot, to_lecture = masks
        zl = rms_norm(zl + self.lecture_self(zl, zl, self_mask))
        zl = rms_norm(zl + self.lecture_cross(zl, zs, to_slot))
        zl = rms_norm(zl + self.lecture_mlp(zl))
        zs = rms_norm(zs + self.slot_cross(zs, zl, to_lecture))
        zs = rms_norm(zs + self.slot_mlp(zs))
        return zl, zs


class TRMCTT(nn.Module):
    """Lecture tokens and (period, room) slot tokens; each lecture points to one slot."""

    def __init__(self, config: TRMCTTConfig):
        """Input projections, embeddings, shared layers, decoder and feedback projections."""
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.lecture_proj = Linear(config.lecture_dim, h)
        self.tag_proj = Linear(TAG_DIM, h)
        self.position_embed = nn.Parameter(trunc_normal_init_(torch.empty(MAX_LECTURES, h)))
        if config.course_embed:
            self.course_embed = nn.Parameter(trunc_normal_init_(torch.empty(MAX_COURSES, h)))
        self.room_proj = Linear(config.room_dim, h)
        self.period_embed = nn.Parameter(trunc_normal_init_(torch.empty(config.period_vocab, h)))
        self.relation_proj = Linear(len(RELATION_SCALE), config.num_heads)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.l_layers))
        self.query, self.key = Linear(h, h), Linear(h, h)
        self.zl_init = nn.Buffer(trunc_normal_init_(torch.empty(h)))
        self.zs_init = nn.Buffer(trunc_normal_init_(torch.empty(h)))
        if config.feedback:
            self.fb_slot = Linear(h, h)
            self.fb_lecture = Linear(1 + len(RELATION_SCALE) + 1, h)
            self.fb_occupancy = Linear(2, h)
            for m in (self.fb_slot, self.fb_lecture, self.fb_occupancy):
                nn.init.zeros_(m.weight)

    def init_z(self, batch: Dict[str, torch.Tensor]) -> Tuple:
        """Initial lecture and slot states, no previous assignment."""
        B, N = batch["node_mask"].shape
        S = batch["period_mask"].shape[-1] * batch["room_mask"].shape[-1]
        return self.zl_init.expand(B, N, -1).clone(), self.zs_init.expand(B, S, -1).clone(), None

    def _feedback(self, slot, xs, batch):
        """Previous assignment: slot embedding and clash counts per lecture, occupancy per slot."""
        R = batch["room_mask"].shape[1]
        if slot is None:
            return 0, 0
        fl = self.fb_slot(xs.gather(1, slot[..., None].expand(-1, -1, xs.shape[-1])))
        if not self.config.feedback_counts:
            return fl, 0
        node = batch["node_mask"].to(xs.dtype)
        onehot = F.one_hot(slot, xs.shape[1]).to(xs.dtype) * node[..., None]
        occ = onehot.sum(1)
        period = F.one_hot(slot // R, xs.shape[1] // R).to(xs.dtype) * node[..., None]
        clashes = []
        for key in ("course_membership", "curriculum_membership", "teacher_membership"):
            m = batch[key].to(xs.dtype)
            group_occ = torch.einsum("bng,bnp->bgp", m, period)
            clashes.append((torch.einsum("bng,bgp->bnp", m, group_occ) * period).sum(-1) - m.sum(-1))
        clashes.append(occ.gather(1, slot) - 1)
        ones = torch.ones_like(node)
        fl = fl + self.fb_lecture((torch.stack([ones, *clashes], -1) * node[..., None]).clamp_min(0).log1p())
        fs = self.fb_occupancy(torch.stack([torch.ones_like(occ), occ], -1).log1p())
        return fl, fs

    def _masks(self, batch: Dict[str, torch.Tensor]):
        """Relation-biased lecture mask, slot/lecture masks, and valid-slot mask for decoding."""
        node_mask, room_mask, period_mask = batch["node_mask"], batch["room_mask"], batch["period_mask"]
        groups = (batch["course_membership"], batch["curriculum_membership"], batch["teacher_membership"])
        rel = torch.stack([m.float() @ m.float().transpose(1, 2) / s for m, s in zip(groups, RELATION_SCALE)], -1)
        self_mask = self.relation_proj(rel).permute(0, 3, 1, 2).masked_fill(~node_mask[:, None, None, :], MASK_VALUE)

        slot_mask = (period_mask.any(1)[:, :, None] & room_mask[:, None, :]).flatten(1)
        decode_mask = (period_mask[..., None] & room_mask[:, None, None, :]).flatten(2)
        return (self_mask, slot_mask[:, None, None, :], node_mask[:, None, None, :]), decode_mask

    def _recurse(self, zl, zs, xl, xs, masks):
        """l_cycles passes over the layers, re-injecting the input each time."""
        for _ in range(self.config.l_cycles):
            zl, zs = zl + xl, zs + xs
            for layer in self.layers:
                zl, zs = layer(zl, zs, masks)
        return zl, zs

    def forward(self, z, batch: Dict[str, torch.Tensor]):
        """One supervision step: h_cycles recursions, then lecture-slot logits."""
        masks, decode_mask = self._masks(batch)
        xl = (self.lecture_proj(batch["lecture_features"]) + self.tag_proj(batch["lecture_tag"])
              + self.position_embed[batch["lecture_index"]])
        if self.config.course_embed:
            xl = xl + self.course_embed[batch["course_membership"].float().argmax(-1)]
        xs = (self.period_embed[None, :, None, :] + self.room_proj(batch["room_features"])[:, None, :, :]).flatten(1, 2)

        zl, zs, slot = z
        if self.config.feedback:
            fl, fs = self._feedback(slot, xs, batch)
            xl, xs = xl + fl, xs + fs
        with torch.no_grad():
            for _ in range(self.config.h_cycles - 1):
                zl, zs = self._recurse(zl, zs, xl, xs, masks)
        zl, zs = self._recurse(zl, zs, xl, xs, masks)

        logits = self.query(zl) @ self.key(zs).transpose(1, 2) * self.config.logit_scale * self.config.hidden_size ** -0.5
        logits = logits.masked_fill(~decode_mask, MASK_VALUE)
        return (zl.detach(), zs.detach(), logits.argmax(-1).detach()), logits

    def assign(self, logits: torch.Tensor) -> torch.Tensor:
        """One-hot argmax slot per lecture, straight-through softmax gradient."""
        soft = logits.softmax(-1)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        return hard - soft.detach() + soft


class _Occupancy(torch.autograd.Function):
    """1[group g uses slot k] for one-hot y; gradient is the exact single-lecture move."""

    @staticmethod
    def forward(ctx, y, membership):
        """Distinct-use indicator per group and slot."""
        hard = (y > 0.5).to(y.dtype)
        count = torch.einsum("bng,bnk->bgk", membership, hard)
        ctx.save_for_backward(hard, membership, count)
        return (count > 0).to(y.dtype)

    @staticmethod
    def backward(ctx, grad):
        """Gradient of moving one lecture into or out of its group's slot."""
        hard, membership, count = ctx.saved_tensors
        alone = torch.einsum("bng,bgk->bnk", membership, grad * (count == 1))
        empty = torch.einsum("bng,bgk->bnk", membership, grad * (count == 0))
        return torch.where(hard.bool(), alone, empty), None


def _occupancy(y: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Group-uses-slot indicator."""
    return _Occupancy.apply(y, membership.to(y.dtype))


def _excess(y: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    """Group size minus distinct slots used."""
    count = torch.einsum("bng,bnk->bgk", membership.to(y.dtype), y)
    return (count - _occupancy(y, membership)).sum()


def constraint_loss(assign: torch.Tensor, batch: Dict[str, torch.Tensor], weights: Dict[str, float]):
    """H1-H3/S1-S4 violations per lecture of a one-hot assignment; H4 is masked."""
    node_mask = batch["node_mask"]
    slot_prob = assign * node_mask.unsqueeze(-1).to(assign.dtype)
    B, N, _ = slot_prob.shape
    joint = slot_prob.view(B, N, MAX_DAYS * MAX_PERIODS_PER_DAY, -1)
    period_prob, room_prob = joint.sum(-1), joint.sum(-2)
    course, curriculum = batch["course_membership"], batch["curriculum_membership"]
    everyone = node_mask.unsqueeze(-1)

    terms = dict(
        h1=_excess(period_prob, course),
        h2=_excess(slot_prob, everyone),
        h3a=_excess(period_prob, curriculum),
        h3b=_excess(period_prob, batch["teacher_membership"]),
    )

    over = (batch["n_students"].unsqueeze(-1) - batch["room_capacity"].unsqueeze(1)).clamp_min(0)
    terms["s1"] = (room_prob * over).sum()

    day_prob = period_prob.view(B, N, MAX_DAYS, MAX_PERIODS_PER_DAY).sum(-1)
    days_used = _occupancy(day_prob, course).sum(-1)
    terms["s2"] = (batch["min_working_days"] - days_used).clamp_min(0).sum()

    occ = _occupancy(period_prob, curriculum).view(B, -1, MAX_DAYS, MAX_PERIODS_PER_DAY)
    before, after = F.pad(occ[..., :-1], (1, 0)), F.pad(occ[..., 1:], (0, 1))
    terms["s3"] = (occ * (1 - before) * (1 - after)).sum()

    rooms_used = _occupancy(room_prob, course).sum(-1)
    terms["s4"] = (rooms_used - 1).clamp_min(0).sum()

    terms = {k: v / node_mask.sum() for k, v in terms.items()}
    loss = sum(weights[k] * terms[k] for k in weights)
    return loss, terms
