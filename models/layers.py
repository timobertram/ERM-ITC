import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_


class Linear(nn.Module):
    """Bias-free linear with truncated normal init."""
    def __init__(self, in_features: int, out_features: int):
        """Weight only, std 1/sqrt(in_features)."""
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=in_features ** -0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x @ W^T."""
        return F.linear(x, self.weight)


class Attention(nn.Module):
    """Multi-head attention of x over context with an additive or boolean mask."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        """Separate query and joint key/value projections."""
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q = Linear(hidden_size, num_heads * head_dim)
        self.kv = Linear(hidden_size, 2 * num_heads * head_dim)
        self.out = Linear(num_heads * head_dim, hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Queries from x, keys/values from context."""
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).view(B, context.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).flatten(2))


class SwiGLU(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size: int, expansion: float):
        """Hidden width 2/3 * expansion * hidden_size, rounded up to a multiple of 256."""
        super().__init__()
        inter = -(round(expansion * hidden_size * 2 / 3) // -256) * 256
        self.gate_up = Linear(hidden_size, 2 * inter)
        self.down = Linear(inter, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """down(silu(gate) * up)."""
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS norm without learned scale."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
