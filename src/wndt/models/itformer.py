"""ITFormer bridge (adapted from Pandalin98/ITFormer-ICML25).

Learns Learnable Instruct Tokens (LIT), prepends them to projected text-query
embeddings, and fuses them with the encoded time series through
Instruct Time Attention (channel-wise then time-wise cross attention).
Returns the first `prefix_num` tokens of shape [B, prefix_num, d_model].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wndt.models.encoder import MLP, SinusoidalTimePosition


class InstructTimeAttention(nn.Module):
    """Two-stage cross attention from prefix tokens into memory [B, L, V, D].

    stage 1 (channel-wise): for each of the L patch positions, prefix tokens
        attend over the V channel embeddings; outputs averaged over L.
    stage 2 (time-wise): for each of the V channels, prefix tokens attend over
        the L patch positions; outputs averaged over V.
    Final = proj(out_channel + out_time).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # stage 1 (channel-wise) projections
        self.q_c = nn.Linear(d_model, d_model, bias=False)
        self.kv_c = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj_c = nn.Linear(d_model, d_model)
        # stage 2 (time-wise) projections
        self.q_t = nn.Linear(d_model, d_model, bias=False)
        self.kv_t = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj_t = nn.Linear(d_model, d_model)
        self.attn_drop = dropout

    def _attend(self, q: torch.Tensor, kv: torch.Tensor,
                q_proj: nn.Linear, kv_proj: nn.Linear, out_proj: nn.Linear,
                groups: int) -> torch.Tensor:
        """q: [B, N, D]; kv: [B*groups, S, D] -> out: [B, N, D] (mean over groups)."""
        b, n, d = q.shape
        h, dh = self.n_heads, self.head_dim
        q_ = q_proj(q)                                   # [B, N, D]
        q_ = q_.unsqueeze(1).expand(b, groups, n, d).reshape(b * groups, n, d)
        q_ = q_.view(b * groups, n, h, dh).transpose(1, 2)
        k, v = kv_proj(kv).chunk(2, dim=-1)              # [B*groups, S, D]
        s = k.shape[1]
        k = k.view(b * groups, s, h, dh).transpose(1, 2)
        v = v.view(b * groups, s, h, dh).transpose(1, 2)
        drop = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q_, k, v, dropout_p=drop)
        out = out.transpose(1, 2).reshape(b * groups, n, d)
        out = out_proj(out)
        return out.view(b, groups, n, d).mean(dim=1)

    def forward(self, prefix: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """prefix [B, N, D]; memory [B, L, V, D] -> [B, N, D]"""
        b, l, v, d = memory.shape
        kv_chan = memory.reshape(b * l, v, d)                        # stage1 kv
        kv_time = memory.permute(0, 2, 1, 3).reshape(b * v, l, d)    # stage2 kv
        out_c = self._attend(prefix, kv_chan, self.q_c, self.kv_c, self.proj_c, groups=l)
        out_t = self._attend(prefix, kv_time, self.q_t, self.kv_t, self.proj_t, groups=v)
        return out_c + out_t


class ITAttBlock(nn.Module):
    """Pre-norm residual wrapper around InstructTimeAttention (official pattern:
    x = x_in + norm(attn_out))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = InstructTimeAttention(d_model, n_heads, dropout)

    def forward(self, prefix: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.attn(prefix, memory))


class ITFormerBlock(nn.Module):
    """DecoderBasicBlock: self-attn + FFN over the whole (prefix+text) sequence,
    then IT-fusion + FFN on the prefix tokens only."""

    def __init__(self, d_model: int, n_heads: int, prefix_num: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.prefix_num = prefix_num
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ffn_instr = MLP(d_model, mlp_ratio, dropout)
        self.it_block = ITAttBlock(d_model, n_heads, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ffn_prefix = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm_sa(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn_instr(self.norm_ff1(x))
        prefix = x[:, :self.prefix_num]
        rest = x[:, self.prefix_num:]
        prefix = prefix + self.it_block(prefix, memory)
        prefix = prefix + self.ffn_prefix(self.norm_ff2(prefix))
        return torch.cat([prefix, rest], dim=1)


class ITFormer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 2,
                 prefix_num: int = 25, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 max_text_len: int = 128):
        super().__init__()
        self.prefix_num = prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, prefix_num, d_model) * 0.02)
        self.instruc_pos = SinusoidalTimePosition(d_model, max_len=max_text_len + prefix_num)
        self.blocks = nn.ModuleList(
            ITFormerBlock(d_model, n_heads, prefix_num, mlp_ratio, dropout)
            for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_text: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """x_text: [B, T, D] projected text-query embeddings (may be T=0);
        memory: [B, L, V, D]. Returns fused LIT tokens [B, prefix_num, D]."""
        b = memory.shape[0]
        prefix = self.prefix_token.expand(b, -1, -1)
        x = torch.cat([prefix, x_text], dim=1) if x_text.shape[1] > 0 else prefix
        x = x + self.instruc_pos(x.shape[1])[None, :, :]
        for blk in self.blocks:
            x = blk(x, memory)
        x = self.norm(x)
        return x[:, :self.prefix_num]
