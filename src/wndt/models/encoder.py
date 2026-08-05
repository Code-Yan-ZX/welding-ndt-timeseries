"""Time-series encoder adapted from the official ITFormer TimeSeriesEncoder
(Pandalin98/ITFormer-ICML25), specialized for single-cycle welding signals.

Input  : x [B, seq_len, n_vars]   (seq_len=200, n_vars=2 -> (V, I))
Output : z [B, n_vars, n_patches, d_model]   (e.g. [B, 2, 10, 512])

Patch layout mirrors the paper: patch_len == stride == 20 -> 10 patches per
channel (the paper used 600 pts / patch 60 -> 10 patches per channel).
Positional coding: sinusoidal time-position along the patch axis + learnable
per-channel position. (The official cross-cycle / rotary branch is removed:
this task is single-cycle.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimePosition(nn.Module):
    """Sinusoidal position encoding, shape (max_len, d_model)."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        assert d_model % 2 == 0, "SinusoidalTimePosition needs even d_model"
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, n: int) -> torch.Tensor:
        return self.pe[:n]


class SeqAttention(nn.Module):
    """Self-attention over the patch axis, independently per variable."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, P, D]
        b, v, p, d = x.shape
        qkv = self.qkv(x).reshape(b * v, p, 3, self.n_heads, d // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*V, H, P, Dh]
        q, k, val = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, val, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b * v, p, d)
        return self.proj(out).view(b, v, p, d)


class VarAttention(nn.Module):
    """Self-attention over the variable (channel) axis, independently per patch."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, P, D]
        b, v, p, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * p, v, d)
        qkv = self.qkv(x).reshape(b * p, v, 3, self.n_heads, d // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, val = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, val, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b * p, v, d)
        out = self.proj(out).view(b, p, v, d).permute(0, 2, 1, 3)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.seq_attn = SeqAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.var_attn = VarAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.seq_attn(self.norm1(x))
        x = x + self.var_attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x


class WeldTSEncoder(nn.Module):
    def __init__(self, seq_len: int = 200, n_vars: int = 2, patch_len: int = 20,
                 stride: int = 20, d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        assert patch_len == stride, "official encoder requires patch_len == stride"
        assert seq_len % patch_len == 0
        self.n_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model, bias=False), nn.Dropout(dropout))
        self.time_pos = SinusoidalTimePosition(d_model, max_len=max(64, self.n_patches))
        self.var_pos = nn.Parameter(torch.randn(1, n_vars, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, dropout) for _ in range(e_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, n_vars] -> [B, n_vars, n_patches, d_model]"""
        x = x.transpose(1, 2)                                  # [B, V, L]
        x = x.unfold(-1, self.patch_len, self.patch_len)       # [B, V, P, patch_len]
        x = self.patch_embedding(x)                            # [B, V, P, D]
        x = x + self.time_pos(self.n_patches)[None, None, :, :]
        x = x + self.var_pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def memory(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output rearranged as fusion memory [B, P, V, D]."""
        z = self.forward(x)
        return z.permute(0, 2, 1, 3).contiguous()
