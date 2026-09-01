"""Stem 层: 把原始信号切成 patch token (每模态轻量, 骨干共享)。

- Stem1D: (B, C, L) 1D 形态 → (B, n_row*C, d) 通道-时间 token 网格 (n_row=C 行, 每行 n_col 个时间 token)
          展平后 (B, C*n_col, d)。注: 保留 token 网格形状便于物理掩码 (ssl/masking)。
- Stem2D: (B, H, W) 2D 形态 → (B, n_h*n_w, d) patch token 网格。
- StemTF: 时频视图 (B, F, T_f) → 同 Stem2D (F 行 × T_f 列 patch 网格)。

掩码控制器在 token 网格 (n_row, n_col) 上工作 (见 general_ndt.ssl.masking)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Stem1D(nn.Module):
    """通道-时间 1D stem: 每通道沿时间切 τ 长 patch → (B, C, n_col, d)。"""

    def __init__(self, in_channels: int, patch_len: int, d_model: int):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=patch_len, stride=patch_len)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, L) → token 网格: 每通道沿时间切 patch, 行=通道, 列=时间 token
        B, C, L = x.shape
        z = self.conv(x)                      # (B, d, n_col)
        z = z.permute(0, 2, 1)                # (B, n_col, d)
        z = self.act(self.norm(z))
        z = z.reshape(B, 1, -1, z.shape[-1])  # (B, 1, n_col, d)
        z = z.expand(B, C, -1, -1).contiguous()  # (B, C, n_col, d) — 每通道 token 独立
        n_col = z.shape[2]
        z = z.reshape(B, C * n_col, -1)       # (B, C*n_col, d) 展平供骨干
        return z, (C, n_col)

    @property
    def n_tokens_per_sample(self) -> int:
        return self.in_channels  # 占位; 实际由 forward 返回网格


class Stem2D(nn.Module):
    """2D patch stem: (B, H, W) → (B, n_h*n_w, d); 返回网格 (n_h, n_w)。"""

    def __init__(self, patch: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch = patch
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, 1, H, W) 或 (B, C, H, W) — 多通道时逐通道 patch 后 concat 到行
        B, C, H, W = x.shape
        z = self.conv(x)                  # (B, d, n_h, n_w)
        z = z.permute(0, 2, 3, 1)         # (B, n_h, n_w, d)
        z = self.act(self.norm(z))
        n_h, n_w = z.shape[1], z.shape[2]
        z = z.reshape(B, n_h * n_w, z.shape[-1])
        return z, (n_h, n_w)


class StemTF(nn.Module):
    """时频视图 stem: (B, F, T_f) → 2D patch 网格 (n_f, n_tf)。底层即 Stem2D。"""

    def __init__(self, patch: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.stem = Stem2D(patch=patch, d_model=d_model, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        return self.stem(x)
