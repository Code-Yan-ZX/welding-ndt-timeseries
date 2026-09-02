"""Stem 层: 把原始信号切成 patch token (每模态轻量, 骨干共享)。

核心原则 (Phase 2A 修复): **不在 patch embedding 阶段混合传感器/通道**。
- Stem1D: (B, C, L) → 每通道独立共享 Conv1d(1, d_model) → (B, C, n_col, d)
          展平 (B, C*n_col, d)。每通道有独立 token; 跨通道交互只在共享 backbone 发生。
- Stem2D: (B, C, H, W) → 每通道独立共享 Conv2d(1, d_model) → (B, C, n_h, n_w, d)
          展平 (B, C*n_h*n_w, d)。grid = (C, n_h, n_w)。
- StemTF: 时频视图 → 复用 Stem2D (单通道时 C=1)。

掩码控制器在 token 网格上工作 (1d: (C, n_col); 2d: (C, n_h, n_w)), 见 general_ndt.ssl.masking。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Stem1D(nn.Module):
    """通道-时间 1D stem: 每通道沿时间切 τ 长 patch → (B, C, n_col, d)。

    实现: reshape (B,C,L) -> (B*C, 1, L) -> 共享 Conv1d(1, d_model, kernel=τ, stride=τ)
    -> (B*C, d, n_col) -> reshape (B, C, n_col, d)。共享权重 → 通道数可变。
    不同输入通道交换/修改时, 对应 token 必然变化 (每通道独立投影, 无跨通道混合)。
    """

    def __init__(self, in_channels: int, patch_len: int, d_model: int):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.conv = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=patch_len)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, L) → (B*C, 1, L) 共享 per-channel patch projection
        B, C, L = x.shape
        z = x.reshape(B * C, 1, L)              # (B*C, 1, L)
        z = self.conv(z)                         # (B*C, d_model, n_col)
        z = z.permute(0, 2, 1)                   # (B*C, n_col, d_model)
        z = self.act(self.norm(z))
        z = z.reshape(B, C, -1, z.shape[-1])     # (B, C, n_col, d_model)
        n_col = z.shape[2]
        z = z.reshape(B, C * n_col, -1)          # (B, C*n_col, d_model) 展平供骨干
        return z, (C, n_col)

    @property
    def n_tokens_per_sample(self) -> int:
        return self.in_channels  # 占位; 实际由 forward 返回网格


class Stem2D(nn.Module):
    """2D patch stem: (B, C, H, W) → 每通道共享 Conv2d(1, d_model) → (B, C, n_h, n_w, d)。

    单通道输入 (B, 1, H, W) 时 grid=(1, n_h, n_w); 多通道 (如 EddyCus native_grid_2d
    的 4 频×I/Q = 8 通道) 时 grid=(C, n_h, n_w), 每通道独立 token。
    """

    def __init__(self, patch: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch = patch
        self.conv = nn.Conv2d(1, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        # x: (B, C, H, W) — 逐通道独立 patch 后展平, 通道身份保留在行块内
        B, C, H, W = x.shape
        z = x.reshape(B * C, 1, H, W)        # (B*C, 1, H, W)
        z = self.conv(z)                     # (B*C, d_model, n_h, n_w)
        z = z.permute(0, 2, 3, 1)            # (B*C, n_h, n_w, d_model)
        z = self.act(self.norm(z))
        n_h, n_w = z.shape[1], z.shape[2]
        z = z.reshape(B, C, n_h * n_w, -1)   # (B, C, n_h*n_w, d_model)
        z = z.reshape(B, C * n_h * n_w, -1)  # (B, C*n_h*n_w, d_model)
        return z, (C, n_h, n_w)


class StemTF(nn.Module):
    """时频视图 stem: (B, F, T_f) → 2D patch 网格 (n_f, n_tf)。底层即 Stem2D。"""

    def __init__(self, patch: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.stem = Stem2D(patch=patch, d_model=d_model, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        return self.stem(x)
