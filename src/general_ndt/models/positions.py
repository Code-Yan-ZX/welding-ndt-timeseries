"""网格位置编码 (可审计): 支持 1d (C, n_col) 与 2d (C, n_h, n_w) token 网格。

设计 (Phase 2A):
- 第 0 维 (1d: 通道/传感器; 2d: 通道): **可学习身份嵌入** (离散通道身份, max_rows 上限,
  超界 clamp)。交换通道 → 嵌入变化 → 模型可感知。
- 其余维 (1d: 时间; 2d: 空间行/列): **正弦嵌入** (连续位置, 支持任意长度, 可外推)。
  交换时间/空间位置 → 嵌入变化。
- 位置逐维相加: 1d: pos(r,c)=row[r]+col[c]; 2d: pos(c,rh,cw)=chan[c]+row[rh]+col[cw]。
- **padding 不变性**: 有效 token 恒锚定在网格左上 (row<C_i, col<n_col_i), 位置只依赖
  样本自身的真实网格索引, 与 batch 内 padding 到的最大网格无关。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GridPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_rows: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_rows = max_rows
        # 通道/传感器身份: 可学习
        self.row_embed = nn.Embedding(max_rows, d_model)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        # 正弦频率 (时间/空间连续维)
        n_freq = max(1, d_model // 2)
        inv = 1.0 / (10000 ** (torch.arange(n_freq, dtype=torch.float32) / d_model))
        self.register_buffer("_inv_freq", inv, persistent=False)

    def _sinusoidal(self, n: int, device: torch.device) -> torch.Tensor:
        """(n, d_model) 正弦嵌入 (支持任意长度)。d_model 为奇数时末位补零。"""
        t = torch.arange(n, device=device, dtype=torch.float32)
        pos = t[:, None] * self._inv_freq[None, :]            # (n, n_freq)
        pe = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # (n, 2*n_freq)
        if pe.shape[-1] < self.d_model:
            pe = torch.cat([pe, torch.zeros(n, self.d_model - pe.shape[-1], device=device)], dim=-1)
        return pe[:n]

    def forward(
        self, grid: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """grid: (n_row, n_col) 1d 或 (C, n_h, n_w) 2d → (n_row, n_col, d) / (C, n_h, n_w, d)。"""
        if len(grid) == 2:
            n_row, n_col = grid
            row = self.row_embed(
                torch.arange(n_row, device=device).clamp(max=self.max_rows - 1)
            )                                        # (n_row, d)
            col = self._sinusoidal(n_col, device)     # (n_col, d)
            return row[:, None, :] + col[None, :, :]  # (n_row, n_col, d)
        elif len(grid) == 3:
            C, n_h, n_w = grid
            chan = self.row_embed(
                torch.arange(C, device=device).clamp(max=self.max_rows - 1)
            )                                        # (C, d)
            row = self._sinusoidal(n_h, device)       # (n_h, d)
            col = self._sinusoidal(n_w, device)       # (n_w, d)
            return (
                chan[:, None, None, :]
                + row[None, :, None, :]
                + col[None, None, :, :]
            )                                        # (C, n_h, n_w, d)
        raise ValueError(f"grid 必须是 2 或 3 元组, 得到 {grid}")
