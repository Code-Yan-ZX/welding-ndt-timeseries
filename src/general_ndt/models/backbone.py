"""共享骨干: 轻量 Patch Transformer (与仓库 UltrasoundMAE 同构但简化)。

Phase 2A 修复:
1. **valid_mask 真正生效**: token 级 valid mask (B, N) 1=有效 → Transformer 的
   src_key_padding_mask; **CLS 位置恒 valid**; padded token 不参与 attention。
2. **网格位置编码**: 1d (C, n_col) 通道+时间位置; 2d (C, n_h, n_w) 通道+行+列位置;
   支持可变长度 (正弦时间/空间维)。位置由 PatchTransformer 内部按 grid 添加。

第一版只实现这一个 backbone (方法规格 §3.4)。Mamba / 1D-2D 混合列为后续扩展。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from general_ndt.models.positions import GridPositionalEncoding


class PatchTransformer(nn.Module):
    """可变形长 token 序列的轻量 Transformer encoder。

    输入: (B, N, d), valid_mask (B, N) 1=有效, grid (n_row, n_col)/(C, n_h, n_w)
    输出: (B, N+1, d) + cls 池化表征 (B, d)。
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_rows: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc = GridPositionalEncoding(d_model=d_model, max_rows=max_rows)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=int(d_model * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def _key_padding_mask(
        self, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, N) 1=有效 → (B, N+1) True=忽略 (padded); CLS 位置恒 False (valid)。"""
        pad = ~valid_mask.bool()                     # (B, N) True=padding
        cls_valid = torch.zeros(
            pad.shape[0], 1, dtype=pad.dtype, device=pad.device
        )
        return torch.cat([cls_valid, pad], dim=1)    # (B, N+1)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        grid: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        # x: (B, N, d); valid_mask: (B, N) 1=有效 (padding 位置=0); grid: 网格维度
        B, N, _ = x.shape
        if grid is not None:
            pe = self.pos_enc(grid, x.device).reshape(1, N, self.d_model)
            x = x + pe
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)               # (B, N+1, d)
        # 全有效 (无 padding) 时不传 src_key_padding_mask → 与不传 mask 完全一致
        # (PyTorch 在提供全 False mask 时走不同 kernel, 数值不等价)
        src_mask = None
        if valid_mask is not None:
            sm = self._key_padding_mask(valid_mask)
            if bool(sm.any()):
                src_mask = sm
        for blk in self.blocks:
            h = blk(h, src_key_padding_mask=src_mask)
        h = self.norm(h)
        return h                                      # (B, N+1, d)

    def pooled(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        grid: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """CLS 池化表征 (B, d)。"""
        return self.forward(x, valid_mask, grid)[:, 0]

    def forward_token_features(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        grid: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        return self.forward(x, valid_mask, grid)[:, 1:]  # 去掉 CLS, 返回 (B, N, d)
