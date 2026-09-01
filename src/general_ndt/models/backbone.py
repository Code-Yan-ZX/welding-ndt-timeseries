"""共享骨干: 轻量 Patch Transformer (与仓库 UltrasoundMAE 同构但简化)。

第一版只实现这一个 backbone (方法规格 §3.4)。Mamba / 1D-2D 混合列为后续扩展。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchTransformer(nn.Module):
    """可变形长 token 序列的轻量 Transformer encoder。

    输入: (B, N, d); 输出: (B, N, d) + cls 池化表征 (B, d)。
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, d); valid_mask: (B, N) 0/1 (padding 位置)
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)  # (B, N+1, d)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h  # (B, N+1, d)

    def pooled(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """CLS 池化表征 (B, d); valid_mask 预留 (padding 位置在 mask 时已被置 0)。"""
        h = self.forward(x, valid_mask)
        return h[:, 0]

    def forward_token_features(self, x: torch.Tensor, valid_mask=None):
        return self.forward(x, valid_mask)[:, 1:]  # 去掉 CLS, 返回 (B, N, d)
