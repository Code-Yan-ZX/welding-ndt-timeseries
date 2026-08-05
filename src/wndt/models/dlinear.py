"""DLinear (Zeng et al., AAAI 2023) adapted to classification.

Series decomposition (moving-average kernel) into trend + seasonal, one linear
per channel for each component, then a classification head over the flattened
output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, L)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        x_pad = torch.cat([front, x, end], dim=2)
        trend = self.avg(x_pad)
        # AvgPool1d output length = L + 2*pad - kernel + 1 = L (kernel odd)
        trend = trend[:, :, : x.shape[2]]
        seasonal = x - trend
        return seasonal, trend


class DLinearClassifier(nn.Module):
    def __init__(self, seq_len: int = 200, n_vars: int = 2, kernel_size: int = 25,
                 n_classes: int = 2):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.linear_seasonal = nn.ModuleList(
            [nn.Linear(seq_len, seq_len) for _ in range(n_vars)])
        self.linear_trend = nn.ModuleList(
            [nn.Linear(seq_len, seq_len) for _ in range(n_vars)])
        self.head = nn.Linear(n_vars * seq_len, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        seasonal, trend = self.decomp(x)
        outs = []
        for i in range(self.n_vars):
            s = self.linear_seasonal[i](seasonal[:, i, :])
            t = self.linear_trend[i](trend[:, i, :])
            outs.append(s + t)
        out = torch.stack(outs, dim=1).reshape(x.shape[0], -1)
        return self.head(out)
