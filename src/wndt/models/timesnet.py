"""TimesNet (Wu et al., ICLR 2023) adapted to classification.

Compact faithful version: FFT-based period discovery (top_k), 2D reshaping of
each period, inception-style 2D convolution, amplitude-weighted aggregation;
stacked blocks with residual; global average pooling + linear head.
"""
from __future__ import annotations

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Average of 2D convolutions with increasing odd kernel sizes."""

    def __init__(self, c_in: int, c_out: int, num_kernels: int = 6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_kernels):
            k = 2 * i + 1
            self.convs.append(nn.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(conv(x) for conv in self.convs)
        return out / len(self.convs)


class TimesBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, d_ff: int, top_k: int = 3,
                 num_kernels: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = InceptionBlock(d_model, d_ff, num_kernels)
        self.down = nn.Conv2d(d_ff, d_model, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        b, l, c = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = x_fft.abs().mean(dim=-1)            # (B, L//2+1)
        amplitude[:, 0] = float("-inf")                  # ignore DC
        weights, indices = torch.topk(amplitude, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)              # (B, k)
        periods = (l / indices.float().clamp(min=1)).long().clamp(min=2)

        res = []
        for i in range(self.top_k):
            period = int(periods[0, i])                  # same length for all rows
            n_seg = (l + period - 1) // period
            pad_len = n_seg * period - l
            xi = x
            if pad_len > 0:
                xi = F.pad(xi, (0, 0, 0, pad_len))
            x_2d = xi.reshape(b, n_seg, period, c).permute(0, 3, 1, 2)
            o = self.down(self.act(self.conv(x_2d)))
            o = o.permute(0, 2, 3, 1).reshape(b, n_seg * period, c)[:, :l]
            res.append(weights[:, i].mean() * o)
        return sum(res)


class TimesNetClassifier(nn.Module):
    def __init__(self, seq_len: int = 200, n_vars: int = 2, top_k: int = 3,
                 e_layers: int = 2, d_model: int = 64, d_ff: int = 128,
                 num_kernels: int = 6, n_classes: int = 2):
        super().__init__()
        self.embed = nn.Linear(n_vars, d_model)
        self.blocks = nn.ModuleList(
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels)
            for _ in range(e_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(e_layers))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        x = self.embed(x)
        for blk, norm in zip(self.blocks, self.norms):
            x = norm(x + blk(x))
        return self.head(x.mean(dim=1))
