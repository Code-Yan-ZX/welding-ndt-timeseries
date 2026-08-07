"""Spectral-Spatial-Frequency (SSF) classifier for PAUT B-scans.

Inspired by *Alliance* (All-in-One Spectral-Spatial-Frequency Awareness
Foundation Model): a single defect decision is made from three complementary
"awarenesses" of the per-position PAUT B-scan (n_beams x seq_len):

  1. spatial  : a 2-D conv stack over the raw B-scan (beam x time) - captures
                the lateral/time echo geometry of a defect indication.
  2. spectral : a 2-D conv stack over the magnitude FFT along the TIME axis
                (depth frequency content of each A-scan).
  3. frequency: a 2-D conv stack over the magnitude FFT along the BEAM axis
                (lateral spatial-frequency; reveals periodic/structured
                indications across the aperture).

The three branch embeddings are concatenated and classified by a small MLP.
This is a numeric deep model (not an image -> LLM pipeline), so it is
compatible with the project's text-only-LLM constraint: the B-scan is a 2-D
real tensor, never rendered as an image.

Input  : x (B, n_beams, seq_len)         [e.g. (B, 49, 512)]
Output : logits (B, n_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    """Small 2-D conv stack -> global average pool -> (B, d_model)."""

    def __init__(self, in_h: int, in_w: int, channels=(32, 64, 128),
                 d_model: int = 128, dropout: float = 0.2, in_channels: int = 1):
        super().__init__()
        layers = []
        c = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(c, ch, kernel_size=(3, 7), padding=(1, 3)))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.GELU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
            c = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        z = self.net(x)
        z = self.pool(z)
        return self.proj(z)


class SSFClassifier(nn.Module):
    """Spectral-spatial-frequency classifier on a PAUT B-scan.

    ``in_channels`` > 1 enables multi-view input: each channel is a different
    PAUT view (e.g. 90/270 skew, 71°/47° refracted), and all three branches
    convolve over the stacked channels jointly."""

    def __init__(self, *, n_beams: int = 49, seq_len: int = 512,
                 d_model: int = 128, dropout: float = 0.2, n_classes: int = 2,
                 in_channels: int = 1):
        super().__init__()
        self.n_beams = n_beams
        self.seq_len = seq_len
        self.in_channels = in_channels
        h_t = n_beams            # spatial branch input H
        w_t = seq_len            # W
        self.spatial = ConvBranch(h_t, w_t, d_model=d_model, dropout=dropout,
                                  in_channels=in_channels)
        # spectral (FFT along time): magnitude -> (n_beams, seq_len//2 + 1)
        self.spectral = ConvBranch(n_beams, seq_len // 2 + 1, d_model=d_model,
                                   dropout=dropout, in_channels=in_channels)
        # frequency (FFT along beam): magnitude -> (n_beams//2 + 1, seq_len)
        self.frequency = ConvBranch(n_beams // 2 + 1, seq_len, d_model=d_model,
                                    dropout=dropout, in_channels=in_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Dropout(dropout),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    @staticmethod
    def _mag_fft(x: torch.Tensor, dim: int) -> torch.Tensor:
        """Return log-magnitude of rfft along ``dim`` (B, C, H, W) -> (B, C, H', W')."""
        xf = torch.fft.rfft(x, dim=dim, norm="ortho")
        mag = torch.log1p(xf.abs())
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, n_beams, seq_len) or (B, n_beams, seq_len)
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)                                  # (B, 1, Bm, L)
        # 1. spatial: raw B-scan (all channels)
        z_sp = self.spatial(x)
        # 2. spectral: FFT along time (last axis), all channels
        sp_t = self._mag_fft(x, dim=-1)                         # (B, C, Bm, L/2+1)
        z_spec = self.spectral(sp_t)
        # 3. frequency: FFT along beam axis, all channels
        sp_b = self._mag_fft(x, dim=-2)                         # (B, C, Bm/2+1, L)
        z_freq = self.frequency(sp_b)
        z = torch.cat([z_sp, z_spec, z_freq], dim=1)
        return self.head(z)
