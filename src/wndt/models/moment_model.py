"""MOMENT time-series foundation model wrapper for SAW defect classification.

Loads the pretrained MOMENT-1 encoder (frozen by default) as a feature
extractor. The 4 SAW channels (current_a/b, voltage_a/b) are each run through
the shared pretrained encoder; patch embeddings are pooled per channel and fed
to a lightweight linear head. This is the "time-series foundation model"
experiment: zero training of the backbone, only a probe head is learned.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentClassifier(nn.Module):
    def __init__(self, *, n_classes: int = 2, n_channels: int = 4, seq_len: int = 512,
                 ckpt: str = "AutonLab/MOMENT-1-large", freeze: bool = True,
                 dropout: float = 0.1, moment_input_len: int = 512,
                 use_bf16: bool = True):
        super().__init__()
        from momentfm import MOMENTPipeline
        self.pipe = MOMENTPipeline.from_pretrained(
            ckpt, model_kwargs={"task_name": "embedding"})
        self.pipe.init()
        self.backbone = self.pipe  # MOMENTPipeline is itself the nn.Module
        self.n_channels = n_channels
        self.moment_input_len = moment_input_len
        self.freeze = freeze
        self.use_bf16 = use_bf16
        self.d_model = int(self.backbone.config.d_model)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(n_channels * self.d_model),
            nn.Dropout(dropout),
            nn.Linear(n_channels * self.d_model, n_classes),
        )

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, C, moment_input_len)
        if x.shape[-1] != self.moment_input_len:
            x = F.interpolate(x, size=self.moment_input_len, mode="linear",
                              align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._adapt(x.float())                       # (B, C, Lm)
        B, C, L = x.shape
        input_mask = torch.ones(B, L, device=x.device)
        # frozen backbone: no graph + bf16 autocast for speed
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=self.use_bf16):
                out = self.backbone.embed(x_enc=x, input_mask=input_mask,
                                          reduction="none")
        emb = out.embeddings.float()                      # (B, C, n_patches, d)
        emb = emb.mean(dim=2)                             # (B, C, d)  pool over patches
        emb = emb.flatten(1)                              # (B, C*d)
        return self.head(emb)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()                         # deterministic frozen features
        return self
