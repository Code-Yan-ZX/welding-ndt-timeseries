"""Cheap ITFormer variants without an LLM (ablations / sanity checks).

- ITFormerProbe : encoder + ITFormer bridge (text-query length 0, LIT only)
                  + mean-pool over fused tokens + Linear head.
- EncoderOnly   : encoder + mean-pool over patches + Linear head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from wndt.models.encoder import WeldTSEncoder
from wndt.models.itformer import ITFormer


class ITFormerProbe(nn.Module):
    def __init__(self, *, prefix_num: int = 25, d_model: int = 512, n_heads: int = 8,
                 it_layers: int = 2, enc_layers: int = 4, patch_len: int = 20,
                 seq_len: int = 200, n_vars: int = 2, dropout: float = 0.1,
                 n_classes: int = 2):
        super().__init__()
        self.encoder = WeldTSEncoder(seq_len=seq_len, n_vars=n_vars,
                                     patch_len=patch_len, stride=patch_len,
                                     d_model=d_model, n_heads=n_heads,
                                     e_layers=enc_layers, dropout=dropout)
        self.itformer = ITFormer(d_model=d_model, n_heads=n_heads,
                                 n_layers=it_layers, prefix_num=prefix_num,
                                 dropout=dropout, max_text_len=prefix_num + 4)
        self.head = nn.Linear(d_model, n_classes)

    def fused(self, waves: torch.Tensor) -> torch.Tensor:
        """waves (B, 2, 200) -> fused tokens (B, prefix_num, d_model)."""
        memory = self.encoder.memory(waves.transpose(1, 2))
        b = memory.shape[0]
        empty = memory.new_zeros(b, 0, memory.shape[-1])
        return self.itformer(empty, memory)

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        fused = self.fused(waves)                    # (B, P, d)
        return self.head(fused.mean(dim=1))          # (B, n_classes)


class EncoderOnly(nn.Module):
    def __init__(self, *, d_model: int = 512, n_heads: int = 8, enc_layers: int = 4,
                 patch_len: int = 20, seq_len: int = 200, n_vars: int = 2,
                 dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        self.encoder = WeldTSEncoder(seq_len=seq_len, n_vars=n_vars,
                                     patch_len=patch_len, stride=patch_len,
                                     d_model=d_model, n_heads=n_heads,
                                     e_layers=enc_layers, dropout=dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        z = self.encoder(waves.transpose(1, 2))      # (B, V, P, d)
        return self.head(z.mean(dim=(1, 2)))         # (B, n_classes)
