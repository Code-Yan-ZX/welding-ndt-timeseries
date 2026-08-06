"""Torch Dataset over memmapped welding signal segments for the SAW dataset.

Generalizes ``WeldCycleDataset``: one sample = one signal segment of shape
(n_channels, seq_len), where both dimensions are inferred from the memmap
(instead of hard-coded (2, 200)). Labels are arbitrary non-negative ints
(binary or multi-class); -1 = unlabeled (excluded upstream).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wndt.data.dataset import make_weighted_sampler  # re-export for convenience

NORM_MODES = ("global", "sample_z", "none")


class SAWSeriesDataset(Dataset):
    """One sample = one signal segment, tensor shape (C, L).

    norm modes:
      global   - per-channel mean/std computed on TRAIN only
      sample_z - per-sample per-channel z-norm
      none     - raw values
    """

    def __init__(self, processed_dir: str | Path, indices: np.ndarray,
                 norm_mode: str = "global"):
        assert norm_mode in NORM_MODES, f"unknown norm mode {norm_mode}"
        processed_dir = Path(processed_dir)
        self.waves = np.load(processed_dir / "waves.npy", mmap_mode="r")
        labels_all = np.load(processed_dir / "meta_label.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels_all[self.indices]).astype(np.int64)
        self.norm_mode = norm_mode
        self.n_channels = int(self.waves.shape[1])
        self.seq_len = int(self.waves.shape[2])
        self.mean = self.std = None
        if norm_mode == "global":
            with open(processed_dir / "norm_stats.json", "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(self.n_channels, 1)
            self.std = np.asarray(stats["std"], dtype=np.float32).reshape(self.n_channels, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x = np.array(self.waves[self.indices[i]], dtype=np.float32)  # (C, L) copy
        if self.norm_mode == "global":
            x = (x - self.mean) / self.std
        elif self.norm_mode == "sample_z":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mu) / sd
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)
