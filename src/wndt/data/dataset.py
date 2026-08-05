"""Torch Dataset over the memmapped welding cycles."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

NORM_MODES = ("global", "cycle_z", "none")


class WeldCycleDataset(Dataset):
    """One sample = one welding cycle, tensor shape (2, 200) = (V, I).

    norm modes:
      global  - per-channel mean/std computed on TRAIN only (official protocol)
      cycle_z - per-sample per-channel z-norm (ablation)
      none    - raw values (ablation)
    """

    def __init__(self, processed_dir: str | Path, indices: np.ndarray,
                 norm_mode: str = "global"):
        assert norm_mode in NORM_MODES, f"unknown norm mode {norm_mode}"
        processed_dir = Path(processed_dir)
        self.waves = np.load(processed_dir / "waves.npy", mmap_mode="r")
        labels_all = np.load(processed_dir / "meta_label.npy", mmap_mode="r")
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels_all[self.indices]).astype(np.int64)
        assert set(np.unique(self.labels)) <= {0, 1}, "unlabeled rows leaked into split"
        self.norm_mode = norm_mode
        self.mean = self.std = None
        if norm_mode == "global":
            with open(processed_dir / "norm_stats.json", "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(2, 1)
            self.std = np.asarray(stats["std"], dtype=np.float32).reshape(2, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x = np.array(self.waves[self.indices[i]], dtype=np.float32)  # (2,200) copy
        if self.norm_mode == "global":
            x = (x - self.mean) / self.std
        elif self.norm_mode == "cycle_z":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mu) / sd
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Official protocol: weight 1-ratio for class 0, ratio for class 1,
    where ratio = fraction of class-0 samples in train."""
    labels = np.asarray(labels)
    ratio = float(np.mean(labels == 0))
    weights = np.zeros(len(labels), dtype=np.float64)
    weights[labels == 0] = 1.0 - ratio
    weights[labels == 1] = ratio
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
