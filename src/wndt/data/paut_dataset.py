"""Torch Dataset over PAUT (phased-array ultrasonic) per-position signals.

Mirrors ``SAWSeriesDataset`` so the same ``ClassificationTrainer`` /
``compute_metrics`` infrastructure is reused, but exposes several input views
of the same scan position:

  beam="env"    -> (1, T)        max-over-beams envelope (deterministic; eval / classic-ML)
  beam="mean"   -> (1, T)        mean-over-beams A-scan
  beam="random" -> (1, T)        a uniformly random beam per __getitem__ call
                                 (training augmentation; 49x effective samples)
  beam="bscan"  -> (49, T)       full per-beam B-scan (for the spectral-spatial-
                                 frequency model)

The on-disk ``ascans.npy`` is (N, 49, T) float32 and ``env.npy`` is (N, T).
Labels are read from ``meta_label.npy``.  Normalization uses per-timestep
mean/std computed on TRAIN (``norm_stats.json``); ``global`` / ``sample_z`` /
``none`` are also supported for ablations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wndt.data.dataset import make_weighted_sampler  # re-export for convenience

NORM_MODES = ("per_timestep", "global", "sample_z", "none")
BEAM_MODES = ("env", "mean", "random", "bscan", "expand")


class PAUTSeriesDataset(Dataset):
    """One sample = one scan-position signal, tensor shape (C, T).

    ``n_channels`` is 1 for env/mean/random/expand and ``n_beams`` (49) for
    bscan.  ``expand`` unrolls every position into ``n_beams`` per-beam samples
    (length = N_pos * n_beams), giving SAW-scale training data with the same
    position-level label repeated per beam; the label array is expanded to
    match so ``WeightedRandomSampler`` works unchanged.
    """

    def __init__(self, processed_dir: str | Path, indices: np.ndarray,
                 *, beam: str = "env", norm_mode: str = "per_timestep",
                 n_beams: int | None = None):
        assert norm_mode in NORM_MODES, f"unknown norm mode {norm_mode}"
        assert beam in BEAM_MODES, f"unknown beam mode {beam}"
        processed_dir = Path(processed_dir)
        self.ascans = np.load(processed_dir / "ascans.npy", mmap_mode="r")
        self.env = np.load(processed_dir / "env.npy", mmap_mode="r")
        labels_all = np.load(processed_dir / "meta_label.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels_all[self.indices]).astype(np.int64)
        self.beam = beam
        self.norm_mode = norm_mode
        self.n_beams = int(self.ascans.shape[1]) if n_beams is None else int(n_beams)
        self.seq_len = int(self.ascans.shape[2])
        self.n_channels = self.n_beams if beam == "bscan" else 1
        if beam == "expand":
            # repeat each position's label n_beams times
            self.labels = np.repeat(self.labels, self.n_beams)
        self.mean = self.std = None
        self.ts_mean = self.ts_std = None
        if norm_mode == "per_timestep":
            with open(processed_dir / "norm_stats.json", "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            self.ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
            self.ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
        elif norm_mode == "global":
            with open(processed_dir / "norm_stats.json", "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            self.mean = np.float32(stats["global"]["mean"])
            self.std = np.float32(stats["global"]["std"])

    def __len__(self) -> int:
        if self.beam == "expand":
            return len(self.indices) * self.n_beams
        return len(self.indices)

    def _get_raw(self, gi: int) -> np.ndarray:
        """Return the (C, T) float32 signal for global index ``gi`` (pre-norm)."""
        if self.beam == "env":
            x = np.array(self.env[gi], dtype=np.float32)[None, :]        # (1, T)
        elif self.beam == "bscan":
            x = np.array(self.ascans[gi], dtype=np.float32)              # (49, T)
        else:
            full = np.array(self.ascans[gi], dtype=np.float32)           # (49, T)
            if self.beam == "random":
                b = int(np.random.randint(self.n_beams))
                x = full[b:b + 1]                                        # (1, T)
            else:  # mean
                x = full.mean(axis=0, keepdims=True)                     # (1, T)
        return x

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.norm_mode == "per_timestep":
            x = (x - self.ts_mean) / self.ts_std
        elif self.norm_mode == "global":
            x = (x - self.mean) / self.std
        elif self.norm_mode == "sample_z":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            x = (x - mu) / sd
        return x

    def __getitem__(self, i: int):
        if self.beam == "expand":
            pos_i = i // self.n_beams
            beam_i = i % self.n_beams
            x = np.array(self.ascans[int(self.indices[pos_i])][beam_i],
                         dtype=np.float32)[None, :]                  # (1, T)
            lab = int(self.labels[i])
        else:
            x = self._norm(self._get_raw(int(self.indices[i])))
            lab = int(self.labels[i])
        x = self._norm(x)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(lab, dtype=torch.long)
