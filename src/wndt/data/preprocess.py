"""Convert the Zenodo ASIMoW CSV into memory-mapped numpy arrays.

Input : data/raw/processed_asimow_dataset.csv
        columns: experiment, welding_run, labels, V_0..V_199, I_0..I_199
        (one row per welding cycle; label -1 = unlabeled)
Output: data/processed/waves.npy        float32 (N, 2, 200)  channel order (V, I)
        data/processed/meta_exp.npy     int16   (N,)
        data/processed/meta_run.npy     int16   (N,)
        data/processed/meta_label.npy   int8    (N,)

Channel order (V, I) matches the official tmdt-buw repository
(np.concatenate((v, i), axis=2) in asimow_dataloader.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wndt.utils.logging import get_logger

log = get_logger(__name__)

CYCLE_LEN = 200
V_COLS = [f"V_{i}" for i in range(CYCLE_LEN)]
I_COLS = [f"I_{i}" for i in range(CYCLE_LEN)]


def count_rows(csv_path: Path) -> int:
    """Fast row count (excluding header)."""
    n = 0
    with open(csv_path, "rb") as fh:
        for _ in fh:
            n += 1
    return n - 1


def _label_col(chunk: pd.DataFrame) -> str:
    for name in ("labels", "label"):
        if name in chunk.columns:
            return name
    raise KeyError(f"label column not found in {list(chunk.columns)[:6]}...")


def csv_to_memmap(csv_path: Path, out_dir: Path, chunksize: int = 50_000) -> int:
    """Two-pass conversion: count rows, allocate memmaps, fill chunk by chunk.

    Returns the number of rows written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = count_rows(csv_path)
    log.info("CSV rows (excl. header): %d", n_total)

    waves = np.lib.format.open_memmap(
        out_dir / "waves.npy", mode="w+", dtype=np.float32,
        shape=(n_total, 2, CYCLE_LEN),
    )
    meta_exp = np.lib.format.open_memmap(
        out_dir / "meta_exp.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    meta_run = np.lib.format.open_memmap(
        out_dir / "meta_run.npy", mode="w+", dtype=np.int16, shape=(n_total,))
    meta_label = np.lib.format.open_memmap(
        out_dir / "meta_label.npy", mode="w+", dtype=np.int8, shape=(n_total,))

    start = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        stop = start + len(chunk)
        if stop > n_total:
            raise RuntimeError(f"more rows than counted: {stop} > {n_total}")
        waves[start:stop, 0, :] = chunk[V_COLS].to_numpy(dtype=np.float32)
        waves[start:stop, 1, :] = chunk[I_COLS].to_numpy(dtype=np.float32)
        meta_exp[start:stop] = chunk["experiment"].to_numpy(dtype=np.int16)
        meta_run[start:stop] = chunk["welding_run"].to_numpy(dtype=np.int16)
        meta_label[start:stop] = chunk[_label_col(chunk)].to_numpy(dtype=np.int8)
        start = stop
        log.info("processed %d / %d rows", stop, n_total)

    # flush
    del waves, meta_exp, meta_run, meta_label
    return start


def spot_check(csv_path: Path, out_dir: Path, n_rows: int = 5, seed: int = 0) -> None:
    """Verify that random memmap rows equal the CSV values exactly."""
    rng = np.random.default_rng(seed)
    waves = np.load(out_dir / "waves.npy", mmap_mode="r")
    meta_label = np.load(out_dir / "meta_label.npy", mmap_mode="r")
    meta_exp = np.load(out_dir / "meta_exp.npy", mmap_mode="r")
    meta_run = np.load(out_dir / "meta_run.npy", mmap_mode="r")
    idx = sorted(rng.choice(len(waves), size=n_rows, replace=False).tolist())
    df = pd.read_csv(csv_path, skiprows=lambda i: i > 0 and i not in {r + 1 for r in idx})
    for k, row_pos in enumerate(idx):
        row = df.iloc[k]
        v_csv = row[V_COLS].to_numpy(dtype=np.float32)
        i_csv = row[I_COLS].to_numpy(dtype=np.float32)
        assert np.array_equal(waves[row_pos, 0], v_csv), f"V mismatch at row {row_pos}"
        assert np.array_equal(waves[row_pos, 1], i_csv), f"I mismatch at row {row_pos}"
        assert int(meta_label[row_pos]) == int(row[_label_col(df)])
        assert int(meta_exp[row_pos]) == int(row["experiment"])
        assert int(meta_run[row_pos]) == int(row["welding_run"])
    log.info("spot check OK on rows %s", idx)


def compute_norm_stats(out_dir: Path, train_idx: np.ndarray) -> dict:
    """Per-channel mean/std over all training timesteps (official MyScaler protocol:
    StandardScaler fit on train reshaped to (-1, 2))."""
    waves = np.load(out_dir / "waves.npy", mmap_mode="r")
    train_waves = waves[train_idx]  # (n, 2, 200)
    # channel-last before flattening, so each row is a (V_t, I_t) pair --
    # this reproduces the official MyScaler (fit on train reshaped to (-1, 2))
    flat = train_waves.transpose(0, 2, 1).reshape(-1, 2).astype(np.float64)
    stats = {
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "n_train_cycles": int(len(train_idx)),
    }
    with open(out_dir / "norm_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("norm stats (per channel V,I): mean=%s std=%s",
             stats["mean"], stats["std"])
    return stats
