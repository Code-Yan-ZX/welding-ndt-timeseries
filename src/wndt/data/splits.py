"""Canonical train/val/test splits for the ASIMoW / Metal Arc Welding dataset.

Convention = the PAPER / dataset README convention (Hahn et al., CIKM 2024 and
Zenodo 10017718 README):

  val  (early stopping, small overlap-joint set):
      (3,32),(3,18),(1,27),(3,19),(3,17),(2,21),(1,20),(1,11)
  test (T-joints, distribution shift by design):
      (3,3),(2,10),(1,24),(3,24),(1,32),(2,1),(1,10),(1,16)
  train = everything else.

WARNING: the official tmdt-buw repository code returns these same two pair-lists
with SWAPPED names ('test_ids' = our val pairs, 'val_ids' = our test pairs) in
dataloader/utils.py::get_val_test_ids(). We match by pair SETS and assert exact
row counts instead of trusting variable names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wndt.utils.logging import get_logger

log = get_logger(__name__)

VAL_PAIRS = [(3, 32), (3, 18), (1, 27), (3, 19), (3, 17), (2, 21), (1, 20), (1, 11)]
TEST_PAIRS = [(3, 3), (2, 10), (1, 24), (3, 24), (1, 32), (2, 1), (1, 10), (1, 16)]

# Expected LABELED row counts for Zenodo v2 (record 10017718), verified from
# the downloaded CSV on 2026-08-05 (209,185 rows total, 96,408 labeled).
# NOTE: the 2025 OOD paper's counts (48,758/4,676/104,531) refer to a NEWER
# dataset version with extended labels; the CIKM 2024 anchor (~79.7% acc) and
# the official repo both operate on this v2 file.
EXPECTED_COUNTS = {"train": 74_732, "val": 10_614, "test": 11_062}


def _pair_mask(exp: np.ndarray, run: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(exp), dtype=bool)
    for e, r in pairs:
        mask |= (exp == e) & (run == r)
    return mask


def build_splits(out_dir: Path, verify_counts: bool = True) -> dict[str, np.ndarray]:
    """Build labeled train/val/test index arrays; save under out_dir/splits/.

    Returns dict split-name -> int64 index array (into the full row array).
    """
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    meta_exp = np.asarray(np.load(out_dir / "meta_exp.npy", mmap_mode="r"))
    meta_run = np.asarray(np.load(out_dir / "meta_run.npy", mmap_mode="r"))
    meta_label = np.asarray(np.load(out_dir / "meta_label.npy", mmap_mode="r"))
    labeled = meta_label != -1

    val_mask = _pair_mask(meta_exp, meta_run, VAL_PAIRS)
    test_mask = _pair_mask(meta_exp, meta_run, TEST_PAIRS)
    assert not np.any(val_mask & test_mask), "val/test (exp,run) pairs overlap"
    train_mask = ~(val_mask | test_mask)

    pairs_seen = set(zip(np.asarray(meta_exp).tolist(), np.asarray(meta_run).tolist()))
    n_pairs = len(pairs_seen)
    log.info("distinct (experiment, welding_run) pairs: %d", n_pairs)

    idx = {
        "train": np.nonzero(train_mask & labeled)[0].astype(np.int64),
        "val": np.nonzero(val_mask & labeled)[0].astype(np.int64),
        "test": np.nonzero(test_mask & labeled)[0].astype(np.int64),
    }

    for name, arr in idx.items():
        np.save(splits_dir / f"{name}_idx.npy", arr)
        pos_rate = float(meta_label[arr].mean())
        log.info("split %-5s labeled rows: %7d | pos-rate (good): %.4f",
                 name, len(arr), pos_rate)

    if verify_counts:
        for name, expected in EXPECTED_COUNTS.items():
            got = len(idx[name])
            assert got == expected, (
                f"split '{name}' has {got} labeled rows, expected {expected}. "
                "Check split pair mapping before trusting any comparison!"
            )
        log.info("all split counts match expected values: %s", EXPECTED_COUNTS)
    else:
        log.warning("count verification skipped")

    return idx


def load_split_idx(out_dir: Path) -> dict[str, np.ndarray]:
    splits_dir = Path(out_dir) / "splits"
    return {name: np.load(splits_dir / f"{name}_idx.npy")
            for name in ("train", "val", "test")}
