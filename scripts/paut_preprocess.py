#!/usr/bin/env python
"""Preprocess the PAUT (phased-array ultrasonic) .nde files of the SAW Zenodo
dataset into position-level defect-detection samples.

An ``.nde`` file is an HDF5 container written by Evident/Olympus OmniScan X3
(NDE-FileFormat-Schema-3.1.0).  For each coupon PP3-PP7 we read the **90-family
file** (``PAUT_90.nde`` / ``PAUT_90+.nde`` / ``..._90.nde``) **DataGroup 0**:
the 71-degree, 49-beam group, amplitude volume of shape
``(n_pos, 49, 3500)`` int16 (rectified).  This is the genuine NDT signal — as
opposed to the GMAW/SAW process signals (current/voltage) used previously.

Pipeline per coupon:
  1. read amplitude (n_pos, 49, 3500) + scan geometry from the Setup JSON
     (scan-axis resolution, beam u-coordinate offset) so defect x[mm] maps
     onto scan positions
  2. downsample every beam A-scan 3500 -> 512 by max-pooling (the signal is a
     rectified envelope, so max-pool preserves echo peaks; the per-position
     A-scan is extremely sparse, ~0.5-0.9 % active samples)
  3. build a per-position label from defects_xlocation.xlsx:
       - defect (1) if the position's x-range overlaps any **localized** defect
         (axial extent < ``big_defect_mm``, default 50 mm)
       - clean (0) otherwise
     Full-length blanket defects (>= ``big_defect_mm`` — e.g. a crack spanning
     the whole weld on one bead) are treated as background.  Rationale: at the
     PAUT position level such defects blanket the whole scan (every position is
     defective), which makes binary detection degenerate (4/5 coupons would be
     100 % positive).  Localized-defect detection is the non-trivial, standard
     PAUT inspection task; the blanket-crack exclusion is a documented scope
     limitation (see reports/PAUT相控阵缺陷检测实验报告.md).
  4. splits by coupon, mirroring the SAW process-signal experiment for a
     direct modality comparison:  train = PP3, PP4, PP5 ; val = PP6 ; test = PP7

Outputs to data/processed/paut/:
  ascans.npy        (N, 49, 512) float32   per-position per-beam A-scan (B-scan)
  env.npy           (N, 512)     float32   max-over-beams envelope (downsampled)
  meta_label.npy    (N,) int64              {0=clean, 1=defect}
  meta_coupon.npy   (N,) <U8               coupon id
  meta_pos.npy      (N,) int64             scan-position index within coupon
  meta_defect_type.npy (N,) int64          dominant localized defect code (0 clean)
  splits.npz        train/val/test index arrays
  norm_stats.json   per-timestep mean/std from TRAIN (+ global scalars)
  meta_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "data/raw/saw/ZENODO_Penelope"
OUT_DEFAULT = REPO / "data/processed/paut"

DEFECT_CODES = {1: "Porosity", 2: "Lack of fusion", 3: "Slag inclusion",
                4: "Metallic inclusion", 5: "Projections", 6: "Cracks"}
BIG_DEFECT_MM = 50.0      # axial extent >= this -> treated as background
TARGET_LEN = 512          # downsampled A-scan length (matches SAW seq_len + MOMENT)
TRAIN_COUPONS = ["PP3", "PP4", "PP5"]
VAL_COUPONS = ["PP6"]
TEST_COUPONS = ["PP7"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--target-len", type=int, default=TARGET_LEN)
    ap.add_argument("--big-defect-mm", type=float, default=BIG_DEFECT_MM)
    ap.add_argument("--group", type=int, default=0, help="DataGroup index (0=71deg/49beam)")
    return ap.parse_args()


def find_90_file(ndt_dir: Path) -> Path:
    """Pick the 90-family PAUT file (90 / 90+ / *_90.nde), never 270."""
    cands = sorted(ndt_dir.glob("*.nde"))
    for p in cands:
        n = p.name
        if "270" in n:
            continue
        if "PAUT_90" in n or n.endswith("_90.nde"):
            return p
    raise FileNotFoundError(f"no 90-family .nde under {ndt_dir}")


def read_group0(nde: Path, gidx: int):
    """Return (amplitude (n_pos,n_beams,n_samples) float32, offset_mm, res_mm)."""
    with h5py.File(nde, "r") as f:
        ds = f[f"Domain/DataGroups/{gidx}/Datasets/0/Amplitude"]
        amp = ds[:].astype(np.float32)
        setup = json.loads(f["Domain/Setup"][()].decode("utf-8"))
    g = setup["groups"][gidx]
    dims = g["dataset"]["ascan"]["amplitude"]["dimensions"]
    scan_dim = next(d for d in dims if d["axis"] == "UCoordinate")
    beam_dim = next(d for d in dims if d["axis"] == "Beam")
    beams = beam_dim["beams"]
    offset_mm = float(beams[0]["uCoordinateOffset"]) * 1e3
    res_mm = float(scan_dim["resolution"]) * 1e3
    return amp, offset_mm, res_mm


def downsample_max(x: np.ndarray, target: int) -> np.ndarray:
    """Max-pool the last axis to ``target`` bins (rectified envelope -> peaks)."""
    L = x.shape[-1]
    if L == target:
        return x
    if L < target:
        # linear upsample (rare; keeps dtype/shape contract)
        idx = np.linspace(0, L - 1, target)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, L - 1)
        frac = (idx - lo).astype(np.float32)
        return x[..., lo] * (1 - frac) + x[..., hi] * frac
    parts = np.array_split(x, target, axis=-1)
    return np.stack([p.max(axis=-1) for p in parts], axis=-1)


def load_defects(ndt_dir: Path, coupon: str) -> pd.DataFrame:
    df = pd.read_excel(ndt_dir / "defects_xlocation.xlsx", sheet_name=coupon)
    df = df.rename(columns=lambda s: s.strip())
    df["len_mm"] = df["x_end [mm]"] - df["x_init [mm]"]
    return df


def position_labels(defects: pd.DataFrame, offset_mm: float, res_mm: float,
                    n_pos: int, big_mm: float):
    """Localized-defect label + dominant type per scan position.

    A position is defect(1) if its x-range overlaps any defect with axial
    extent < big_mm.  Defects >= big_mm (blanket cracks/regions) are ignored
    (treated as background).  Returns (label[n_pos], type[n_pos]).
    """
    lab = np.zeros(n_pos, dtype=np.int64)
    typ = np.zeros(n_pos, dtype=np.int64)
    local = defects[defects["len_mm"] < big_mm]
    for _, r in local.iterrows():
        s = (float(r["x_init [mm]"]) - offset_mm) / res_mm
        e = (float(r["x_end [mm]"]) - offset_mm) / res_mm
        i0, i1 = max(0, int(round(s))), min(n_pos - 1, int(round(e)))
        if i1 < i0:
            continue
        lab[i0:i1 + 1] = 1
        # dominant type = max code overlapping (only among localized defects)
        cur = typ[i0:i1 + 1]
        typ[i0:i1 + 1] = np.maximum(cur, int(r["defect"]))
    return lab, typ


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ascans, envs = [], []
    labels, coupons, poses, types = [], [], [], []
    cursor = 0
    per_coupon = {}
    print(f"=== PAUT preprocess | group {args.group} | target_len {args.target_len} "
          f"| big_defect {args.big_defect_mm}mm ===")
    for c in TRAIN_COUPONS + VAL_COUPONS + TEST_COUPONS:
        ndt = args.root / c / "2. ndt_data"
        nde = find_90_file(ndt)
        amp, off, res = read_group0(nde, args.group)
        n_pos, n_beams, n_samp = amp.shape
        ds = downsample_max(amp, args.target_len)            # (n_pos, n_beams, T)
        env = ds.max(axis=1)                                  # (n_pos, T)
        lab, typ = position_labels(load_defects(ndt, c), off, res, n_pos,
                                   args.big_defect_mm)
        ascans.append(ds.astype(np.float32))
        envs.append(env.astype(np.float32))
        labels.append(lab)
        types.append(typ)
        coupons.append(np.full(n_pos, c, dtype="<U8"))
        poses.append(np.arange(cursor, cursor + n_pos, dtype=np.int64))
        cursor += n_pos
        per_coupon[c] = {"n_pos": n_pos, "offset_mm": off, "res_mm": res,
                         "n_beams": n_beams, "n_samples": n_samp,
                         "defect_pos": int(lab.sum()),
                         "defect_rate": float(lab.mean()),
                         "nde": nde.name}
        print(f"  {c}: {nde.name} | n_pos={n_pos} beams={n_beams} samp={n_samp} "
              f"offset={off:.1f}mm | defect_pos={int(lab.sum())} "
              f"({100*lab.mean():.1f}%) | types={ {DEFECT_CODES[t]: int((typ==t).sum()) for t in np.unique(typ) if t>0} }")

    ascans = np.concatenate(ascans, axis=0)          # (N, 49, T)
    envs = np.concatenate(envs, axis=0)              # (N, T)
    labels = np.concatenate(labels, axis=0)
    types = np.concatenate(types, axis=0)
    coupons = np.concatenate(coupons, axis=0)
    poses = np.concatenate(poses, axis=0)
    N, n_beams, T = ascans.shape

    # splits by coupon
    def idx_of(cs):
        return np.nonzero(np.isin(coupons, cs))[0].astype(np.int64)
    tr, va, te = idx_of(TRAIN_COUPONS), idx_of(VAL_COUPONS), idx_of(TEST_COUPONS)

    # norm stats from TRAIN: per-timestep (across beams & positions) + global
    tr_asc = ascans[tr]                               # (ntr, 49, T)
    tr_flat = tr_asc.reshape(-1, T)                   # (ntr*49, T)
    ts_mean = tr_flat.mean(axis=0).astype(np.float32)     # (T,)
    ts_std = (tr_flat.std(axis=0) + 1e-8).astype(np.float32)
    g_mean = float(tr_flat.mean())
    g_std = float(tr_flat.std() + 1e-8)
    norm_stats = {"per_timestep": {"mean": ts_mean.tolist(), "std": ts_std.tolist()},
                  "global": {"mean": g_mean, "std": g_std},
                  "target_len": T, "n_beams": int(n_beams)}

    np.save(out / "ascans.npy", ascans)
    np.save(out / "env.npy", envs)
    np.save(out / "meta_label.npy", labels)
    np.save(out / "meta_defect_type.npy", types)
    np.save(out / "meta_coupon.npy", coupons)
    np.save(out / "meta_pos.npy", poses)
    np.savez(out / "splits.npz", train=tr, val=va, test=te)
    with open(out / "norm_stats.json", "w", encoding="utf-8") as fh:
        json.dump(norm_stats, fh, indent=2)
    summary = {
        "n_samples": int(N), "n_beams": int(n_beams), "target_len": int(T),
        "defect_rate": float(labels.mean()),
        "big_defect_mm": args.big_defect_mm,
        "label_policy": "localized defects (< big_defect_mm) only; "
                        "blanket defects (>= big_defect_mm) treated as background",
        "splits": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "train_coupons": TRAIN_COUPONS, "val_coupons": VAL_COUPONS,
        "test_coupons": TEST_COUPONS,
        "defect_codes": {str(k): v for k, v in DEFECT_CODES.items()},
        "per_coupon": per_coupon,
        "defect_rate_train": float(labels[tr].mean()),
        "defect_rate_val": float(labels[va].mean()),
        "defect_rate_test": float(labels[te].mean()),
    }
    with open(out / "meta_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nN={N} beams={n_beams} T={T} | defect_rate={labels.mean():.4f} "
          f"(tr {labels[tr].mean():.4f} / va {labels[va].mean():.4f} / "
          f"te {labels[te].mean():.4f})")
    print(f"splits: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
