#!/usr/bin/env python
"""Preprocess the Submerged Arc Welding (SAW) Zenodo dataset into window-level
defect-detection samples.

Pipeline per bead:
  1. load HDF5  -> data1 (5 kHz): current_a, current_b, voltage_a, voltage_b
                   data0 (66.7 Hz): process_status, idx_data1 (-> data1 index)
  2. active welding region = data1 indices spanned by data0 rows with
     process_status == 1 (drops pre/post-weld lead-in zeros)
  3. slide a window (default 512 samples = 0.1 s @ 5 kHz) with stride 256
  4. label each window 1 if its [i, i+W) sample range overlaps ANY defect
     sample range for that bead (from defects_xlocation.xlsx, '# sample' cols
     are data1 indices), else 0
  5. splits by coupon: train = PP3,PP4,PP5 ; val = PP6 ; test = PP7

Outputs to data/processed/saw/:
  waves.npy        (N, 4, W) float32
  meta_label.npy   (N,) int64   {0=clean, 1=defect}
  meta_coupon.npy  (N,) <U8     coupon id
  meta_bead.npy    (N,) <U16    bead name
  meta_defect_type.npy (N,) int  dominant defect code in window (0 if clean)
  norm_stats.json  per-channel mean/std from TRAIN
  splits.npz       train/val/test index arrays
  meta_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA1_FIELDS = ["current_a", "current_b", "voltage_a", "voltage_b"]
DEFECT_CODES = {1: "Porosity", 2: "Lack of fusion", 3: "Slag inclusion",
                4: "Metallic inclusion", 5: "Projections", 6: "Cracks"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/saw/ZENODO_Penelope")
    ap.add_argument("--out", default="data/processed/saw")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--train-coupons", nargs="+", default=["PP3", "PP4", "PP5"])
    ap.add_argument("--val-coupons", nargs="+", default=["PP6"])
    ap.add_argument("--test-coupons", nargs="+", default=["PP7"])
    return ap.parse_args()


def coupon_dirs(root: Path, coupons: list[str]) -> list[Path]:
    out = []
    for c in coupons:
        # match e.g. "PP3" possibly with surrounding dirs
        matches = [p for p in root.iterdir() if p.is_dir() and p.name == c]
        out.extend(matches or [p for p in root.iterdir() if p.is_dir() and p.name.startswith(c)])
    return out


def load_defect_ranges(xlsx: Path, coupon: str) -> dict[int, list[tuple[int, int, int]]]:
    """Return {bead_int: [(defect_code, x_init_sample, x_end_sample), ...]}."""
    xls = pd.ExcelFile(xlsx)
    # the per-coupon sheet is named like the coupon (PP3, PP4, ...)
    sheet = coupon if coupon in xls.sheet_names else xls.sheet_names[-1]
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df.rename(columns=lambda s: s.strip())
    out: dict[int, list[tuple[int, int, int]]] = {}
    for _, r in df.iterrows():
        bead = int(r["bead"])
        dcode = int(r["defect"])
        s = int(r["x_init [# sample]"])
        e = int(r["x_end [# sample]"])
        if e < s:
            s, e = e, s
        out.setdefault(bead, []).append((dcode, s, e))
    return out


def bead_signal_and_active(h5path: Path):
    """Return (data1 array [L,4], active_start, active_end) for a bead."""
    with h5py.File(h5path, "r") as fh:
        # data1 table (5 kHz) -- some files use 'data1', some 'Data1'
        key = next(k for k in fh.keys() if k.lower() == "data1")
        d1 = fh[key][:]
        cols = {n: i for i, n in enumerate(d1.dtype.names)}
        sig = np.stack([d1[c].astype(np.float32) for c in DATA1_FIELDS], axis=1)  # [L,4]
        key0 = next(k for k in fh.keys() if k.lower() == "data0")
        d0 = fh[key0][:]
        ps = d0["process_status"].astype(np.float32)
        idx1 = d0["idx_data1"].astype(np.int64)
    active = ps > 0.5
    if active.any():
        s = int(idx1[active].min())
        e = int(idx1[active].max()) + 75  # ~one 66.7 Hz interval past last active row
    else:
        # fallback: active where current magnitude is non-trivial
        mag = np.abs(sig[:, 0]) + np.abs(sig[:, 2])
        thr = 0.1 * mag.max()
        act = mag > thr
        s = int(np.argmax(act)) if act.any() else 0
        e = int(len(sig) - np.argmax(act[::-1])) if act.any() else len(sig)
    e = min(e, len(sig))
    return sig, s, e


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coupons = args.train_coupons + args.val_coupons + args.test_coupons
    waves, labels, coupon_arr, bead_arr, defect_arr = [], [], [], [], []

    for coupon in coupons:
        cdir = root / coupon
        if not cdir.exists():
            print(f"[warn] missing coupon dir {cdir}")
            continue
        xlsx = list((cdir / "2. ndt_data").glob("defects_xlocation.xlsx")) if (cdir / "2. ndt_data").exists() else []
        xlsx += list(cdir.glob("**/defects_xlocation.xlsx"))
        defect_map = load_defect_ranges(xlsx[0], coupon) if xlsx else {}
        pdir = cdir / "1. process_data"
        bead_dirs = sorted(p for p in pdir.iterdir() if p.is_dir()) if pdir.exists() else []
        print(f"{coupon}: {len(bead_dirs)} beads, defects in {len(defect_map)} beads")
        for bdir in bead_dirs:
            m = re.search(r"(\d+)\s*$", bdir.name)
            bead_no = int(m.group(1)) if m else -1
            h5s = sorted(bdir.glob("*.hdf5"))
            if not h5s:
                continue
            sig, s, e = bead_signal_and_active(h5s[0])
            ranges = defect_map.get(bead_no, [])
            # clip defect ranges to signal bounds
            ranges = [(dc, max(0, a), min(e, b)) for dc, a, b in ranges if b > s and a < e]
            L = e - s
            if L < args.window:
                continue
            for i0 in range(s, e - args.window + 1, args.stride):
                w = sig[i0:i0 + args.window]  # [W,4]
                i_end = i0 + args.window
                hit = [(dc, a, b) for dc, a, b in ranges if a < i_end and b > i0]
                lab = 1 if hit else 0
                dcode = max((dc for dc, _, _ in hit), default=0)
                waves.append(np.transpose(w, (1, 0)))          # [4,W]
                labels.append(lab)
                coupon_arr.append(coupon)
                bead_arr.append(bdir.name)
                defect_arr.append(dcode)

    waves = np.stack(waves).astype(np.float32)            # [N,4,W]
    labels = np.asarray(labels, dtype=np.int64)
    coupon_arr = np.asarray(coupon_arr)
    bead_arr = np.asarray(bead_arr)
    defect_arr = np.asarray(defect_arr, dtype=np.int64)
    N = len(labels)
    pos = float(labels.mean())
    print(f"\nTotal windows: {N} | defect-rate: {pos:.4f}")
    for c in coupons:
        m = coupon_arr == c
        if m.any():
            print(f"  {c}: {int(m.sum())} windows, defect-rate {float(labels[m].mean()):.4f}")

    # splits by coupon
    tr = np.where(np.isin(coupon_arr, args.train_coupons))[0]
    va = np.where(np.isin(coupon_arr, args.val_coupons))[0]
    te = np.where(np.isin(coupon_arr, args.test_coupons))[0]
    print(f"  splits -> train {len(tr)} / val {len(va)} / test {len(te)}")

    # norm stats from train only
    tr_w = waves[tr].reshape(-1, waves.shape[1], 1)
    # pool over time -> per-channel mean/std
    flat = waves[tr].transpose(0, 2, 1).reshape(-1, waves.shape[1])  # [Ntr*W, 4]
    mean = flat.mean(axis=0).tolist()
    std = (flat.std(axis=0) + 1e-8).tolist()

    np.save(out / "waves.npy", waves)
    np.save(out / "meta_label.npy", labels)
    np.save(out / "meta_coupon.npy", coupon_arr)
    np.save(out / "meta_bead.npy", bead_arr)
    np.save(out / "meta_defect_type.npy", defect_arr)
    np.savez(out / "splits.npz", train=tr, val=va, test=te)
    with open(out / "norm_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"mean": mean, "std": std,
                   "channels": DATA1_FIELDS, "window": args.window,
                   "stride": args.stride, "fs_hz": 5000}, fh, indent=2)
    summary = {"n_windows": int(N), "defect_rate": pos,
               "n_channels": 4, "window": args.window, "stride": args.stride,
               "channels": DATA1_FIELDS, "defect_codes": DEFECT_CODES,
               "splits": {"train": len(tr), "val": len(va), "test": len(te)},
               "train_coupons": args.train_coupons, "val_coupons": args.val_coupons,
               "test_coupons": args.test_coupons}
    with open(out / "meta_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
