#!/usr/bin/env python3
"""Inspect / validate the PAUT (.nde) files of the SAW Zenodo dataset.

An ``.nde`` file is an HDF5 container written by Evident/Olympus OmniScan X3
(NDE-FileFormat-Schema-3.1.0).  This script:

  1. reads the main DataGroup amplitude volume  (n_pos, n_beams, n_samples)
  2. parses the Setup JSON for per-group geometry
     (scan-axis resolution / beam u-coordinate offset / sound velocity /
      ascan sampling) so defect x[mm] can be mapped onto scan positions
  3. builds a per-scan-position defect label from defects_xlocation.xlsx
  4. **empirically calibrates the mm->position offset** by sweeping it and
     measuring how well the amplitude envelope separates defect vs clean
     positions (AUC + mean separation); this validates the spatial alignment
  5. saves a diagnostic PNG (envelope + defect bands + offset-sweep curve)

Exploratory only; informs the PAUT preprocessing pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/raw/saw/ZENODO_Penelope"

DEFECT_CODES = {1: "Porosity", 2: "Lack of fusion", 3: "Slag inclusion",
                4: "Metallic inclusion", 5: "Projections", 6: "Cracks"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--coupon", default="PP3")
    ap.add_argument("--angle", default="90",
                    help="nde name fragment: PAUT_<angle>.nde (90/270/90+)")
    ap.add_argument("--group", type=int, default=0, help="DataGroup index")
    ap.add_argument("--out", type=Path, default=REPO.parent.parent / "reports/figs")
    return ap.parse_args()


def find_nde(root: Path, coupon: str, angle: str) -> Path:
    d = root / coupon / "2. ndt_data"
    cands = sorted(d.glob("*.nde"))
    if not cands:
        sys.exit(f"no .nde under {d}")
    if angle:
        for p in cands:
            if f"PAUT_{angle}.nde" in p.name or p.name.endswith(f"_{angle}.nde"):
                return p
    return cands[0]


def group_geometry(setup: dict, gidx: int) -> dict:
    """Extract scan/beam/time geometry for one DataGroup from the Setup JSON."""
    g = setup["groups"][gidx]
    dims = g["dataset"]["ascan"]["amplitude"]["dimensions"]
    # dim 0 = scan axis (UCoordinate), dim 1 = Beam, dim 2 = time
    scan_dim = next(d for d in dims if d["axis"] == "UCoordinate")
    beam_dim = next(d for d in dims if d["axis"] == "Beam")
    beams = beam_dim["beams"]
    info = {
        "name": g.get("name"), "n_pos": scan_dim["quantity"],
        "scan_res_m": scan_dim["resolution"],        # m per position
        "n_beams": len(beams),
        "u_offset_m": beams[0]["uCoordinateOffset"],  # probe index offset (m)
        "velocity": beams[0]["velocity"],
        "refracted_angle": beams[0]["refractedAngle"],
        "skew_angle": beams[0]["skewAngle"],
        "ultrasound_offset_s": beams[0]["ultrasoundOffset"],
    }
    paut = g.get("paut", {})
    b0 = (paut.get("beams") or [{}])[0]
    info["ascan_start_s"] = b0.get("ascanStart", info["ultrasound_offset_s"])
    info["ascan_length_s"] = b0.get("ascanLength", 0.0)
    # n_samples from the on-disk dataset (caller fills) ; time resolution:
    # digitizingFrequency / decimation. We derive sample_dt from ascan_length
    # / n_samples later once shape is known.
    return info


def read_amplitude(ndepath: Path, gidx: int):
    with h5py.File(ndepath, "r") as f:
        ds = f[f"Domain/DataGroups/{gidx}/Datasets/0/Amplitude"]
        amp = ds[:]                       # (n_pos, n_beams, n_samples) int16
        setup = json.loads(f["Domain/Setup"][()].decode("utf-8"))
    return amp, setup


def load_defects(root: Path, coupon: str) -> pd.DataFrame:
    xlsx = root / coupon / "2. ndt_data" / "defects_xlocation.xlsx"
    df = pd.read_excel(xlsx, sheet_name=coupon)
    df = df.rename(columns=lambda s: s.strip())
    return df


def label_vector(defects: pd.DataFrame, offset_mm: float, res_mm: float,
                 n_pos: int) -> np.ndarray:
    """1 if scan position i (x_mm = offset + i*res) lies inside any defect."""
    lab = np.zeros(n_pos, dtype=np.int64)
    for _, r in defects.iterrows():
        s = (float(r["x_init [mm]"]) - offset_mm) / res_mm
        e = (float(r["x_end [mm]"]) - offset_mm) / res_mm
        i0, i1 = int(round(s)), int(round(e))
        i0, i1 = max(0, i0), min(n_pos - 1, i1)
        if i1 >= i0:
            lab[i0:i1 + 1] = 1
    return lab


def envelope(amp: np.ndarray) -> np.ndarray:
    """Per-position detection strength: max |amplitude| over beams & time."""
    a = amp.astype(np.float32)
    # PAUT rectified amplitude is >=0; use max as the envelope. If RF (signed)
    # around a DC baseline, max abs is a safer energy proxy.
    if a.min() < 0:
        a = np.abs(a - a.mean())
    return a.reshape(a.shape[0], -1).max(axis=1)   # (n_pos,)


def auc(sep_scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, sep_scores))


def main() -> None:
    args = parse_args()
    nde = find_nde(args.root, args.coupon, args.angle)
    print(f"=== {args.coupon} / {nde.name} (group {args.group}) ===")
    amp, setup = read_amplitude(nde, args.group)
    geo = group_geometry(setup, args.group)
    geo["n_samples"] = amp.shape[2]
    geo["sample_dt_s"] = geo["ascan_length_s"] / geo["n_samples"]
    geo["max_depth_mm"] = 0.5 * geo["velocity"] * geo["ascan_length_s"] * 1e3
    print(json.dumps(geo, indent=2))
    print(f"amplitude: shape={amp.shape} dtype={amp.dtype} "
          f"min={amp.min()} max={amp.max()} mean={amp.mean():.1f} "
          f"nonzero%={100*np.mean(amp != 0):.2f}")

    env = envelope(amp)
    defects = load_defects(args.root, args.coupon)
    res_mm = geo["scan_res_m"] * 1e3
    nominal_offset = geo["u_offset_m"] * 1e3
    n_pos = geo["n_pos"]
    print(f"\ndefects: {len(defects)} rows | x[mm] "
          f"[{defects['x_init [mm]'].min()}, {defects['x_end [mm]'].max()}] "
          f"| types={defects['defect'].value_counts().to_dict()}")

    # --- offset sweep -------------------------------------------------------
    offsets = np.arange(0.0, 160.0 + 1e-9, 5.0)
    rows = []
    for off in offsets:
        lab = label_vector(defects, off, res_mm, n_pos)
        if lab.sum() == 0 or lab.sum() == n_pos:
            continue
        d = env[lab == 1]; c = env[lab == 0]
        sep = d.mean() - c.mean()
        a = auc(env, lab)
        rows.append((off, lab.sum(), sep, a))
    sweep = np.array([r[::2] for r in rows], dtype=float) if rows else np.zeros((0, 2))
    print("\noffset[mm] | #defect_pos | mean_sep | AUC")
    for off, nd, sep, a in rows:
        print(f"  {off:6.1f} | {nd:11d} | {sep:8.1f} | {a:.4f}")
    best = max(rows, key=lambda r: r[3]) if rows else (nominal_offset, 0, 0, 0)
    print(f"\nnominal offset (uCoordOffset) = {nominal_offset:.1f} mm")
    print(f"best AUC offset              = {best[0]:.1f} mm  (AUC={best[3]:.4f}, "
          f"#defect_pos={best[1]}, sep={best[2]:.1f})")

    # --- diagnostic plot ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.out.mkdir(parents=True, exist_ok=True)
        off = best[0]
        lab = label_vector(defects, off, res_mm, n_pos)
        x_mm = nominal_offset + np.arange(n_pos) * res_mm  # nominal axis for ref
        fig, ax = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 2]})
        ax[0].plot(x_mm, env, color="steelblue", lw=0.9)
        ax[0].fill_between(x_mm, 0, env.max(), where=lab == 1, color="red",
                           alpha=0.18, step="mid", label="defect x-range")
        ax[0].set_xlabel("weld position x [mm] (nominal offset)")
        ax[0].set_ylabel("max amplitude envelope")
        ax[0].set_title(f"{args.coupon} {nde.name} group{args.group} | "
                        f"offset={off:.0f}mm AUC={best[3]:.3f} "
                        f"(nominal {nominal_offset:.0f}mm)")
        ax[0].legend(loc="upper right")
        if len(rows):
            ax[1].plot([r[0] for r in rows], [r[3] for r in rows], "o-", color="darkgreen")
            ax[1].axvline(nominal_offset, color="k", ls="--", lw=0.8, label="nominal")
            ax[1].axvline(best[0], color="red", ls="--", lw=0.8, label="best")
            ax[1].set_xlabel("assumed offset [mm]")
            ax[1].set_ylabel("AUC (envelope vs defect)")
            ax[1].legend(loc="lower right")
        fig.tight_layout()
        png = args.out / f"paut_align_{args.coupon}_{args.angle}_g{args.group}.png"
        fig.savefig(png, dpi=110)
        print(f"\nsaved {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
