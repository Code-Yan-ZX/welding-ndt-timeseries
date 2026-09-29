#!/usr/bin/env python3
"""Export one sample per modality exactly as fed to the experiments, plus the
specimen photos and rendered B-scan images, for sharing.

Contents (data/exports/input_samples/):
  gmaw_sample_exp<E>_run<R>_row<I>.npz   one welding cycle, (2, 200) = (V, I)
  saw_sample_<bead>_win<K>.npz           one 4ch window, (4, 512) = (Ia, Ib, Va, Vb)
  photos/...                             PP3-PP7 specimen face photos (as-is)
  paut_bscan_images/...                  rendered B-scan / spectral PNGs for the
                                         two previously exported PAUT positions
  metadata.json                          units, sampling rates, labels, source IDs
  README.md                              format description

Label semantics differ per dataset and are recorded explicitly:
  GMAW (ASIMoW):  1 = good quality, 0 = poor quality, -1 unlabeled
  SAW  (PENELOPE): 1 = defect window, 0 = clean
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/exports/input_samples"

GMAW_CSV = REPO / "data/raw/processed_asimow_dataset.csv"
GMAW_DOI = "10.5281/zenodo.10017718"          # Metal Arc Welding (ASIMoW), v2
SAW_DOI = "10.5281/zenodo.15083865"           # Submerged Arc Welding (PENELOPE)

TEST_PAIRS = [(3, 3), (2, 10), (1, 24), (3, 24), (1, 32), (2, 1), (1, 10), (1, 16)]

SAW_GLOBAL_IDX = 156424   # PP7 crack window (defect_type 6 = Cracks)
PEN_ROOT = REPO / "data/raw/saw/ZENODO_Penelope"
PAUT_IMG = REPO / "data/processed/paut/images"
# previously exported PAUT positions (data/exports/paut_sample)
PAUT_REF = {"specimen-A": 270, "specimen-B": 2399}


def export_gmaw() -> dict | None:
    """One labeled test-split welding cycle straight from the source CSV."""
    if not GMAW_CSV.exists():
        print(f"[gmaw] {GMAW_CSV} missing -- run scripts/01_download_data.sh first")
        return None
    chunk_iter = pd.read_csv(GMAW_CSV, chunksize=200_000)
    found = None
    start = 0
    for chunk in chunk_iter:
        mask = np.zeros(len(chunk), dtype=bool)
        for e, r in TEST_PAIRS:
            mask |= (chunk["experiment"] == e) & (chunk["welding_run"] == r)
        labeled = mask & (chunk["labels"] == 0)        # 0 = poor quality
        if labeled.any():
            i = int(np.nonzero(labeled.to_numpy())[0][0])
            found = (start + i, chunk.iloc[i])
            break
        start += len(chunk)
    if found is None:
        print("[gmaw] no labeled poor-quality test-split cycle found")
        return None
    row_i, row = found
    v = np.asarray([row[f"V_{k}"] for k in range(200)], dtype=np.float32)
    i_ = np.asarray([row[f"I_{k}"] for k in range(200)], dtype=np.float32)
    wave = np.stack([v, i_], axis=0)                   # (2, 200), channel order (V, I)
    exp, run, lab = int(row["experiment"]), int(row["welding_run"]), int(row["labels"])
    sample_id = f"gmaw_sample_exp{exp}_run{run}_row{row_i}"
    np.savez_compressed(OUT / f"{sample_id}.npz",
                        waveform=wave,
                        experiment=np.int64(exp),
                        welding_run=np.int64(run),
                        source_row=np.int64(row_i),
                        label=np.int64(lab))
    print(f"[gmaw] {sample_id}  label={lab} (0=poor quality)  "
          f"V[{v.min():.1f},{v.max():.1f}]  I[{i_.min():.1f},{i_.max():.1f}]")
    return {
        "sample_id": sample_id,
        "modality": "GMAW process signals (arc voltage + welding current)",
        "waveform_shape": [2, 200],
        "channels": ["voltage", "current"],
        "units": ["V", "A"],
        "sampling_rate_hz": 100_000,
        "window_duration_s": 0.002,
        "label": lab,
        "label_semantics": "1 = good quality, 0 = poor quality, -1 = unlabeled",
        "source_record_id": {"experiment": exp, "welding_run": run,
                             "csv_row_index": row_i},
        "split": "test (T-joint)",
        "source_dataset": {"name": "Metal Arc Welding - Predictive Quality Arc "
                                   "Welding Dataset (ASIMoW), v2",
                           "doi": GMAW_DOI,
                           "url": f"https://zenodo.org/records/10017718",
                           "license": "CC-BY-4.0"},
    }


def export_saw() -> dict:
    """One defect window, exactly as stored in data/processed/saw/waves.npy."""
    waves = np.load(REPO / "data/processed/saw/waves.npy", mmap_mode="r")
    bead = np.load(REPO / "data/processed/saw/meta_bead.npy")
    coup = np.load(REPO / "data/processed/saw/meta_coupon.npy")
    lab = np.load(REPO / "data/processed/saw/meta_label.npy")
    typ = np.load(REPO / "data/processed/saw/meta_defect_type.npy")
    g = SAW_GLOBAL_IDX
    wave = np.asarray(waves[g], dtype=np.float32)      # (4, 512)
    b = str(bead[g])
    first = int(np.nonzero(bead == b)[0][0])
    ordinal = g - first                                 # window index within bead
    start_sample = ordinal * 256                        # stride 256
    codes = {1: "Porosity", 2: "Lack of fusion", 3: "Slag inclusion",
             4: "Metallic inclusion", 5: "Projections", 6: "Cracks"}
    sample_id = f"saw_sample_{b}_win{ordinal}"
    np.savez_compressed(OUT / f"{sample_id}.npz",
                        waveform=wave,
                        coupon=np.array(str(coup[g])),
                        bead=np.array(b),
                        window_ordinal=np.int64(ordinal),
                        window_start_sample=np.int64(start_sample),
                        global_window_index=np.int64(g),
                        label=np.int64(lab[g]),
                        defect_type_code=np.int64(typ[g]))
    print(f"[saw] {sample_id}  label={int(lab[g])}  type={codes[int(typ[g])]}")
    return {
        "sample_id": sample_id,
        "modality": "SAW process signals (two-wire submerged arc welding, "
                    "dual-electrode current/voltage)",
        "waveform_shape": [4, 512],
        "channels": ["current_a", "current_b", "voltage_a", "voltage_b"],
        "units": ["A", "A", "V", "V"],
        "sampling_rate_hz": 5_000,
        "window_duration_s": 0.1024,
        "window_stride_samples": 256,
        "label": int(lab[g]),
        "label_semantics": "1 = defect window (overlaps a defect sample range "
                           "in defects_xlocation.xlsx), 0 = clean",
        "defect_type": codes[int(typ[g])],
        "source_record_id": {"coupon": str(coup[g]), "bead": b,
                             "window_ordinal": ordinal,
                             "window_start_sample_data1": start_sample,
                             "global_window_index": g},
        "split": "test coupon",
        "source_dataset": {"name": "Submerged Arc Welding dataset (PENELOPE)",
                           "doi": SAW_DOI,
                           "url": "https://zenodo.org/records/15083865",
                           "license": "see Zenodo record"},
    }


def copy_photos() -> list[str]:
    """Copy the PP3-PP7 specimen face photos as-is (original filenames)."""
    dst = OUT / "photos"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for c in ["PP3", "PP4", "PP5", "PP6", "PP7"]:
        src = PEN_ROOT / c / "3. pics"
        if not src.is_dir():
            continue
        for p in sorted(src.iterdir()):
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    print(f"[photos] copied {len(copied)} specimen photos")
    return copied


def copy_bscan_images() -> list[str]:
    """Rendered B-scan / spectral PNGs for the two exported PAUT positions."""
    dst = OUT / "paut_bscan_images"
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name, g in PAUT_REF.items():
        for kind in ["bscan", "spec"]:
            src = PAUT_IMG / f"{g:05d}_{kind}.png"
            tgt = dst / f"{name}_pos{g}_{kind}.png"
            shutil.copy2(src, tgt)
            copied.append(tgt.name)
    print(f"[bscan] copied {len(copied)} rendered images")
    return copied


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gmaw = export_gmaw()
    saw = export_saw()
    photos = copy_photos()
    bscans = copy_bscan_images()

    meta = {
        "description": "One sample per modality exactly as fed to the "
                       "experiments, plus specimen photos and rendered "
                       "B-scan images",
        "datasets": {
            "gmaw": {
                "record": gmaw,
                "note": "no specimen photos in this dataset; samples are "
                        "welding cycles, not spatial scans",
            },
            "saw": {
                "record": saw,
                "specimen_photos": photos,
                "photo_note": "weld face photos taken after welding "
                              "(final face A/B), original dataset filenames",
            },
            "paut": {
                "note": "B-scan waveform volumes (49 beams x 512 samples) for "
                        "two positions were exported previously under "
                        "data/exports/paut_sample/; the rendered images here "
                        "correspond to the same two positions",
                "rendered_images": bscans,
                "image_note": "bscan = grayscale (49,512) envelope, 2-98 "
                              "percentile normalized; spec = rfft log-magnitude "
                              "turbo colormap",
                "cross_reference": {
                    "specimen-A": "coupon PP3, scan position 270 (x=350 mm), "
                                  "defect = Cracks",
                    "specimen-B": "coupon PP7, scan position 0 (x=80 mm), clean",
                },
            },
        },
        "label_semantics_warning": "GMAW labels are QUALITY labels (1=good, "
                                   "0=poor); SAW/PAUT labels are DEFECT labels "
                                   "(1=defect, 0=clean) -- do not mix",
    }
    with open(OUT / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
