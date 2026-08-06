#!/usr/bin/env python3
"""Inspect the Submerged Arc Welding (SAW) Zenodo dataset structure.

Run AFTER extracting ZENODO_Penelope_vs2.zip:
  python scripts/saw_inspect.py --root data/raw/saw/Database

Walks the extracted tree and prints:
  - file-type inventory (counts by extension)
  - per-HDF5 structure: groups/datasets, shapes, dtypes, attributes
  - per-XLSX (defects locations): sheet names + head rows
  - sample README.txt contents
  - NDE / PAUT file list

This is exploratory; output informs the SAW data pipeline design.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--max-hdf5", type=int, default=3, help="inspect first N HDF5 files in full")
    ap.add_argument("--max-xlsx", type=int, default=5)
    ap.add_argument("--max-readme", type=int, default=2)
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        sys.exit(f"root not found: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"=== {len(files)} files under {root} ===")
    ext = Counter(p.suffix.lower() for p in files)
    for k, v in ext.most_common():
        print(f"  {k or '(no ext)':<12} {v}")

    hdf5_files = [p for p in files if p.suffix.lower() in {".h5", ".hdf5"}]
    xlsx_files = [p for p in files if p.suffix.lower() == ".xlsx"]
    nde_files = [p for p in files if p.suffix.lower() == ".nde"]
    txt_files = [p for p in files if p.suffix.lower() == ".txt"]

    print(f"\n=== HDF5 files: {len(hdf5_files)} ===")
    for p in hdf5_files[: args.max_hdf5]:
        print(f"\n--- {p.relative_to(root)} ---")
        try:
            import h5py
            with h5py.File(p, "r") as fh:
                _walk_h5(fh, depth=0)
                print("  attrs:", {k: _safe_attr(v) for k, v in fh.attrs.items()})
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR reading: {exc}")

    print(f"\n=== XLSX files: {len(xlsx_files)} ===")
    for p in xlsx_files[: args.max_xlsx]:
        print(f"\n--- {p.relative_to(root)} ---")
        try:
            import pandas as pd
            xls = pd.ExcelFile(p)
            print("  sheets:", xls.sheet_names)
            for sh in xls.sheet_names[:3]:
                df = pd.read_excel(p, sheet_name=sh)
                print(f"  [{sh}] shape={df.shape} cols={list(df.columns)[:12]}")
                print(df.head(5).to_string(max_cols=12).replace("\n", "\n  "))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR reading: {exc}")

    print(f"\n=== NDE files: {len(nde_files)} ===")
    for p in nde_files[:10]:
        print(f"  {p.relative_to(root)}  ({p.stat().st_size} B)")

    print(f"\n=== sample README/TXT files ===")
    for p in txt_files[: args.max_readme]:
        print(f"\n--- {p.relative_to(root)} ---")
        try:
            txt = p.read_text(errors="replace")[:1500]
            print(txt)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")

    # top-level layout
    print("\n=== top-level dirs ===")
    for d in sorted(p for p in root.iterdir() if p.is_dir())[:20]:
        print(f"  {d.name}/")
        for sub in sorted(p for p in d.iterdir() if p.is_dir())[:10]:
            print(f"    {sub.name}/")


def _walk_h5(group, depth: int, max_depth: int = 4) -> None:
    pad = "  " * (depth + 1)
    for key in list(group.keys())[:40]:
        item = group[key]
        if isinstance(item, group.__class__) or hasattr(item, "keys"):
            print(f"{pad}{key}/ (group)")
            if depth < max_depth:
                _walk_h5(item, depth + 1, max_depth)
        else:
            print(f"{pad}{key}: shape={getattr(item, 'shape', None)} "
                  f"dtype={getattr(item, 'dtype', None)}")


def _safe_attr(v):
    try:
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return list(v)[:8]
        return v
    except Exception:  # noqa: BLE001
        return repr(v)


if __name__ == "__main__":
    main()
