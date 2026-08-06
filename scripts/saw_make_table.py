#!/usr/bin/env python
"""Aggregate SAW experiment results into a comparison table.

Reads experiments/results/saw_*.json (excluding *_smoke*) and prints a
markdown table with per-seed and mean±std metrics.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"


def main():
    rows = defaultdict(list)  # model -> [result dict]
    for f in sorted(glob.glob(str(RES / "saw_*.json"))):
        if "_smoke" in f:
            continue
        d = json.load(open(f))
        rows[d["model"]].append(d)

    if not rows:
        sys.exit("no SAW results found")

    print("# SAW defect-detection results (window-level, test = PP7)\n")
    print("| model | seeds | test acc | test F1(bin) | test F1(macro) | test AUC | val F1(macro) | val AUC | params (trainable) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model, rs in sorted(rows.items()):
        acc = np.array([r["test_metrics"]["acc"] for r in rs])
        f1b = np.array([r["test_metrics"]["f1_bin"] for r in rs])
        f1m = np.array([r["test_metrics"]["f1_macro"] for r in rs])
        auc = np.array([r["test_metrics"].get("auc", float("nan")) for r in rs])
        vf1m = np.array([r["val_metrics"]["f1_macro"] for r in rs])
        vauc = np.array([r["val_metrics"].get("auc", float("nan")) for r in rs])
        nseeds = len(rs)
        pt = rs[0].get("n_params_trainable", 0)

        def ms(x):
            x = x[~np.isnan(x)]
            return f"{x.mean():.4f}±{x.std():.4f}" if len(x) > 1 else f"{x.mean():.4f}"

        pstr = f"{pt/1e6:.1f}M" if pt < 1e7 else f"{pt/1e6:.0f}M"
        print(f"| {model} | {nseeds} | {ms(acc)} | {ms(f1b)} | {ms(f1m)} | {ms(auc)} | {ms(vf1m)} | {ms(vauc)} | {pstr} |")

    # majority baseline
    mb = rs[0].get("majority_baseline_test", {})
    if mb:
        print(f"\nMajority baseline (test): acc {mb.get('acc',0):.4f} | "
              f"F1(macro) {mb.get('f1_macro',0):.4f}")


if __name__ == "__main__":
    main()
