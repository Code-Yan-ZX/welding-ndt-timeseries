#!/usr/bin/env python
"""Aggregate PAUT experiment results and cross-modality comparison vs SAW.

Reads experiments/results/paut_*.json (excluding *_smoke*) and, for the
comparison, the matching SAW (process-signal) results. Prints:
  1. PAUT defect-detection results (position-level, test = PP7)
  2. Cross-modality table: PAUT (phased-array NDT) vs SAW (current/voltage)
     for the same model families (classic RF / from-scratch encoder / MOMENT).
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


def load_rows(prefix: str):
    rows = defaultdict(list)
    for f in sorted(glob.glob(str(RES / f"{prefix}_*.json"))):
        if "_smoke" in f:
            continue
        d = json.load(open(f))
        rows[d["model"]].append(d)
    return rows


def ms(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "  -  "
    return f"{x.mean():.4f}±{x.std():.4f}" if len(x) > 1 else f"{x.mean():.4f}"


def table(rows, title):
    print(f"# {title}\n")
    print("| model | seeds | test acc | test F1(bin) | test F1(macro) | test AUC | val F1(macro) | val AUC | params (trainable) |")
    print("|---|---|---|---|---|---|---|---|---|")
    mb = {}
    for model, rs in sorted(rows.items()):
        acc = [r["test_metrics"]["acc"] for r in rs]
        f1b = [r["test_metrics"]["f1_bin"] for r in rs]
        f1m = [r["test_metrics"]["f1_macro"] for r in rs]
        auc = [r["test_metrics"].get("auc", float("nan")) for r in rs]
        vf1m = [r["val_metrics"]["f1_macro"] for r in rs]
        vauc = [r["val_metrics"].get("auc", float("nan")) for r in rs]
        pt = rs[0].get("n_params_trainable", 0)
        pstr = f"{pt/1e6:.1f}M" if pt < 1e7 else f"{pt/1e6:.0f}M"
        print(f"| {model} | {len(rs)} | {ms(acc)} | {ms(f1b)} | {ms(f1m)} | {ms(auc)} | {ms(vf1m)} | {ms(vauc)} | {pstr} |")
        mb = rs[0].get("majority_baseline_test", {}) or mb
    if mb:
        print(f"\nMajority baseline (test): acc {mb.get('acc',0):.4f} | "
              f"F1(macro) {mb.get('f1_macro',0):.4f}")


def pick(rows, model):
    rs = rows.get(model)
    return rs[0]["test_metrics"] if rs else None


def main():
    paut = load_rows("paut")
    if not paut:
        sys.exit("no PAUT results found - run the PAUT experiments first")
    table(paut, "PAUT (phased-array ultrasonic) defect-detection results "
                "(position-level, test = PP7)")

    saw = load_rows("saw")
    if not saw:
        print("\n(no SAW results found for comparison)")
        return
    print("\n\n# Cross-modality comparison: PAUT (NDT signal) vs SAW (process signal)\n")
    print("Both evaluate defect detection on test = PP7 (leave-coupon-out: "
          "train PP3/4/5 -> test PP7). PAUT input = per-beam A-scan (1ch, 512); "
          "SAW input = current/voltage window (4ch, 512).\n")
    print("| model family | modality | test acc | test F1(macro) | test AUC |")
    print("|---|---|---|---|---|")
    families = [("classic_rf", "classic RF"), ("encoder_only", "from-scratch encoder"),
                ("moment", "MOMENT (frozen probe)")]
    for key, label in families:
        for prefix, modality, rows in [("paut", "PAUT (NDT)", paut), ("saw", "SAW (process)", saw)]:
            rs = rows.get(key)
            if not rs:
                continue
            acc = np.mean([r["test_metrics"]["acc"] for r in rs])
            f1m = np.mean([r["test_metrics"]["f1_macro"] for r in rs])
            auc = np.mean([r["test_metrics"].get("auc", float("nan")) for r in rs])
            print(f"| {label} | {modality} | {acc:.4f} | {f1m:.4f} | "
                  f"{auc:.4f} |")
    # majority baselines
    pmb = paut[list(paut)[0]][0].get("majority_baseline_test", {}) if paut else {}
    smb = saw[list(saw)[0]][0].get("majority_baseline_test", {}) if saw else {}
    if pmb or smb:
        print(f"\nMajority baseline (test PP7): PAUT acc {pmb.get('acc',0):.4f} "
              f"F1(macro) {pmb.get('f1_macro',0):.4f} | SAW acc {smb.get('acc',0):.4f} "
              f"F1(macro) {smb.get('f1_macro',0):.4f}")


if __name__ == "__main__":
    main()
