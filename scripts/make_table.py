#!/usr/bin/env python
"""Aggregate experiments/results/*.json into a comparison table (md + csv)."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments/results"
OUT_DIR = RESULTS

METRICS = ["acc", "f1_bin", "f1_macro", "auc"]


def fmt_mean_std(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    rows = defaultdict(lambda: defaultdict(list))
    meta = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("comparison"):
            continue
        # skip smoke runs and lr-sweep probes (tags live in the filename)
        if "_smoke" in path.name or "_lr" in path.name:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("smoke"):
            continue
        model = d["model"]
        if d.get("llm"):
            from pathlib import PurePath
            model += f" ({PurePath(d['llm']).name})"
        meta[model] = {
            "params": d.get("n_params_trainable", 0),
            "wall": d.get("train_wall_s"),
            "llm": d.get("llm"),
        }
        for m in METRICS:
            v = d.get("test_metrics", {}).get(m)
            if v is not None:
                rows[model][m].append(float(v))

    if not rows:
        print("no results found")
        sys.exit(0)

    models = sorted(rows.keys())
    header = ["model", "seeds"] + [f"test_{m}" for m in METRICS] + ["params(M)", "wall(s)"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    csv_rows = [header]

    # majority baseline from any result file
    for path in sorted(RESULTS.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if not d.get("smoke") and d.get("majority_baseline_test"):
            mb = d["majority_baseline_test"]
            lines.append("| majority baseline | - | " +
                         " | ".join(f"{mb.get(m, float('nan')):.4f}" for m in METRICS) +
                         " | - | - |")
            csv_rows.append(["majority baseline", "-"] +
                            [f"{mb.get(m, float('nan')):.4f}" for m in METRICS] + ["-", "-"])
            break

    for model in models:
        seeds = len(next(iter(rows[model].values())))
        cells = [fmt_mean_std(rows[model][m]) for m in METRICS]
        p = meta[model]["params"] / 1e6 if meta[model]["params"] else 0
        w = meta[model]["wall"] if meta[model]["wall"] is not None else "-"
        lines.append(f"| {model} | {seeds} | " + " | ".join(cells) +
                     f" | {p:.1f} | {w} |")
        csv_rows.append([model, seeds] + cells + [f"{p:.1f}", str(w)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "comparison_table.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"wrote {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
