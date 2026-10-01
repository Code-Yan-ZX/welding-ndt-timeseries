#!/usr/bin/env python
"""汇总 experiments/wrt-sam/*/eval_gdxray10.json 成一张对比表 (对齐论文 Table 1/3/4/5)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = {  # 论文 Table 1 / 3 / 4 / 5 (GDXray, 模型 -> R/P/AUC/IoU)
    "U-Net": (74.34, 75.97, None, 66.72),
    "Deeplabv3": (78.36, 79.52, None, 73.54),
    "PspNet": (77.11, 78.75, None, 71.23),
    "U-Net+CBMA": (77.33, 78.43, None, 71.12),
    "SAM-Adapter(Baseline)": (78.87, 78.39, 0.9596, 49.25),
    "SAM Adapter+FPG(top1)": (78.32, 81.87, 0.9859, 51.87),
    "SAM Adapter+MSPG": (79.29, 82.14, 0.9837, 50.88),
    "WRT-SAM(Ours)": (78.87, 84.04, 0.9746, 51.36),
}

def main():
    rows = []
    for d in sorted((REPO / "experiments/wrt-sam").glob("*/eval_gdxray10.json")):
        r = json.load(open(d))
        m = r["macro"]
        rows.append((d.parent.name, m["recall"] * 100, m["precision"] * 100,
                     m["auc"], m["iou"] * 100))
    print(f"{'run':28s} {'Recall':>7s} {'Prec':>7s} {'AUC':>7s} {'IoU':>7s}")
    for name, rc, pr, auc, iou in rows:
        print(f"{name:28s} {rc:7.2f} {pr:7.2f} {auc:7.4f} {iou:7.2f}")
    print("\n论文报告值 (macro 口径对照):")
    for k, (rc, pr, auc, iou) in PAPER.items():
        auc_s = f"{auc:.4f}" if auc else "   --  "
        print(f"{k:28s} {rc:7.2f} {pr:7.2f} {auc_s:>7s} {iou:7.2f}")


if __name__ == "__main__":
    main()
