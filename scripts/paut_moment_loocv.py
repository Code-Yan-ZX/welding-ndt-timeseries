#!/usr/bin/env python
"""MOMENT (冻结时序基础模型) LOOCV 对照 (P1-④)。

用已缓存的 MOMENT 嵌入 (data/processed/paut/moment_feats.npz, 1024 维, 基于 max-envelope)
重建全位置嵌入, 按 5 折 LOOCV (PP3-PP7 轮流 test) 评估。探针: sklearn LogisticRegression
(balanced)。与 SSL/SSF/encoder 对照 -- 验证冻结预训练 TS 大模型是否迁移到 PAUT。

Usage: python scripts/paut_moment_loocv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


def main():
    processed = REPO / "data/processed/paut"
    mf = np.load(processed / "moment_feats.npz")
    sp = np.load(processed / "splits.npz", allow_pickle=True)
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy").astype(int)

    # 重建全位置嵌入: emb[global_idx] = MOMENT 嵌入
    N = len(coupons)
    emb = np.zeros((N, mf["Xtr"].shape[1]), dtype=np.float32)
    emb[sp["train"]] = mf["Xtr"]
    emb[sp["val"]] = mf["Xva"]
    emb[sp["test"]] = mf["Xte"]
    print(f"MOMENT 嵌入: {emb.shape} (1024 维, max-envelope)")

    rows = []
    for c in COUPONS:
        test_idx = np.nonzero(coupons == c)[0]
        rest = np.nonzero(coupons != c)[0]
        scaler = StandardScaler().fit(emb[rest])
        Xr = scaler.transform(emb[rest])
        Xt = scaler.transform(emb[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                 random_state=42)
        clf.fit(Xr, labels[rest])
        s = clf.predict_proba(Xt)[:, 1]
        y = labels[test_idx]
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
        rows.append({"test_coupon": c, "n_pos": int(y.sum()),
                     "auc": auc, "defect_rate": float(y.mean())})
        print(f"  {c}: AUC={auc:.3f} (pos={int(y.sum())}, 缺陷率 {y.mean():.3f})")

    aucs = np.array([r["auc"] for r in rows])
    non_pp4 = [r["auc"] for r in rows if r["test_coupon"] != "PP4"]
    summary = {"model": "moment_lr", "auc_mean": float(np.nanmean(aucs)),
               "auc_std": float(np.nanstd(aucs, ddof=1)),
               "auc_nonpp4": float(np.nanmean(non_pp4)),
               "per_fold": {r["test_coupon"]: r["auc"] for r in rows},
               "rows": rows}
    print(f"\nMOMENT LOOCV AUC: {summary['auc_mean']:.3f}±{summary['auc_std']:.3f} "
          f"(非PP4 {summary['auc_nonpp4']:.3f})")
    out = REPO / "experiments/results/paut_moment_loocv.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
