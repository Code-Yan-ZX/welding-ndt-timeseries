#!/usr/bin/env python
"""PAUT temperature scaling 缺陷率校准 (P0-5)。

读取一个 LOOCV 结果 JSON (含每折 val_scores 与 scores/test_scores), 对每折:
  1. 在 val 上拟合温度 T (最小化 NLL): logit=log(s/(1-s)), p_T=sigmoid(logit/T)
  2. 把 T 应用到 test scores, 得到校准概率
  3. 对比校准前后的: ECE (期望校准误差)、Brier、缺陷率匹配 |mean(p)-defect_rate|、AUC (不变, 单调)

AUC 对温度单调不变, 所以本脚本的产出是「校准质量」(ECE/缺陷率), 验证
temperature scaling 是否让模型概率更可信、跨试件阈值更稳。

Usage:
  python scripts/paut_temperature_scaling.py experiments/results/paut_loocv_seed42.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]


def to_logits(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-logits / T))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_T(y_val: np.ndarray, s_val: np.ndarray) -> float:
    logits = to_logits(s_val)
    res = minimize_scalar(lambda T: nll(T, logits, y_val), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_T(T: float, s: np.ndarray) -> np.ndarray:
    logits = to_logits(s)
    return 1.0 / (1.0 + np.exp(-logits / T))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/paut_temperature_scaling.py <loocv_json> [<loocv_json2> ...]")
        sys.exit(1)
    processed = REPO / "data/processed/paut"
    labels_all = np.load(processed / "meta_label.npy").astype(int)
    coupons = np.load(processed / "meta_coupon.npy")
    from sklearn.model_selection import train_test_split
    COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]

    for jp in sys.argv[1:]:
        data = json.loads(Path(jp).read_text())
        out = [f"# Temperature Scaling 校准分析", "", f"源: `{jp}`", "",
               "每折: 在 val (其余4试件的15%) 上拟合温度 T (最小化NLL), 应用到 test (留出试件)。",
               "AUC 对温度单调不变 -> 关注 ECE / Brier / 缺陷率匹配 (均值p vs 真实缺陷率)。", ""]
        for mname, mdata in data["models"].items():
            out.append(f"## {mname}")
            out.append("| test试件 | T | AUC | ECE 前→后 | Brier 前→后 | 缺陷率 | 均值p 前→后 | 缺陷率误差 前→后 |")
            out.append("|---|---|---|---|---|---|---|---|")
            Ts, ece_b, ece_a, dr_err_b, dr_err_a = [], [], [], [], []
            for f in mdata["folds"]:
                c = f["test_coupon"]
                s_test = np.array(f["scores"])
                s_val = np.array(f.get("val_scores", []))
                # 重建 val y: 复现 fold 的 val 切分
                test_idx = np.nonzero(coupons == c)[0]
                rest = np.nonzero(coupons != c)[0]
                y_rest = labels_all[rest]
                stratify = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
                _, val_idx = train_test_split(rest, test_size=0.15, random_state=data["seed"],
                                              shuffle=True, stratify=stratify)
                y_val = labels_all[val_idx]
                y_test = labels_all[test_idx]
                if len(s_val) == 0 or len(np.unique(y_val)) < 2:
                    T = 1.0
                else:
                    T = fit_T(y_val, s_val)
                p_before = np.clip(s_test, 1e-6, 1 - 1e-6)
                p_after = apply_T(T, s_test)
                dr = float(y_test.mean())
                mp_b, mp_a = float(p_before.mean()), float(p_after.mean())
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float("nan")
                e0, e1 = ece(y_test, p_before), ece(y_test, p_after)
                b0, b1 = brier(y_test, p_before), brier(y_test, p_after)
                Ts.append(T); ece_b.append(e0); ece_a.append(e1)
                dr_err_b.append(abs(mp_b - dr)); dr_err_a.append(abs(mp_a - dr))
                out.append(f"| {c} | {T:.3f} | {auc:.3f} | {e0:.3f}→{e1:.3f} | "
                           f"{b0:.3f}→{b1:.3f} | {dr:.3f} | {mp_b:.3f}→{mp_a:.3f} | "
                           f"{abs(mp_b-dr):.3f}→{abs(mp_a-dr):.3f} |")
            out.append(f"\n**聚合 (mean±std)**: T={np.mean(Ts):.2f} | "
                       f"ECE {np.mean(ece_b):.3f}→{np.mean(ece_a):.3f} | "
                       f"缺陷率误差 {np.mean(dr_err_b):.3f}→{np.mean(dr_err_a):.3f}\n")
        out_path = Path(jp).with_name(Path(jp).stem + "_tempscale.md")
        out_path.write_text("\n".join(out), encoding="utf-8")
        print(f"-> {out_path}")
        print("\n".join(out[-6:]))


if __name__ == "__main__":
    main()
