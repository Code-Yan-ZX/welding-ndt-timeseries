#!/usr/bin/env python
"""PAUT 无标注异常检测 baseline (P1-③) — McKnight 式 Weibull。

用 SSL 预训练的掩码自编码器计算每个位置的重建误差作为异常分。McKnight 式:
在每个 LOOCV 折的 **clean (label=0) 训练位置** 上拟合 Weibull 分布, test 位置的异常分
= 1 - Weibull_CDF(误差) (越偏离 clean 尾部越可能为缺陷)。全程不用缺陷标签训练
(仅用 clean 拟合分布, 标签只用于评估 AUC)。也报告纯重建误差 AUC (无 Weibull)。

Usage:
  python scripts/paut_anomaly.py
  python scripts/paut_anomaly.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import weibull_min  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from wndt.models.ssl_ae import MaskedAE  # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


@torch.no_grad()
def recon_errors(model, ascans, indices, ts_mean, ts_std, device):
    """每个位置的重建 MSE (全图, 不掩码 -- 推理时用完整输入测重建质量)。"""
    model.eval()
    errs = np.empty(len(indices), dtype=np.float32)
    for pi, gi in enumerate(indices):
        x = np.array(ascans[gi], dtype=np.float32)  # (49,512)
        x = (x - ts_mean) / ts_std
        xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,49,512)
        recon, target, _ = model(xt)
        errs[pi] = float(((recon - target.unsqueeze(1)) ** 2).mean().item())
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-ckpt", type=Path,
                    default=REPO / "experiments/runs/ssl_ae/encoder.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    processed = REPO / "data/processed/paut"
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy").astype(int)
    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    with open(processed / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 SSL AE (编码器权重 + 重建解码器)
    ckpt = torch.load(args.ssl_ckpt, map_location=device)
    ae = MaskedAE(d_model=ckpt["d_model"], mask_ratio=0.0, noise_std=0.0).to(device)
    ae.encoder.load_state_dict(ckpt["encoder_state"])

    # 全位置重建误差 (一次算完)
    all_idx = np.arange(len(coupons))
    print("计算全位置重建误差 ...")
    errs = recon_errors(ae, ascans, all_idx, ts_mean, ts_std, device)

    rows = []
    for c in COUPONS:
        test_idx = np.nonzero(coupons == c)[0]
        rest = np.nonzero(coupons != c)[0]
        y_rest = labels[rest]
        stratify = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
        _, val_idx = train_test_split(rest, test_size=0.15, random_state=args.seed,
                                      shuffle=True, stratify=stratify)
        train_idx = np.setdiff1d(rest, val_idx)
        y_test = labels[test_idx]
        # clean 训练位置的误差 -> 拟合 Weibull
        clean_train = train_idx[labels[train_idx] == 0]
        e_clean = errs[clean_train]
        # Weibull 拟合 (weibull_min: c=shape, loc, scale)
        try:
            c_shape, loc, scale = weibull_min.fit(e_clean, floc=0.0)
            wb_cdf = weibull_min.cdf(errs[test_idx], c_shape, loc=loc, scale=scale)
            anom_wb = 1.0 - wb_cdf  # 越大越异常
        except Exception:
            anom_wb = errs[test_idx]
            c_shape = float("nan")
        anom_raw = errs[test_idx]  # 纯重建误差
        auc_wb = float(roc_auc_score(y_test, anom_wb)) if len(np.unique(y_test)) == 2 else float("nan")
        auc_raw = float(roc_auc_score(y_test, anom_raw)) if len(np.unique(y_test)) == 2 else float("nan")
        # 缺陷率校准: 预测缺陷率 = mean(anom > 0.5)? 用 Weibull 1-CDF 阈值 0.5
        pred_rate = float((anom_wb > 0.5).mean())
        rows.append({"test_coupon": c, "n_pos": int(y_test.sum()),
                     "auc_weibull": auc_wb, "auc_raw": auc_raw,
                     "weibull_shape": float(c_shape),
                     "defect_rate": float(y_test.mean()), "pred_rate": pred_rate})
        print(f"  {c}: AUC(Weibull)={auc_wb:.3f} AUC(raw)={auc_raw:.3f} "
              f"shape={c_shape:.2f} | 缺陷率 {y_test.mean():.3f} 预测 {pred_rate:.3f}")

    aucs_wb = np.array([r["auc_weibull"] for r in rows])
    aucs_raw = np.array([r["auc_raw"] for r in rows])
    non_pp4 = [r for r in rows if r["test_coupon"] != "PP4"]
    summary = {
        "auc_weibull_mean": float(np.nanmean(aucs_wb)),
        "auc_weibull_std": float(np.nanstd(aucs_wb, ddof=1)),
        "auc_raw_mean": float(np.nanmean(aucs_raw)),
        "auc_weibull_nonpp4": float(np.nanmean([r["auc_weibull"] for r in non_pp4])),
        "auc_raw_nonpp4": float(np.nanmean([r["auc_raw"] for r in non_pp4])),
        "rows": rows,
    }
    print(f"\n异常检测 AUC: Weibull {summary['auc_weibull_mean']:.3f}±{summary['auc_weibull_std']:.3f} "
          f"(非PP4 {summary['auc_weibull_nonpp4']:.3f}) | raw {summary['auc_raw_mean']:.3f} "
          f"(非PP4 {summary['auc_raw_nonpp4']:.3f})")
    out = REPO / "experiments/results/paut_anomaly_seed42.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
