#!/usr/bin/env python3
"""M0-2C PAUT 回测（灾难性遗忘检查）：P vs P→E 在 PP3–PP7 规范 LOOCV。

**规范协议**（与 P4a baseline 0.571/0.579±0.007 完全一致，见
``paut_p4_ssl_variants.py``）：
- ``P``  : 原 ``experiments/runs/ssl_ae/encoder.pt`` 在当前代码下重新跑一次
  规范头（不只引用历史 0.579）；
- ``P→E``: 加载 M0-2C ECT 续训 encoder（2ch），第一层折回单通道
  ``w_single = w_ect[:,0:1] + w_ect[:,1:2]``，其余权重原样加载；
- LOOCV：test = 1 完整 coupon；其余 4 个 coupons 按标签 **85/15 分层位置级
  train/val**（``fold_splits``）；归一化（per-timestep）只由 train 位置计算；
- 冻结 encoder + ``SSLClassifier`` 规范头（lr 1e-3 / 80ep / batch 128 /
  class-balanced 加权采样 / val-AUC 早停，``train_fold``）；
- 头 seed = model_seed（P 与 P→E 配对比较用相同 head seed）；split_seed=42
  固定（fold 划分不随 model_seed 变）。
- 主指标 = **非PP4 逐折均值**（PP3/PP5/PP6/PP7）；同时报告全 5 折均值。

聚合时比较 P→E − P 逐 seed 差值（>= −0.01 为保持，否则 = 灾难性遗忘）。

Usage:
  python scripts/m0_2c_paut_retention.py --cond P  --seed 42
  python scripts/m0_2c_paut_retention.py --cond PE --seed 42 --steps 10000
  python scripts/m0_2c_paut_retention.py --cond P --seed 42 --smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from wndt.data.paut_dataset import PAUTSeriesDataset  # noqa: E402
from wndt.models.ssl_ae import MAEEncoder, SSLClassifier  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

from m0_2c_ect_pretrain import (  # noqa: E402
    DEFAULT_CONFIG, TRANSFER_SOURCE, ckpt_path, fold_back_first_layer,
    git_commit,
)
from paut_p4_ssl_variants import (  # noqa: E402
    COUPONS, NP4, fold_norm, fold_splits, scores_of, train_fold,
)

RESULTS_DIR = REPO / "experiments" / "results"
CONDS = ("P", "PE")
PROCESSED = REPO / "data" / "processed" / "paut"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


def load_retention_encoder(cond: str, model_seed: int, steps: int,
                           tag: str | None, cfg, device) -> MAEEncoder:
    """1ch MAEEncoder（PAUT 输入 (B,1,49,512)）。

    - P  : 原 ssl_ae/encoder.pt 直接加载（第一层保持 1ch）；
    - PE : M0-2C ECT 2ch encoder 第一层折回 ``w[:,0:1]+w[:,1:2]``。
    """
    enc = MAEEncoder(d_model=int(cfg.model.d_model), in_channels=1).to(device)
    if cond == "P":
        assert TRANSFER_SOURCE.exists(), f"迁移源不存在: {TRANSFER_SOURCE}"
        sd = torch.load(TRANSFER_SOURCE, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd["encoder_state"])
        print(f"[retention P s{model_seed}] 加载 {TRANSFER_SOURCE}")
        return enc
    p = ckpt_path("PE", model_seed, steps, tag)
    assert p.exists(), f"checkpoint 不存在: {p}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    enc2 = MAEEncoder(d_model=int(cfg.model.d_model), in_channels=2)
    enc2.load_state_dict(ck["state_dict"]["encoder"])
    sd2 = enc2.state_dict()
    sd1 = enc.state_dict()
    # 第一层折回单通道，其余原样
    sd1["conv.0.weight"] = fold_back_first_layer(sd2["conv.0.weight"])
    for k in sd1:
        if k != "conv.0.weight":
            sd1[k] = sd2[k]
    enc.load_state_dict(sd1)
    print(f"[retention PE s{model_seed}] 加载 {p}（第一层折回单通道）")
    return enc


def run_retention(cond: str, model_seed: int, steps: int, tag: str | None,
                  cfg, device, smoke: bool) -> dict:
    ascans = np.load(PROCESSED / "ascans.npy", mmap_mode="r")
    coupons = np.load(PROCESSED / "meta_coupon.npy", allow_pickle=True)
    labels = np.load(PROCESSED / "meta_label.npy").astype(np.int64)
    head_epochs = 1 if smoke else 80
    head_lr = 1e-3
    # 冻结 encoder（构建后 eval；head 训练时模型置 train 会更新 BN running
    # stats —— 与 P4a baseline 协议一致）
    enc = load_retention_encoder(cond, model_seed, steps, tag, cfg, device)
    for prm in enc.parameters():
        prm.requires_grad = False
    enc.eval()

    folds = []
    all_scores, all_labels = [], []
    for tc in COUPONS:
        train_idx, val_idx, test_idx = fold_splits(coupons, labels, tc,
                                                   val_frac=0.15, seed=42)
        ts_mean, ts_std = fold_norm(ascans, train_idx)
        train_ds = PAUTSeriesDataset(PROCESSED, train_idx, beam="bscan",
                                     norm_mode="per_timestep",
                                     ts_mean=ts_mean, ts_std=ts_std)
        val_ds = PAUTSeriesDataset(PROCESSED, val_idx, beam="bscan",
                                   norm_mode="per_timestep",
                                   ts_mean=ts_mean, ts_std=ts_std)
        test_ds = PAUTSeriesDataset(PROCESSED, test_idx, beam="bscan",
                                    norm_mode="per_timestep",
                                    ts_mean=ts_mean, ts_std=ts_std)
        set_seed(model_seed)                 # head 初始化只由 model_seed 决定
        model = SSLClassifier(enc, d_model=int(cfg.model.d_model),
                              n_classes=2, freeze_encoder=True).to(device)
        t0 = time.time()
        fit = train_fold(model, train_ds, val_ds, device, epochs=head_epochs,
                         lr=head_lr, wd=1e-4, batch_size=128, seed=model_seed)
        wall = round(time.time() - t0, 1)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                                 num_workers=4, pin_memory=True)
        scores = scores_of(model, test_loader, device)
        yte = labels[test_idx]
        auc = float(roc_auc_score(yte, scores))
        pr = float(average_precision_score(yte, scores))
        folds.append({
            "test_coupon": tc, "n_test": int(len(test_idx)),
            "n_pos": int(yte.sum()), "defect_rate": round(float(yte.mean()), 4),
            "val_auc": round(float(fit.get("best_val_auc", 0.0)), 4),
            "test_auc": round(auc, 4), "pr_auc": round(pr, 4),
            "epochs_run": fit.get("epochs_run"), "wall_s": wall,
        })
        all_scores.append(scores)
        all_labels.append(yte)
        print(f"  fold {tc} | n={len(test_idx)} pos={int(yte.sum())} | "
              f"val_auc={fit.get('best_val_auc', 0.0):.4f} "
              f"test_auc={auc:.4f} pr_auc={pr:.4f} "
              f"ep={fit.get('epochs_run')} ({wall}s)")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    aucs = [f["test_auc"] for f in folds]
    np4 = [f["test_auc"] for f in folds if f["test_coupon"] in NP4]
    return {
        "exp": "m0_2c_paut_retention", "cond": cond, "model_seed": model_seed,
        "split_seed": 42, "smoke": bool(smoke),
        "protocol": "P4a 规范 LOOCV：test=1 coupon；其余 4 coupons 85/15 分层"
                    "位置级 val；per-timestep 归一化（train 位置）；冻结 encoder "
                    "+ SSLClassifier 规范头（lr 1e-3/80ep/batch128/加权/val-AUC 早停）",
        "ckpt": (str(TRANSFER_SOURCE) if cond == "P"
                 else str(ckpt_path("PE", model_seed, steps, tag))),
        "folds": folds,
        "all_folds_mean_auc": round(float(np.mean(aucs)), 4),
        "all_folds_std_auc": round(float(np.std(aucs)), 4),
        "nonPP4_mean_auc": round(float(np.mean(np4)), 4) if np4 else None,
        "nonPP4_std_auc": round(float(np.std(np4)), 4) if np4 else None,
        "pooled_auc": round(float(roc_auc_score(all_labels, all_scores)), 4),
        "pp4_auc": round(float([f["test_auc"] for f in folds
                                if f["test_coupon"] == "PP4"][0]), 4),
        "head": {"lr": head_lr, "epochs": head_epochs, "batch_size": 128,
                 "class_balance": True},
        "code_commit": git_commit(), "code_dirty": git_dirty(),
    }


def per_exp_path(cond: str, model_seed: int, steps: int, tag: str | None,
                 smoke: bool) -> Path:
    tag_s = f"_{tag}" if tag else ""
    suffix = ("_smoke" if (smoke and tag != "smoke") else "")
    return RESULTS_DIR / f"m0_2c_paut_retention_{cond}_seed{model_seed}_s{steps}{tag_s}{suffix}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--seed", type=int, default=42, dest="model_seed")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps is None:
        args.steps = 100 if args.smoke else int(cfg.pretrain.steps)
    if args.smoke and args.tag is None:
        args.tag = "smoke"          # 与 pretrain --smoke 的 tag 一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_retention(args.cond, args.model_seed, args.steps, args.tag,
                        cfg, device, args.smoke)
    out = per_exp_path(args.cond, args.model_seed, args.steps, args.tag, args.smoke)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
