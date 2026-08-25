#!/usr/bin/env python3
"""M0-3 Protocol V2 严格跨试件 LOOCV 评估（P-long vs W→P 冻结表征）。

协议（Protocol V2，与 ``m0_2b_loocv.py`` 一致）：
- outer test ：一个完整 coupon；
- inner val   ：剩余 coupons 中一个完整 coupon；
- train       ：其余完整 coupons；
- 归一化只由 train coupons 计算（per-timestep，无泄漏）；
- PAUT SSL、分类头训练只用本折 train coupons；
- validation coupon 只用于模型选择（val AUC 早停）；
- test coupon 在 SSL / 归一化 / 头训练 / 模型选择全程不可见。
- 不使用历史 P4a 随机位置级 validation 作为正式协议。

条件：
- ``P-long``：加载本折 ``plong_fold{fold}...`` checkpoint（encoder）
- ``WP``    ：加载本折 ``wp_fold{fold}...`` checkpoint（encoder）

下游头（规范协议）：冻结 encoder + ``SSLClassifier`` 规范头（lr 1e-3 /
80 ep / batch 128 / class-balanced / val coupon 驱动早停，``train_fold``）。

指标：主指标 = **非PP4 逐折 mean ROC-AUC**（PP3/PP5/PP6/PP7）；同时报告逐折、
pooled、PR-AUC、balanced accuracy。结果按 (cond, seed) 写
``experiments/results/m0_3_loocv_{cond}_seed{seed}_full.json``。

Usage:
  python scripts/m0_3_loocv.py --cond P-long --seed 42
  python scripts/m0_3_loocv.py --cond WP --seed 42 --tag pilot --steps 2000 2000
  python scripts/m0_3_loocv.py --cond P-long --seed 42 --smoke
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)
from torch.utils.data import DataLoader  # noqa: E402

from wndt.data.paut_dataset import PAUTSeriesDataset  # noqa: E402
from wndt.data.ultrasound_pretrain import (  # noqa: E402
    COUPONS, NP4, load_paut, paut_fold_split,
)
from wndt.models.ssl_ae import MAEEncoder, SSLClassifier  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

from m0_3_weld_ut_pretrain import (  # noqa: E402
    DEFAULT_CONFIG, cond_ckpt_path, git_commit,
)
from paut_p4_ssl_variants import (  # noqa: E402
    fold_norm, fold_splits, scores_of, train_fold,
)

RESULTS_DIR = REPO / "experiments" / "results"
CONDS = ("P-long", "WP")
PROCESSED = REPO / "data" / "processed" / "paut"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


def load_frozen_encoder(cond: str, fold: str, model_seed: int, ext_steps: int,
                        tgt_steps: int, tag: str | None, cfg,
                        device) -> MAEEncoder:
    """加载条件 checkpoint 的 encoder（1ch，冻结）。"""
    p = cond_ckpt_path(cond, fold, model_seed, ext_steps, tgt_steps, tag)
    assert p.exists(), f"checkpoint 不存在: {p}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    enc = MAEEncoder(d_model=int(cfg.model.d_model),
                     in_channels=int(cfg.model.in_channels)).to(device)
    enc.load_state_dict(ck["state_dict"]["encoder"])
    for prm in enc.parameters():
        prm.requires_grad = False
    enc.eval()
    print(f"[{cond}/{fold} s{model_seed}] 加载 {p}，冻结 encoder")
    return enc


def run_loocv(cond: str, model_seed: int, ext_steps: int, tgt_steps: int,
              tag: str | None, cfg, device, smoke: bool) -> dict:
    ascans = np.load(PROCESSED / "ascans.npy", mmap_mode="r")
    coupons = np.load(PROCESSED / "meta_coupon.npy", allow_pickle=True)
    labels = np.load(PROCESSED / "meta_label.npy").astype(np.int64)
    split_seed = int(cfg.pretrain.split_seed)
    head_epochs = 1 if smoke else int(cfg.head.epochs)
    head_lr = float(cfg.head.lr)

    folds = []
    all_scores, all_labels = [], []
    for tc in COUPONS:
        tr_idx, va_idx, te_idx, train_coupons, val_coupon = paut_fold_split(
            coupons, tc, split_seed)
        ts_mean, ts_std = fold_norm(ascans, tr_idx)
        enc = load_frozen_encoder(cond, tc, model_seed, ext_steps, tgt_steps,
                                  tag, cfg, device)

        train_ds = PAUTSeriesDataset(PROCESSED, tr_idx, beam="bscan",
                                     norm_mode="per_timestep",
                                     ts_mean=ts_mean, ts_std=ts_std)
        val_ds = PAUTSeriesDataset(PROCESSED, va_idx, beam="bscan",
                                   norm_mode="per_timestep",
                                   ts_mean=ts_mean, ts_std=ts_std)
        test_ds = PAUTSeriesDataset(PROCESSED, te_idx, beam="bscan",
                                    norm_mode="per_timestep",
                                    ts_mean=ts_mean, ts_std=ts_std)
        set_seed(model_seed)
        model = SSLClassifier(enc, d_model=int(cfg.model.d_model), n_classes=2,
                              freeze_encoder=True).to(device)
        t0 = time.time()
        fit = train_fold(model, train_ds, val_ds, device, epochs=head_epochs,
                         lr=head_lr, wd=1e-4, batch_size=int(cfg.head.batch_size),
                         seed=model_seed)
        wall = round(time.time() - t0, 1)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                                 num_workers=4, pin_memory=True)
        scores = scores_of(model, test_loader, device)
        yte = labels[te_idx]
        # bAcc：val coupon 上选最优阈值（PAUT 协议一致）
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
        val_scores = scores_of(model, val_loader, device)
        yva = labels[va_idx]
        from m0_2c_ect_probe import best_threshold
        thr = best_threshold(yva, val_scores)
        auc = float(roc_auc_score(yte, scores))
        pr = float(average_precision_score(yte, scores))
        bacc = float(balanced_accuracy_score(yte, (scores > thr).astype(int)))
        folds.append({
            "test_coupon": tc, "train_coupons": train_coupons,
            "val_coupon": val_coupon, "n_test": int(len(te_idx)),
            "n_pos": int(yte.sum()), "defect_rate": round(float(yte.mean()), 4),
            "val_auc": round(float(fit.get("best_val_auc", 0.0)), 4),
            "thr": round(thr, 4),
            "test_auc": round(auc, 4), "pr_auc": round(pr, 4),
            "balanced_acc": round(bacc, 4), "epochs_run": fit.get("epochs_run"),
            "wall_s": wall,
        })
        all_scores.append(scores)
        all_labels.append(yte)
        print(f"  fold {tc} train={train_coupons} val={val_coupon} | "
              f"n={len(te_idx)} pos={int(yte.sum())} | val_auc={fit.get('best_val_auc', 0.0):.4f} "
              f"test_auc={auc:.4f} pr_auc={pr:.4f} bacc={bacc:.4f} ({wall}s)")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    aucs = [f["test_auc"] for f in folds]
    np4 = [f["test_auc"] for f in folds if f["test_coupon"] in NP4]
    np4_pr = [f["pr_auc"] for f in folds if f["test_coupon"] in NP4]
    np4_bacc = [f["balanced_acc"] for f in folds if f["test_coupon"] in NP4]
    return {
        "exp": "m0_3_loocv", "cond": cond, "model_seed": model_seed,
        "split_seed": split_seed, "data_seed": int(cfg.pretrain.data_seed),
        "run_type": "smoke" if smoke else "full",
        "protocol": "Protocol V2：test=1 coupon；inner val=1 coupon；train=其余；"
                    "归一化/SSL/头训练只读 train coupons；val 只做模型选择；"
                    "test 全程不可见",
        "ext_steps": ext_steps, "tgt_steps": tgt_steps,
        "total_ssl_steps": ext_steps + tgt_steps,
        "head": {"lr": head_lr, "epochs": head_epochs,
                 "batch_size": int(cfg.head.batch_size),
                 "class_balance": bool(cfg.head.class_balance)},
        "folds": folds,
        "all_folds_mean_auc": round(float(np.mean(aucs)), 4),
        "all_folds_std_auc": round(float(np.std(aucs)), 4),
        "nonPP4_mean_auc": round(float(np.mean(np4)), 4) if np4 else None,
        "nonPP4_std_auc": round(float(np.std(np4)), 4) if np4 else None,
        "nonPP4_mean_pr_auc": round(float(np.mean(np4_pr)), 4) if np4_pr else None,
        "nonPP4_mean_balanced_acc": round(float(np.mean(np4_bacc)), 4) if np4_bacc else None,
        "pooled_auc": round(float(roc_auc_score(all_labels, all_scores)), 4),
        "pp4_auc": round(float([f["test_auc"] for f in folds
                                if f["test_coupon"] == "PP4"][0]), 4),
        "code_commit": git_commit(), "code_dirty": git_dirty(),
    }


def per_exp_path(cond: str, model_seed: int, ext_steps: int, tgt_steps: int,
                 tag: str | None, smoke: bool) -> Path:
    tag_s = f"_{tag}" if tag else ""
    suffix = ("_smoke" if (smoke and tag != "smoke") else "")
    return RESULTS_DIR / f"m0_3_loocv_{cond}_seed{model_seed}_e{ext_steps}_t{tgt_steps}{tag_s}{suffix}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--seed", type=int, default=42, dest="model_seed")
    ap.add_argument("--ext-steps", type=int, default=None)
    ap.add_argument("--tgt-steps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="smoke 用：ext/tgt 各取该值")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        args.ext_steps = args.steps or 20
        args.tgt_steps = args.steps or 20
        if args.tag is None:
            args.tag = "smoke"
    if args.ext_steps is None or args.tgt_steps is None:
        args.ext_steps = int(cfg.pretrain.pilot_external_steps)
        args.tgt_steps = int(cfg.pretrain.pilot_target_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-3 loocv[{args.cond}] seed={args.model_seed} "
          f"ext={args.ext_steps} tgt={args.tgt_steps} device={device}")
    res = run_loocv(args.cond, args.model_seed, args.ext_steps, args.tgt_steps,
                    args.tag, cfg, device, args.smoke)
    out = per_exp_path(args.cond, args.model_seed, args.ext_steps, args.tgt_steps,
                       args.tag, args.smoke)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
