#!/usr/bin/env python3
"""P5b per-fold LOOCV — 用 per-fold 预训练编码器 (cold-start) 评估 SupCon 真实迁移性.

每个 test 折用对应的 per-fold 编码器 (训练时排除该折).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wndt.data.paut_dataset import PAUTSeriesDataset  # noqa: E402
from wndt.models.ssl_ae import MAEEncoder, SSLClassifier  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NP4 = ["PP3", "PP5", "PP6", "PP7"]
DATA = REPO / "data/processed/paut"
RES = REPO / "experiments/results"
PERFOLD_ROOT = REPO / "experiments/runs/ssl_p5b_perfold"


def fold_splits(coupons, labels, test_coupon, val_frac, seed):
    test_idx = np.nonzero(coupons == test_coupon)[0].astype(np.int64)
    rest = np.nonzero(coupons != test_coupon)[0].astype(np.int64)
    y_rest = labels[rest]
    stratify = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
    train_idx, val_idx = train_test_split(rest, test_size=val_frac, random_state=seed,
                                          shuffle=True, stratify=stratify)
    return (np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))


def fold_norm(ascans, train_idx):
    tr = np.array(ascans[train_idx], dtype=np.float32)
    flat = tr.reshape(-1, tr.shape[-1])
    return (flat.mean(axis=0).astype(np.float32),
            (flat.std(axis=0) + 1e-8).astype(np.float32))


def make_weighted_sampler(labels):
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_w = weights[labels]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


@torch.no_grad()
def scores_of(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        logits = model(x)
        out.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_fold(model, train_ds, val_ds, device, *, epochs, lr, wd, batch_size, seed,
               patience=20):
    set_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              sampler=make_weighted_sampler(train_ds.labels))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    best_auc, best_state, bad = -1.0, None, 0
    last_ep = 0
    for ep in range(epochs):
        last_ep = ep + 1
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        sched.step()
        sv = scores_of(model, val_loader, device)
        yv = val_ds.labels
        v_auc = float(roc_auc_score(yv, sv)) if len(np.unique(yv)) == 2 else 0.0
        if v_auc > best_auc + 1e-4:
            best_auc, bad = v_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_auc": best_auc, "epochs_run": last_ep}


def run(args, device):
    coupons = np.load(DATA / "meta_coupon.npy")
    labels = np.load(DATA / "meta_label.npy")
    ascans = np.load(DATA / "ascans.npy", mmap_mode="r")

    folds = []
    for tc in COUPONS:
        # 关键: 用 per-fold 编码器 (训练时排除 tc)
        if tc == "PP4":
            # PP4 不纳入均值, 但仍要测
            ckpt_path = PERFOLD_ROOT / f"test_{tc}_s{args.seed}" / "encoder.pt"
            # PP4 没 per-fold pretrain (排除 PP4 时已包含 PP3/5/6/7), 但作为 test 时编码器应
            # 该用"训练时不含 PP4"的版本 — 实际我们已经训了所有 4 个 per-fold 编码器
            # 包含 test_PP4_s42 (排除 PP4 用 PP3/5/6/7 训)
            if not ckpt_path.exists():
                print(f"  [skip PP4 no per-fold ckpt]")
                continue
        else:
            ckpt_path = PERFOLD_ROOT / f"test_{tc}_s{args.seed}" / "encoder.pt"
        tr, va, te = fold_splits(coupons, labels, tc, args.val_frac, args.seed)
        ts_mean, ts_std = fold_norm(ascans, tr)
        t0 = time.time()
        train_ds = PAUTSeriesDataset(DATA, tr, beam="bscan",
                                     norm_mode="per_timestep",
                                     ts_mean=ts_mean, ts_std=ts_std)
        val_ds = PAUTSeriesDataset(DATA, va, beam="bscan",
                                   norm_mode="per_timestep",
                                   ts_mean=ts_mean, ts_std=ts_std)
        test_ds = PAUTSeriesDataset(DATA, te, beam="bscan",
                                    norm_mode="per_timestep",
                                    ts_mean=ts_mean, ts_std=ts_std)
        enc = MAEEncoder(d_model=args.d_model, dropout=0.2).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        raw = ckpt["encoder_state"]
        conv_idx = {"0", "1", "4", "5", "8", "9"}
        fixed = {}
        for k, v in raw.items():
            idx = k.split(".", 1)[0]
            if idx in conv_idx:
                fixed["conv." + k] = v
            else:
                fixed["proj." + k] = v
        enc.load_state_dict(fixed)
        model = SSLClassifier(enc, d_model=args.d_model,
                              freeze_encoder=True).to(device)
        fit = train_fold(model, train_ds, val_ds, device, epochs=args.epochs,
                         lr=args.lr, wd=args.wd, batch_size=args.batch,
                         seed=args.seed, patience=args.patience)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                  num_workers=4, pin_memory=True)
        scores = scores_of(model, test_loader, device)
        yt = labels[te]
        auc = float(roc_auc_score(yt, scores)) if len(np.unique(yt)) == 2 else float("nan")
        folds.append({"test_coupon": tc, "auc": auc, "n_pos": int(yt.sum()),
                      "defect_rate": float(yt.mean()),
                      "wall_s": round(time.time() - t0, 1),
                      "val_auc": fit["best_val_auc"],
                      "epochs_run": fit["epochs_run"],
                      "scores": scores.tolist()})
        print(f"  [P5b-perfold] test={tc}: AUC={auc:.3f} val={fit['best_val_auc']:.3f} "
              f"ep={fit['epochs_run']} ({round(time.time()-t0,1)}s)", flush=True)

    np4 = [f["auc"] for f in folds if f["test_coupon"] in NP4]
    pooled_scores, pooled_labels = [], []
    for f in folds:
        if f["test_coupon"] in NP4:
            m = coupons == f["test_coupon"]
            pooled_scores.append(np.array(f["scores"]))
            pooled_labels.append(labels[m])
    pooled_scores = np.concatenate(pooled_scores)
    pooled_labels = np.concatenate(pooled_labels)
    pooled_auc = float(roc_auc_score(pooled_labels, pooled_scores))
    summary = {"exp": "p5b_perfold", "seed": args.seed,
               "nonPP4_fold_mean": float(np.mean(np4)),
               "nonPP4_pooled": pooled_auc,
               "folds": [{k: v for k, v in f.items() if k != "scores"} for f in folds]}
    print(f"[P5b-perfold seed={args.seed}] nonPP4 逐折均值={np.mean(np4):.3f} pooled={pooled_auc:.3f}")
    out_tag = f"perfold_s{args.seed}"
    with open(RES / f"paut_p5b_{out_tag}_full.json", "w") as fh:
        json.dump(dict(summary, folds=folds), fh, indent=2, ensure_ascii=False)
    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed} (per-fold strict)")
    run(args, device)


if __name__ == "__main__":
    main()
