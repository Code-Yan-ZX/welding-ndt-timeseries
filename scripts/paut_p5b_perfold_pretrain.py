#!/usr/bin/env python3
"""P5b 严格 per-fold 评估 — 排除信息泄露, 验证 SupCon 是否真能跨试件迁移.

关键差异 vs P5b 原版:
  - pretrain 只用 test 折**之外**的 4 个试件标签
  - 这是真正的 cold-start 跨域评估
  - 如果 0.985 是信息泄露假象, 这里会跌回 ~0.6 附近
  - 如果 per-fold 仍 > 0.65, 则是真实的迁移性提升

Usage:
  python scripts/paut_p5b_perfold_pretrain.py --test-fold PP3 --seed 42
  python scripts/paut_p5b_perfold_pretrain.py --all-folds --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wndt.utils.seed import set_seed  # noqa: E402
from paut_p5b_supcon_pretrain import (  # noqa: E402
    SupConModel, supcon_loss, CrossSpecimenBatchSampler,
)


def pretrain_fold(test_coupon, seed, epochs, batch_size, lr, d_model, proj_dim,
                  temperature, out_dir, device):
    """仅用非 test 试件 pretrain, 保存 encoder."""
    processed = REPO / "data/processed/paut"
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy").astype(np.int64)
    ascans = np.load(processed / "ascans.npy").astype(np.float32)
    with open(processed / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (ascans - ts_mean) / ts_std
    X = np.ascontiguousarray(X)

    keep = coupons != test_coupon
    X_tr = X[keep]
    y_tr = labels[keep]
    cp_tr = coupons[keep]
    print(f"  pretrain: keep n={keep.sum()} | coupons={np.unique(cp_tr)} | "
          f"defect rate={y_tr.mean():.3f}")

    X_t = torch.from_numpy(X_tr).float()
    Y_t = torch.from_numpy(y_tr)
    ds = TensorDataset(X_t, Y_t)
    bsampler = CrossSpecimenBatchSampler(cp_tr, batch_size, seed=seed)
    loader = DataLoader(ds, batch_sampler=bsampler, num_workers=4, pin_memory=True)

    model = SupConModel(d_model=d_model, proj_dim=proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(loader) * epochs

    def lr_at(step):
        warm = 200
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    step = 0
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            _, p = model(x)
            loss = supcon_loss(p, y, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_state = {}
    for k, v in model.state_dict().items():
        if k.startswith("encoder.conv."):
            enc_state[k[len("encoder.conv."):]] = v
        elif k.startswith("encoder.proj."):
            enc_state[k[len("encoder.proj."):]] = v
    torch.save({"encoder_state": enc_state, "d_model": d_model,
                "proj_dim": proj_dim}, out_dir / "encoder.pt")
    print(f"  saved {out_dir / 'encoder.pt'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fold", type=str, help="单折: PP3/PP5/PP6/PP7")
    ap.add_argument("--all-folds", action="store_true", help="所有 4 折")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, default=REPO / "experiments/runs/ssl_p5b_perfold")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.test_fold and not args.all_folds:
        print("Specify --test-fold PP3/PP5/PP6/PP7 or --all-folds")
        return
    folds = ["PP3", "PP5", "PP6", "PP7"] if args.all_folds else [args.test_fold]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    for tc in folds:
        out_dir = args.out_root / f"test_{tc}_s{args.seed}"
        print(f"\n=== Per-fold SupCon pretrain: test={tc} ===")
        t0 = time.time()
        pretrain_fold(tc, args.seed, args.epochs, args.batch_size, args.lr,
                      args.d_model, args.proj_dim, args.temperature, out_dir, device)
        print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
