#!/usr/bin/env python
"""PAUT 自监督掩码自编码器预训练 (P1-①②)。

在全部无标注 B-scan 上预训练 MaskedAE (掩码波束重建 + 去噪), 学焊缝超声专属表征。
数据: ascans_mv.npy (N,4,49,512) 展平为 N*4 个 (49,512) 样本 (4 视角全用, 无标注,
跨所有试件 -- SSL 不用标签, 用全部数据符合范式)。归一化用全量 per-timestep 统计。

Usage:
  python scripts/paut_ssl_pretrain.py --epochs 40 --d-model 128
  python scripts/paut_ssl_pretrain.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from wndt.models.ssl_ae import MaskedAE  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_ae")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    mv = np.load(processed / "ascans_mv.npy")  # (N,4,49,512)
    N, V, H, W = mv.shape
    X = mv.reshape(N * V, H, W).astype(np.float32)  # (N*4, 49, 512)
    # per-timestep norm (全量, SSL 无泄漏概念)
    with open(processed / "norm_stats_mv.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (X - ts_mean) / ts_std
    X = np.ascontiguousarray(X)
    print(f"SSL 预训练数据: {X.shape} (N={N} 试件位置 × {V} 视角) | mask_ratio={args.mask_ratio}")

    ds = TensorDataset(torch.from_numpy(X))
    if args.smoke:
        idx = np.linspace(0, len(ds) - 1, 512).astype(int)
        ds = TensorDataset(torch.from_numpy(X[idx]))
        args.epochs = 2

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAE(d_model=args.d_model, mask_ratio=args.mask_ratio,
                     noise_std=args.noise_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(loader) * args.epochs

    def lr_at(step):
        warm = 200
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    log = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for (x,) in loader:
            x = x.to(device, non_blocking=True)  # (B,49,512)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            recon, target, mask = model(x)
            loss = model.recon_loss(recon, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0); step += 1
        rec = {"epoch": epoch, "loss": tot / max(1, n), "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d} | recon_loss {rec['loss']:.5f} | lr {rec['lr']:.2e}")
    wall = time.time() - t0

    # 保存编码器 (供下游 LOOCV 用)
    enc_path = out / "encoder.pt"
    torch.save({"encoder_state": model.encoder.state_dict(),
                "d_model": args.d_model,
                "mask_ratio": args.mask_ratio,
                "noise_std": args.noise_std,
                "epochs": args.epochs,
                "n_samples": int(len(X)),
                "wall_s": round(wall, 1)}, enc_path)
    with open(out / "pretrain_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nSSL 预训练完成 | {args.epochs} epochs | {wall:.0f}s | 编码器 -> {enc_path}")
    print(f"  最终 recon_loss={log[-1]['loss']:.5f}")


if __name__ == "__main__":
    main()
