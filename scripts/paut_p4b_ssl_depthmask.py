#!/usr/bin/env python3
"""P4b: 深度区域掩码 SSL 预训练 —— 直击 H5 定位的表征天顶。

P1 SSL 掩码整波束 (49×512 的 beam 轴), 学的是"整束统计/孔径缺失重建";
缺陷回波是**深度方向的局部指示** (时间/声程上的局部高反射), 该结构被整束重建稀释。
本脚本改掩码**深度块** (时间轴上的连续区间), 逼编码器用上下文重建被掩码的深度区域,
从而学**局部回波形态** —— 该物理形态比"试件缺陷率捷径"更可跨试件迁移。

架构复用 P1 的 MAEEncoder/MAEDecoder (只换掩码目标), 训练循环与 paut_ssl_pretrain.py 相同。

Usage:
  python scripts/paut_p4b_ssl_depthmask.py --mask-mode depth --n-blocks 8 --mask-blocks 2 --epochs 40
  python scripts/paut_p4b_ssl_depthmask.py --mask-mode both  # beam + depth 联合
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wndt.models.ssl_ae import MAEEncoder, MAEDecoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402


class DepthMaskAE(nn.Module):
    """掩码深度块/波束的自编码器。mask: (B,1,H,W), 1=可见 0=被掩码。

    mask_mode:
      beam  -> 掩码 n_mask 个整波束 (P1 复现)
      depth -> 把时间轴分 n_blocks 块, 掩码 n_mask 个连续块
      both  -> 同时掩码若干整波束 + 若干深度块
    """

    def __init__(self, d_model=128, mask_mode="depth", n_blocks=8, mask_blocks=2,
                 mask_ratio=0.3, noise_std=0.02, dropout=0.2):
        super().__init__()
        self.encoder = MAEEncoder(d_model, dropout)
        self.decoder = MAEDecoder(d_model)
        self.mask_mode = mask_mode
        self.n_blocks = n_blocks
        self.mask_blocks = mask_blocks
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def build_mask(self, x: torch.Tensor):
        B, C, H, W = x.shape
        mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        if self.mask_mode in ("beam", "both"):
            n_mask = max(1, int(H * self.mask_ratio))
            rand = torch.rand(B, H, device=x.device)
            idx = rand.topk(n_mask, dim=1).indices
            for b in range(B):
                mask[b, 0, idx[b], :] = 0.0
        if self.mask_mode in ("depth", "both"):
            bw = W // self.n_blocks
            # 随机选 mask_blocks 个起始块 (允许跨块)
            n_mask = min(self.mask_blocks, self.n_blocks)
            starts = torch.randint(0, self.n_blocks - n_mask + 1, (B,), device=x.device)
            for b in range(B):
                s = int(starts[b])
                for k in range(n_mask):
                    mask[b, 0, :, (s + k) * bw:(s + k + 1) * bw] = 0.0
        return mask

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        mask = self.build_mask(x)
        x_masked = x * mask
        if self.noise_std > 0 and self.training:
            x_masked = x_masked + torch.randn_like(x_masked) * self.noise_std * x_masked.std()
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, x, mask

    def recon_loss(self, recon, target, mask):
        inv = 1.0 - mask
        diff = (recon - target) * inv
        masked = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / \
            inv.sum().clamp(min=1)
        full = F.smooth_l1_loss(recon, target)
        return masked + 0.5 * full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-mode", choices=["beam", "depth", "both"], default="depth")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--mask-blocks", type=int, default=2)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_ae_depth")
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    mv = np.load(processed / "ascans_mv.npy")  # (N,4,49,512)
    N, V, H, W = mv.shape
    X = mv.reshape(N * V, H, W).astype(np.float32)
    with open(processed / "norm_stats_mv.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (X - ts_mean) / ts_std
    X = np.ascontiguousarray(X)
    print(f"P4b SSL 预训练: {X.shape} | mask_mode={args.mask_mode} "
          f"blocks={args.n_blocks}/{args.mask_blocks} | epochs={args.epochs}")

    ds = TensorDataset(torch.from_numpy(X))
    if args.smoke:
        idx = np.linspace(0, len(ds) - 1, 512).astype(int)
        ds = TensorDataset(torch.from_numpy(X[idx]))
        args.epochs = 2
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthMaskAE(d_model=args.d_model, mask_mode=args.mask_mode,
                        n_blocks=args.n_blocks, mask_blocks=args.mask_blocks,
                        mask_ratio=args.mask_ratio, noise_std=args.noise_std).to(device)
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
            x = x.to(device, non_blocking=True)
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

    enc_path = out / "encoder.pt"
    torch.save({"encoder_state": model.encoder.state_dict(),
                "d_model": args.d_model, "mask_mode": args.mask_mode,
                "n_blocks": args.n_blocks, "mask_blocks": args.mask_blocks,
                "epochs": args.epochs, "n_samples": int(len(X)),
                "wall_s": round(wall, 1)}, enc_path)
    with open(out / "pretrain_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nP4b SSL 预训练完成 | {args.epochs} epochs | {wall:.0f}s | -> {enc_path}")


if __name__ == "__main__":
    main()
