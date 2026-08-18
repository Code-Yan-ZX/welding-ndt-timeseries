#!/usr/bin/env python3
"""P6: 文献创新 SSL 预训练 —— BSS 批次样式标准化 / 模糊目标 MAE (PixMIM 式)。

动机 (P6, 2026-08-14): P0-P5d 全部廉价杠杆证伪, 天花板在"SSL 表征的跨试件可判别性"
(缺陷率 0.5%-76% 与试件身份耦合)。文献方向: 在**预训练阶段**教编码器对"试件样式"
(增益/衰减/散斑统计)不变, 同时保留内容(缺陷回波结构), 而非有监督对齐(已证伪的 DANN)。

变体:
  base   : P1 复现 (掩码 30% 波束 + 高斯噪声, Huber 重建) —— 对照
  bss    : Batch Styles Standardization [2303.06088] —— 每批 2D FFT, 各样本幅值谱
           替换为该批平均幅值(保留相位), 逆变换后进 MAE。去试件增益/噪声风格,
           逼编码器依赖相位(回波位置结构)而非幅值(试件风格)。
  deblur : 模糊目标 MAE [2306.08249 / PixMIM 2303.02416] —— 重建目标 = 高斯模糊的
           原图 (sigma=2, 核 9), 避免编码器把容量花在拟合散斑噪声上, 专注回波结构。

下游评估同规范头协议: 冻结编码器 + 分类头 lr=1e-3/80ep, 5 折 LOOCV, 非PP4 逐折均值。
多 seed (42/43/44) 才算数 (P4a 教训)。

Usage:
  python scripts/paut_p6_litssl_pretrain.py --variant base  --out experiments/runs/ssl_p6_base  --seed 42
  python scripts/paut_p6_litssl_pretrain.py --variant bss   --out experiments/runs/ssl_p6_bss   --seed 42
  python scripts/paut_p6_litssl_pretrain.py --variant deblur --out experiments/runs/ssl_p6_deblur --seed 42
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from wndt.models.ssl_ae import MaskedAE  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "bss", "deblur"], default="base")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_p6_base")
    return ap.parse_args()


def _gauss_kernel(k=9, sigma=2.0, device="cpu"):
    """一维高斯核 (沿深度/时间轴平滑)。"""
    t = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2
    g = torch.exp(-t**2 / (2 * sigma**2))
    g = g / g.sum()
    return g.view(1, 1, 1, k)


class GaussianBlur2D:
    """深度轴高斯模糊 (B,1,H,W) -> (B,1,H,W)。只沿 W (声程) 平滑, 保波束方向峰结构。"""

    def __init__(self, k=9, sigma=2.0, device="cpu"):
        self.k = k
        self.sigma = sigma
        self.device = device

    def __call__(self, x):
        g = _gauss_kernel(self.k, self.sigma, x.device)
        return F.conv2d(x, g, padding=(0, self.k // 2))


def bss_standardize(x):
    """BSS: 每批 2D FFT, 各样本幅值谱替换为批平均幅值(保留相位)。x: (B,1,H,W)。"""
    B, C, H, W = x.shape
    xc = x.squeeze(1)  # (B,H,W)
    spec = torch.fft.rfft2(xc)  # (B,H,W//2+1)
    amp = spec.abs()
    ph = spec.angle()
    mean_amp = amp.mean(dim=0, keepdim=True)  # (1,H,W//2+1)
    new_spec = mean_amp * torch.exp(1j * ph)
    out = torch.fft.irfft2(new_spec, s=(H, W))
    return out.unsqueeze(1)


def main():
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    mv = np.load(processed / "ascans_mv.npy")  # (N,4,49,512)
    N, V, H, W = mv.shape
    X = mv.reshape(N * V, H, W).astype(np.float32)  # (N*4, 49, 512)
    with open(processed / "norm_stats_mv.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (X - ts_mean) / ts_std
    X = np.ascontiguousarray(X)
    print(f"P6[{args.variant}] SSL 预训练数据: {X.shape} | mask_ratio={args.mask_ratio} | seed={args.seed}")

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
    blur = GaussianBlur2D() if args.variant == "deblur" else None
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
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if args.variant == "bss":
                x = bss_standardize(x)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            recon, target, mask = model(x)
            if blur is not None:
                target = blur(target)
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
                "variant": args.variant,
                "d_model": args.d_model,
                "mask_ratio": args.mask_ratio,
                "noise_std": args.noise_std,
                "seed": args.seed,
                "epochs": args.epochs,
                "n_samples": int(len(X)),
                "wall_s": round(wall, 1)}, enc_path)
    with open(out / "pretrain_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nP6[{args.variant}] 预训练完成 | {args.epochs} epochs | {wall:.0f}s | -> {enc_path}")
    print(f"  最终 recon_loss={log[-1]['loss']:.5f}")


if __name__ == "__main__":
    main()
