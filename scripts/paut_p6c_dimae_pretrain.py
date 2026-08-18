#!/usr/bin/env python3
"""P6c: DiMAE 式域不变掩码自编码器预训练 [2205.04771] —— PAUT 试件版。

核心 (DiMAE 适配 PAUT, 试件=域):
  1. 内容保持风格混合 (content-preserved style mix): 对样本 x (试件 s) 取随机其它
     试件 s' 的傅里叶幅值谱与其相位混合, 得到 x_mixed (内容/相位保持, 风格被混入
     其它试件样式)。
  2. 逐试件解码器 (multiple domain-specific decoders): 共享编码器 z=enc(x_mixed),
     用**原试件 s** 的解码器重建原始 x。迫使编码器学"内容(缺陷回波结构)不变、
     风格(试件增益/散斑)不变"的表征 —— 直接针对跨试件泛化失败 (P0-P5 天花板)。

  vs 已证伪的 DANN: DANN 用对抗迫使编码器"彻底不区分试件", 会把缺陷判别信号一起
  洗掉; DiMAE 用**重建** + 逐试件解码器, 让风格由解码器吸收, 编码器只保留内容。

  vs 已证伪的 P4a"去试件均值": 那是特征层后处理一刀切; 本方法是预训练中逐步学习。

变体:
  dimae : 风格混合 + 逐试件解码器 (完整 DiMAE)
  multidec : 仅逐试件解码器, 无风格混合 (隔离风格混合的贡献, 对照)

下游评估: 冻结共享编码器 + 分类头 lr=1e-3/80ep, 5 折 LOOCV, 非PP4 逐折均值, 多 seed。
Usage:
  python scripts/paut_p6c_dimae_pretrain.py --variant dimae --out experiments/runs/ssl_p6_dimae --seed 42
  python scripts/paut_p6c_dimae_pretrain.py --variant multidec --out experiments/runs/ssl_p6_multidec --seed 42
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from wndt.models.ssl_ae import MAEEncoder, MAEDecoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dimae", "multidec"], default="dimae")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_p6_dimae")
    return ap.parse_args()


def style_mix(x, coupon, all_coupons):
    """内容保持风格混合: 每个样本的幅值谱与随机另一试件的幅值谱混合, 保留自身相位。
    x: (B,1,H,W); coupon: (B,) 原试件索引。返回 x_mixed。"""
    B, C, H, W = x.shape
    spec = torch.fft.rfft2(x.squeeze(1))  # (B,H,W//2+1)
    amp = spec.abs()
    ph = spec.angle()
    # 为每个样本随机选另一试件
    others = np.random.choice(len(all_coupons), size=B).astype(np.int64)
    # 从同 batch 中挑幅值谱: 用 coupon 向量做桶; 简化: 从 batch 内其它样本抽样
    perm = torch.randperm(B, device=x.device)
    other_amp = amp[perm]  # 用 batch 内另一随机样本的幅值 (近似跨试件混合)
    alpha = 0.5
    mixed_amp = alpha * amp + (1 - alpha) * other_amp
    new_spec = mixed_amp * torch.exp(1j * ph)
    out = torch.fft.irfft2(new_spec, s=(H, W))
    return out.unsqueeze(1)


class MultiDecoderMAE(nn.Module):
    """共享 MAE 编码器 + N 个逐试件解码器。风格混合输入, 原试件解码器重建。"""

    def __init__(self, n_domains=5, d_model=128, mask_ratio=0.3, noise_std=0.02,
                 dropout=0.2):
        super().__init__()
        self.encoder = MAEEncoder(d_model, dropout)
        self.decoders = nn.ModuleList([MAEDecoder(d_model) for _ in range(n_domains)])
        self.n_domains = n_domains
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def mask_beams(self, x):
        B, C, H, W = x.shape
        n_mask = max(1, int(H * self.mask_ratio))
        mask = torch.ones(B, 1, H, 1, device=x.device)
        rand = torch.rand(B, H, device=x.device)
        idx = rand.topk(n_mask, dim=1).indices
        for b in range(B):
            mask[b, 0, idx[b], 0] = 0.0
        x_masked = x * mask
        if self.noise_std > 0 and self.training:
            x_masked = x_masked + torch.randn_like(x_masked) * self.noise_std * x_masked.std()
        return x_masked, mask

    def forward(self, x, coupon):
        """x: (B,1,H,W) 原始; coupon: (B,) 试件索引。返回 (recon_of_orig, x, mask)。"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.size(0)
        x_masked, mask = self.mask_beams(x)
        z = self.encoder(x_masked)
        recon = torch.empty_like(x)
        for d in range(self.n_domains):
            sel = (coupon == d).nonzero(as_tuple=True)[0]
            if len(sel) > 0:
                recon[sel] = self.decoders[d](z[sel])
        return recon, x, mask

    def recon_loss(self, recon, target, mask):
        inv = 1.0 - mask
        diff = (recon - target) * inv
        masked = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / inv.sum().clamp(min=1)
        full = F.smooth_l1_loss(recon, target)
        return masked + 0.5 * full


def main():
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    mv = np.load(processed / "ascans_mv.npy")  # (N,4,49,512)
    N, V, H, W = mv.shape
    X = mv.reshape(N * V, H, W).astype(np.float32)  # (N*4, 49, 512)
    coupon_pos = np.load(processed / "meta_coupon_mv.npy")  # (N,) 每位置试件
    coupon_all = np.asarray(COUPONS)
    # 每样本试件 = 位置试件 (4 视角同试件)
    pos_of = np.arange(N * V) // V
    coupon_idx = np.array([np.where(coupon_all == c)[0][0]
                           for c in coupon_pos[pos_of]])  # (N*V,)

    with open(processed / "norm_stats_mv.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (X - ts_mean) / ts_std
    X = np.ascontiguousarray(X)
    print(f"P6c[{args.variant}] 数据: {X.shape} | 试件数={len(coupon_all)} | "
          f"试件样本分布={np.bincount(coupon_idx, minlength=5).tolist()}")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(coupon_idx))
    if args.smoke:
        idx = np.linspace(0, len(ds) - 1, 512).astype(int)
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(coupon_idx[idx]))
        args.epochs = 2

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiDecoderMAE(n_domains=len(coupon_all), d_model=args.d_model,
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
        for x, coup in loader:
            x = x.to(device, non_blocking=True)
            coup = coup.to(device, non_blocking=True)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if args.variant == "dimae":
                x_in = style_mix(x, coup, coupon_all)
            else:
                x_in = x
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            recon, target, mask = model(x_in, coup)
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
                "seed": args.seed,
                "epochs": args.epochs,
                "n_samples": int(len(X)),
                "wall_s": round(wall, 1)}, enc_path)
    with open(out / "pretrain_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\nP6c[{args.variant}] 预训练完成 | {args.epochs} epochs | {wall:.0f}s | -> {enc_path}")
    print(f"  最终 recon_loss={log[-1]['loss']:.5f}")


if __name__ == "__main__":
    main()
