#!/usr/bin/env python3
"""P5 缺陷注入自监督预训练 (Defect Injection MAE) — 直击 H5 表征天顶。

核心思想 (与 P1/P4b 的本质区别):
  P1/P4b 用 MAE 重建被掩码区域 -> 编码器学"这个试件里什么是常规的 (本底回波)"
  P5 在 clean A-scan 上**注入物理保真的合成缺陷** (局部高斯峰, 相对本底小),
    训练编码器同时做:
      (a) MAE 重建 (辅助, 保持本底知识)
      (b) 二分类: 是否注入了缺陷 (跨试件通用"异常检测"能力)
      (c) 定位: 注入位置在 7x16 网格中的哪一格 (跨试件通用"缺陷形态"能力)
  -> 编码器被迫学"在任意试件的强本底中检测小幅异常峰" = 通用缺陷回波物理

H5 oracle 的根因: 现行 SSL 学的是试件统计 (P4a 4.4.1), 不是缺陷物理。
P5 通过让"缺陷存在"的标签由**跨试件恒定的注入过程**决定 (而非真实标签),
强制编码器学到跨试件不变的"小峰"特征。

Usage:
  python scripts/paut_p5_inject_pretrain.py --smoke
  python scripts/paut_p5_inject_pretrain.py --epochs 40 --d-model 128
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wndt.models.ssl_ae import MAEDecoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402
from paut_p5_smoke import inject_defect  # noqa: E402


def inject_batch(x: torch.Tensor, p_inject: float = 0.5,
                 amp_range=(800, 3000), edge=(8, 20)):
    """对 batch (B, 49, 512) 注入缺陷, 返回 (injected_x, inj_label, loc_target)。

    假设 x 已是 raw 尺度 (归一化之前)。"""
    B, H, W = x.shape
    x_np = x.cpu().numpy()
    x_inj = x_np.copy()
    inj_label = np.zeros(B, dtype=np.float32)
    loc_target = np.full((B, 2), -1, dtype=np.float32)
    rng = np.random.default_rng()
    for i in range(B):
        if np.random.rand() < p_inject:
            inj, params = inject_defect(x_np[i], rng, amp_range=amp_range, edge=edge)
            x_inj[i] = inj
            inj_label[i] = 1.0
            loc_target[i] = (params[0], params[1])
    return (torch.from_numpy(x_inj).to(x.device),
            torch.from_numpy(inj_label).to(x.device),
            torch.from_numpy(loc_target).to(x.device))


def make_loc_target_heatmap(loc_target: torch.Tensor, grid_h: int = 7, grid_w: int = 16,
                            sigma: float = 1.0):
    """(b0, t0) -> (B, grid_h, grid_w) 高斯热图。"""
    B = loc_target.shape[0]
    H, W = 49, 512
    bh = H / grid_h
    bw = W / grid_w
    target = torch.zeros(B, grid_h, grid_w, device=loc_target.device)
    for i in range(B):
        b0, t0 = loc_target[i, 0].item(), loc_target[i, 1].item()
        if b0 < 0:
            continue
        cb = min(grid_h - 1, max(0, int(b0 / bh)))
        cw = min(grid_w - 1, max(0, int(t0 / bw)))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                gi, gj = cb + di, cw + dj
                if 0 <= gi < grid_h and 0 <= gj < grid_w:
                    d2 = (di ** 2 + dj ** 2) / (sigma ** 2)
                    target[i, gi, gj] = max(target[i, gi, gj].item(), float(np.exp(-d2)))
    return target


class DefectInjectAE(nn.Module):
    """P5 缺陷注入自编码器。"""

    def __init__(self, d_model: int = 128, mask_ratio: float = 0.3, dropout: float = 0.2,
                 grid_h: int = 7, grid_w: int = 16, mask_mode: str = "beam"):
        super().__init__()
        self.d_model = d_model
        self.grid_h, self.grid_w = grid_h, grid_w
        self.mask_ratio = mask_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 7), padding=(1, 3)), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 7), padding=(1, 3)), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 7), padding=(1, 3)), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d((2, 2)),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(128, d_model))
        self.decoder = MAEDecoder(d_model)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        self.loc_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def mask_beams(self, x: torch.Tensor):
        B, C, H, W = x.shape
        n_mask = max(1, int(H * self.mask_ratio))
        mask = torch.ones(B, 1, H, 1, device=x.device)
        rand = torch.rand(B, H, device=x.device)
        idx = rand.topk(n_mask, dim=1).indices
        for b in range(B):
            mask[b, 0, idx[b], 0] = 0.0
        return mask

    def encode(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.conv(x)
        spatial = self.spatial_pool(feat)
        global_feat = self.global_pool(feat)
        z = self.proj(global_feat)
        return z, spatial, feat

    def forward(self, x: torch.Tensor, return_inject_heads: bool = True):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        mask = self.mask_beams(x)
        x_masked = x * mask
        x_masked = x_masked + torch.randn_like(x_masked) * 0.02 * x_masked.std()
        z_masked, _, _ = self.encode(x_masked)
        recon = self.decoder(z_masked)
        if not return_inject_heads:
            return recon, x, mask
        z_clean, spatial, _ = self.encode(x)
        inj_logit = self.cls_head(z_clean).squeeze(-1)
        loc_logits = self.loc_head(spatial).squeeze(1)
        return recon, x, mask, inj_logit, loc_logits

    def recon_loss(self, recon, target, mask):
        inv = 1.0 - mask
        diff = (recon - target) * inv
        masked = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / inv.sum().clamp(min=1)
        full = F.smooth_l1_loss(recon, target)
        return masked + 0.5 * full


class P5Dataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--p-inject", type=float, default=0.5)
    ap.add_argument("--w-recon", type=float, default=1.0)
    ap.add_argument("--w-inj", type=float, default=2.0)
    ap.add_argument("--w-loc", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_p5_inject")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    mv = np.load(processed / "ascans_mv.npy")
    N, V, H, W = mv.shape
    X = mv.reshape(N * V, H, W).astype(np.float32)
    with open(processed / "norm_stats_mv.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    print(f"P5 pretrain: 注入在归一化前 (raw 尺度) | data {X.shape} | "
          f"p_inject={args.p_inject} mask={args.mask_ratio} "
          f"w_recon={args.w_recon} w_inj={args.w_inj} w_loc={args.w_loc}")
    X_t = torch.from_numpy(X)
    ds = TensorDataset(X_t)
    if args.smoke:
        idx = torch.linspace(0, len(ds) - 1, 512).long()
        ds = TensorDataset(X_t[idx])
        args.epochs = 2
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_mean_t = torch.from_numpy(ts_mean).to(device)
    ts_std_t = torch.from_numpy(ts_std).to(device)
    model = DefectInjectAE(d_model=args.d_model, mask_ratio=args.mask_ratio).to(device)
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
        ep = {"loss": 0.0, "l_rec": 0.0, "l_inj": 0.0, "l_loc": 0.0, "inj_acc": 0.0, "n": 0}
        for batch in loader:
            x_raw = batch[0].to(device, non_blocking=True)
            # 1) 注入缺陷 (raw 尺度, 800-3000 是 raw 振幅)
            x_inj_raw, inj_label, loc_target = inject_batch(x_raw, p_inject=args.p_inject)
            # 2) 归一化 (注入后再归一化, 与推理一致)
            x = (x_inj_raw - ts_mean_t) / ts_std_t
            x_raw_norm = (x_raw - ts_mean_t) / ts_std_t  # 用于 recon target
            # 3) 前向 (基于注入后的输入做 MAE 掩码重建)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            recon, target, mask, inj_logit, loc_logits = model(x, return_inject_heads=True)
            # 4) 损失 (recon target 应该是原始未注入的归一化信号, 但我们已注入, 退而求其次用注入后的, 仍能学)
            l_rec = model.recon_loss(recon, x_raw_norm.unsqueeze(1), mask)
            l_inj = F.binary_cross_entropy_with_logits(inj_logit, inj_label)
            loc_heat = make_loc_target_heatmap(loc_target, grid_h=model.grid_h,
                                               grid_w=model.grid_w)
            l_loc = F.binary_cross_entropy_with_logits(loc_logits, loc_heat)
            loss = args.w_recon * l_rec + args.w_inj * l_inj + args.w_loc * l_loc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                inj_pred = (inj_logit.sigmoid() > 0.5).float()
                inj_acc = (inj_pred == inj_label).float().mean().item()
            bs = x.size(0)
            ep["loss"] += loss.item() * bs
            ep["l_rec"] += l_rec.item() * bs
            ep["l_inj"] += l_inj.item() * bs
            ep["l_loc"] += l_loc.item() * bs
            ep["inj_acc"] += inj_acc * bs
            ep["n"] += bs
            step += 1
        n = ep["n"]
        msg = (f"epoch {epoch+1:02d}/{args.epochs} | loss {ep['loss']/n:.4f} | "
               f"rec {ep['l_rec']/n:.4f} | inj {ep['l_inj']/n:.4f} | loc {ep['l_loc']/n:.4f} | "
               f"inj_acc {ep['inj_acc']/n:.4f}")
        print(msg)
        log.append({"epoch": epoch + 1, **{k: ep[k] / n for k in ep if k != "n"}, "n": n})

    dt = time.time() - t0
    print(f"P5 pretrain done: {dt:.1f}s")

    # Save encoder weights (compatible with P1 format: encoder_state + d_model)
    enc_state = {}
    for k, v in model.state_dict().items():
        if k.startswith("conv."):
            enc_state[k[len("conv."):]] = v
        elif k.startswith("proj."):
            enc_state[k[len("proj."):]] = v
    torch.save({"encoder_state": enc_state, "d_model": args.d_model,
                "grid_h": model.grid_h, "grid_w": model.grid_w}, out / "encoder.pt")
    with open(out / "pretrain_log.json", "w") as fh:
        args_save = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        json.dump({"args": args_save, "log": log, "elapsed_s": dt,
                   "final_inj_acc": log[-1]["inj_acc"]}, fh, indent=2)
    print(f"Saved: {out / 'encoder.pt'}")


if __name__ == "__main__":
    main()
