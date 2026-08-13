#!/usr/bin/env python3
"""P5b 跨试件监督对比学习 (Supervised Contrastive, SupCon)。

P5a 缺陷注入失败的根因: 合成的高斯峰形态空间与真实 PAUT 缺陷形态空间不重合。
P5b 直接用真实标签, 跨试件采样让 positive pair 天然来自不同试件,
  强制编码器学"试件不变的'有缺陷 vs 无缺陷'判别特征"。

H5 oracle 根因 (P4a): SSL 学的是"试件统计"而非"缺陷物理"。
P5a 失败: 学的是"高斯峰"形态, 同样不跨试件。
P5b 假设: 用真实标签 + 跨试件 positive pair, 编码器被迫学"试件不变的判别边界"。

关键机制:
  - 监督对比损失 (Khosla et al. 2020): positives = 同标签样本 (跨试件自然形成),
    negatives = 不同标签样本
  - 跨试件 batch 采样: 每 batch 从 4 个训练试件按比例采样, 物理保证 positives 来自不同试件
  - 投影头: encoder 128 -> 64 (低维空间做对比, 上游任务用 128 全特征)
  - 温度 τ = 0.07 (SupCon 标准)

Usage:
  python scripts/paut_p5b_supcon_pretrain.py --smoke
  python scripts/paut_p5b_supcon_pretrain.py --epochs 40 --seed 42
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

from wndt.models.ssl_ae import MAEEncoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402


class SupConModel(nn.Module):
    """MAEEncoder + 投影头, 输出 64 维投影用于对比学习。

    投影头 (2-layer MLP) 是 SimCLR/SupCon 标配, 让对比损失作用在低维流形上,
    编码器主特征 (128-dim) 保留高维判别信息。
    """

    def __init__(self, d_model: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = MAEEncoder(d_model, dropout=0.2)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.proj_dim = proj_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        p = self.proj_head(z)
        return z, F.normalize(p, dim=-1)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Supervised Contrastive Loss (Khosla et al. 2020, Eq. 2).

    features: (B, D), L2-normalized
    labels: (B,)
    """
    B = features.size(0)
    sim = features @ features.t() / temperature  # (B, B)
    # 数值稳定: 减去每行最大值
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)

    # mask: 同标签 (排除自身)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()  # (B, B)
    pos_mask.fill_diagonal_(0)  # 排除自身
    # 排除自身 (denom 中)
    eye_mask = torch.eye(B, device=features.device)
    denom_mask = 1.0 - eye_mask
    denom = (exp_sim * denom_mask).sum(dim=1) + 1e-12  # (B,)

    # 每个 anchor 的平均 positive log-prob
    pos_count = pos_mask.sum(dim=1)  # (B,) 每个 anchor 的 positive 数
    # 跳过没有 positive 的 anchor (label 只出现一次)
    valid = pos_count > 0
    log_prob = sim - torch.log(denom).unsqueeze(1)  # (B, B)
    loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    # 跳过无 positive 的
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)
    return loss_per_anchor[valid].mean()


class CrossSpecimenBatchSampler:
    """跨试件 batch 采样器: 每 batch 从 5 个试件按比例采样, 让 positives 自然跨试件。

    作为 batch_sampler 使用, 每个 yield 是一 batch 索引列表。
    """

    def __init__(self, coupons: np.ndarray, batch_size: int, seed: int = 42):
        self.coupons = np.asarray(coupons)
        self.batch_size = batch_size
        self.unique_coupons = np.unique(self.coupons)
        self.idx_by_coupon = {c: np.where(self.coupons == c)[0] for c in self.unique_coupons}
        self.rng = np.random.default_rng(seed)
        self.per_coupon = max(1, batch_size // len(self.unique_coupons))

    def __iter__(self):
        per_coupon_perm = {c: self.rng.permutation(len(self.idx_by_coupon[c]))
                            for c in self.unique_coupons}
        max_len = max(len(p) for p in per_coupon_perm.values())
        n_batches = (max_len + self.per_coupon - 1) // self.per_coupon
        for b in range(n_batches):
            batch = []
            for c in self.unique_coupons:
                perm = per_coupon_perm[c]
                start = b * self.per_coupon
                end = min(start + self.per_coupon, len(perm))
                if start < len(perm):
                    batch.extend(self.idx_by_coupon[c][perm[start:end]].tolist())
            if batch:
                yield batch

    def __len__(self):
        max_len = max(len(idx) for idx in self.idx_by_coupon.values())
        return (max_len + self.per_coupon - 1) // self.per_coupon


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/runs/ssl_p5b_supcon")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    processed = REPO / "data/processed/paut"
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy").astype(np.int64)
    # SupCon 用 4 个训练试件 (排除 test), 但 pretrain 用全 5 个 (LOOCV fold 内)
    # 这里: 预训练用所有 labeled 数据 (5 试件) — 跨试件 batch sampling 自动涵盖所有
    # 关键: pretrain 不应 leak test label, 但 SupCon 不需要 "label 监督" 的真值,
    # 它用的是数据自带的 label (coupon + defect/non-defect)
    # 因为我们用全 labeled 数据 pretrain, 然后在 LOOCV 时只评估 encoder 迁移性
    # (与 P1 SSL pretrain 跨全 11980 unlabeled 样本类似, 都不 leak test label)

    ascans = np.load(processed / "ascans.npy").astype(np.float32)  # (3000, 49, 512)
    with open(processed / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    X = (ascans - ts_mean) / ts_std
    X = np.ascontiguousarray(X)
    print(f"P5b SupCon: data {X.shape} | coupons {np.unique(coupons)} | "
          f"defect rate {labels.mean():.3f}")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"τ={args.temperature}")

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(labels)
    ds = TensorDataset(X_t, Y_t)
    batch_sampler = CrossSpecimenBatchSampler(coupons, args.batch_size, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupConModel(d_model=args.d_model, proj_dim=args.proj_dim).to(device)
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
        ep = {"loss": 0.0, "n": 0, "valid_frac": 0.0}
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            _, p = model(x)
            loss = supcon_loss(p, y, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = x.size(0)
            ep["loss"] += loss.item() * bs
            ep["n"] += bs
            step += 1
        n = ep["n"]
        avg = ep["loss"] / max(1, n)
        print(f"epoch {epoch+1:02d}/{args.epochs} | loss {avg:.4f}")
        log.append({"epoch": epoch + 1, "loss": avg})

    dt = time.time() - t0
    print(f"P5b SupCon done: {dt:.1f}s")

    # 保存 encoder 权重 (与 P1/P5a 格式兼容: 0.weight, 1.weight, ... 不带 conv./proj. 前缀)
    enc_state = {}
    for k, v in model.state_dict().items():
        if k.startswith("encoder.conv."):
            enc_state[k[len("encoder.conv."):]] = v
        elif k.startswith("encoder.proj."):
            enc_state[k[len("encoder.proj."):]] = v
    torch.save({"encoder_state": enc_state, "d_model": args.d_model,
                "proj_dim": args.proj_dim}, out / "encoder.pt")
    with open(out / "pretrain_log.json", "w") as fh:
        args_save = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        json.dump({"args": args_save, "log": log, "elapsed_s": dt,
                   "final_loss": log[-1]["loss"]}, fh, indent=2)
    print(f"Saved: {out / 'encoder.pt'}")


if __name__ == "__main__":
    main()
