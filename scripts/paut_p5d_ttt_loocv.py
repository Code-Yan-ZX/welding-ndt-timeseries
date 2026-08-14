#!/usr/bin/env python3
"""P5d Test-Time Training (TTT) — 推理时用 P1 SSL MAE 在 test 试件无标签数据上继续微调编码器。

核心假设 (P5d, P4a H3 的"表征级"版本):
  P4a H3-TENT (熵最小化) / H3-BN (BN 重统计) 都是**决策边界层/BN 统计的浅层适配**, 失败。
  P5d 在推理时用 **P1 同款 SSL MAE 目标** (掩码波束重建 + 去噪) 在 test 折的
  **无标签** B-scan 上继续微调编码器权重 (表征级适配), 不接触 test 标签。
  问: 表征级 TTT 能否跨越 H5 oracle 定位的"跨试件天花板" (~0.58)?

per-fold 严格协议 (P5b 教训: 严格 cold-start + val-test gap 是唯一最可靠指示器):
  - head: 冻结 P1 编码器, 只用 4 个训练试件标签训练二分类头, val-AUC 早停 — 与 baseline 完全一致
  - decoder warm-up: P1 只保存了 encoder, 补一个 decoder 在**训练试件**无标签数据上预热
    (只训 decoder, 冻结 encoder; 用训练域数据合法, 不接触 test 标签)
  - TTT: 在 **test 折无标签 bscan** (归一化用 train 折统计, 与 eval 一致) 上微调 encoder
    (decoder 快 LR 跟随), 固定预算 (ttt-steps), 不按 test AUC 调参 = 无泄漏
  - eval: TTT 后冻结 encoder + 已训好的 head, 在 test 折上出 AUC
  - 同脚本内一并输出 no-TTT 对照 (同一 head / 同一 seed → 直接 delta)

主指标: nonPP4 逐折均值 + pooled (与 P4a baseline 同口径), 多 seed (42/43/44)。

Usage:
  python scripts/paut_p5d_ttt_loocv.py --smoke
  CUDA_VISIBLE_DEVICES=2 python scripts/paut_p5d_ttt_loocv.py --seed 42
  CUDA_VISIBLE_DEVICES=2 python scripts/paut_p5d_ttt_loocv.py --seed 42 --ttt-steps 10,30,60  # 灵敏度表(探索)
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wndt.data.paut_dataset import PAUTSeriesDataset  # noqa: E402
from wndt.models.ssl_ae import MAEEncoder, MaskedAE, SSLClassifier  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NP4 = ["PP3", "PP5", "PP6", "PP7"]
DATA = REPO / "data/processed/paut"
RES = REPO / "experiments/results"
P1_CKPT = REPO / "experiments/runs/ssl_ae/encoder.pt"


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
    """与 P4a baseline 完全一致: 冻结 encoder 只训 head, val-AUC 早停。
    返回 (summary, head_state) — head_state 用于 TTT 后复用。"""
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
    return {"best_val_auc": best_auc, "epochs_run": last_ep}, model.head.state_dict()


def build_mae(enc_state, d_model, mask_ratio, noise_std):
    """MaskedAE = P1 encoder + 新 decoder。"""
    mae = MaskedAE(d_model=d_model, mask_ratio=mask_ratio, noise_std=noise_std)
    mae.encoder.load_state_dict(enc_state)
    return mae


@torch.no_grad()
def recon_loss_on(model, X, device, batch_size=256):
    """平均 recon loss (与 P1 同款 Huber, 掩码波束 + 全图)。"""
    model.eval()
    tot, n = 0.0, 0
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    for (x,) in loader:
        x = x.to(device).unsqueeze(1)
        recon, target, mask = model(x)
        tot += model.recon_loss(recon, target, mask).item() * x.size(0)
        n += x.size(0)
    return tot / max(1, n)


def ssl_ttt(enc_state, X_ttt, *, d_model, mask_ratio, noise_std, ttt_steps,
            ttt_batch, ttt_enc_lr, ttt_dec_lr, seed, device, dec_state=None):
    """在 test 折无标签数据上微调 encoder (P1 MAE 目标)。返回 (adapted_encoder, recon_before, recon_after)。
    dec_state: 在训练域 warm-up 好的 decoder 权重 (避免 TTT 梯度被"从零学解码"污染)。"""
    set_seed(seed)
    mae = build_mae(enc_state, d_model, mask_ratio, noise_std).to(device)
    if dec_state is not None:
        mae.decoder.load_state_dict(dec_state)
    r0 = recon_loss_on(mae, X_ttt, device)
    enc_params = [p for n, p in mae.named_parameters() if n.startswith("encoder.")]
    dec_params = [p for n, p in mae.named_parameters() if n.startswith("decoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": ttt_enc_lr},
         {"params": dec_params, "lr": ttt_dec_lr}],
        weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_ttt)), batch_size=ttt_batch,
                        shuffle=True, drop_last=False)
    it = iter(loader)
    mae.train()
    for step in range(ttt_steps):
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x,) = next(it)
        x = x.to(device).unsqueeze(1)
        opt.zero_grad(set_to_none=True)
        recon, target, mask = mae(x)
        loss = mae.recon_loss(recon, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        opt.step()
    r1 = recon_loss_on(mae, X_ttt, device)
    return mae.encoder, r0, r1


@torch.no_grad()
def feature_shift(enc_a, enc_b, X_te, device, batch_size=256):
    """初始 vs TTT 后 encoder 在 test 折上的特征平均 L1 距离 (归一化)。"""
    enc_a.eval(); enc_b.eval()
    tot, n = 0.0, 0
    loader = DataLoader(TensorDataset(torch.from_numpy(X_te)), batch_size=batch_size, shuffle=False)
    for (x,) in loader:
        x = x.to(device).unsqueeze(1)
        za = enc_a(x)
        zb = enc_b(x)
        tot += (za - zb).abs().mean().item() * x.size(0)
        n += x.size(0)
    return tot / max(1, n)


def run(args, device, ttt_steps):
    """ttt_domain: "test" 在 test 折无标签上 TTT (主实验); "val" 在 val 折无标签上 TTT
    (对照: 若 val-TTT 同样提升, 说明是通用 SSL 微调而非 test 域适配)。"""
    coupons = np.load(DATA / "meta_coupon.npy")
    labels = np.load(DATA / "meta_label.npy")
    ascans = np.load(DATA / "ascans.npy", mmap_mode="r")
    p1 = torch.load(P1_CKPT, map_location="cpu")
    enc_state0 = p1["encoder_state"]
    d_model = args.d_model

    folds = []
    for tc in COUPONS:
        tr, va, te = fold_splits(coupons, labels, tc, args.val_frac, args.seed)
        rest = np.concatenate([tr, va])  # 4 个训练试件 (decoder warm-up 用, 无标签)
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
        # --- 1) 初始 P1 encoder ---
        enc0 = MAEEncoder(d_model=d_model, dropout=0.2).to(device)
        enc0.load_state_dict(enc_state0)
        model = SSLClassifier(enc0, d_model=d_model, freeze_encoder=True).to(device)
        fit, head_state = train_fold(model, train_ds, val_ds, device, epochs=args.epochs,
                                     lr=args.lr, wd=args.wd, batch_size=args.batch,
                                     seed=args.seed, patience=args.patience)

        # --- 2) no-TTT 对照 (同一 head, 初始编码器) ---
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 num_workers=4, pin_memory=True)
        s_base = scores_of(model, test_loader, device)
        yt = labels[te]
        auc_base = float(roc_auc_score(yt, s_base)) if len(np.unique(yt)) == 2 else float("nan")

        # --- 3) decoder warm-up (只训 decoder, 训练试件无标签数据) ---
        set_seed(args.seed)
        mae = build_mae(enc_state0, d_model, args.mask_ratio, args.noise_std).to(device)
        for p in mae.encoder.parameters():
            p.requires_grad_(False)
        opt_w = torch.optim.AdamW(mae.decoder.parameters(), lr=args.warm_lr, weight_decay=1e-4)
        X_src = (np.array(ascans[rest], dtype=np.float32) - ts_mean) / ts_std
        src_loader = DataLoader(TensorDataset(torch.from_numpy(X_src)), batch_size=args.ttt_batch,
                                shuffle=True, drop_last=False)
        it_w = iter(src_loader)
        mae.train()
        for _ in range(args.warm_steps):
            try:
                (x,) = next(it_w)
            except StopIteration:
                it_w = iter(src_loader)
                (x,) = next(it_w)
            x = x.to(device).unsqueeze(1)
            opt_w.zero_grad(set_to_none=True)
            recon, target, mask = mae(x)
            loss = mae.recon_loss(recon, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.decoder.parameters(), 1.0)
            opt_w.step()
        dec_warmed = {k: v.detach().cpu().clone() for k, v in mae.decoder.state_dict().items()}
        r_src = recon_loss_on(mae, X_src, device)  # decoder warm-up 后的源域 recon (检查 decoder 是否称职)
        for p in mae.encoder.parameters():
            p.requires_grad_(True)

        # --- 4) TTT: 在指定域的无标签 bscan 上微调 encoder ---
        if args.ttt_domain == "test":
            ttt_idx, ttt_name = te, "test"
        else:  # "val" 对照: 用 val 折 (源域) 无标签做 TTT
            ttt_idx, ttt_name = va, "val"
        X_ttt = (np.array(ascans[ttt_idx], dtype=np.float32) - ts_mean) / ts_std
        enc_adapted, r0, r1 = ssl_ttt(
            enc_state0, X_ttt, d_model=d_model, mask_ratio=args.mask_ratio,
            noise_std=args.noise_std, ttt_steps=ttt_steps, ttt_batch=args.ttt_batch,
            ttt_enc_lr=args.ttt_enc_lr, ttt_dec_lr=args.ttt_dec_lr,
            seed=args.seed, device=device, dec_state=dec_warmed)

        # --- 5) 评估: TTT 后编码器 + 已训 head ---
        X_te = (np.array(ascans[te], dtype=np.float32) - ts_mean) / ts_std
        shift = feature_shift(enc0, enc_adapted, X_te, device)
        model2 = SSLClassifier(enc_adapted, d_model=d_model, freeze_encoder=True).to(device)
        model2.head.load_state_dict({k: v.to(device) for k, v in head_state.items()})
        model2.eval()
        s_ttt = scores_of(model2, test_loader, device)
        auc_ttt = float(roc_auc_score(yt, s_ttt)) if len(np.unique(yt)) == 2 else float("nan")

        folds.append({"test_coupon": tc, "auc_base": auc_base, "auc_ttt": auc_ttt,
                      "delta": auc_ttt - auc_base if not np.isnan(auc_ttt) else float("nan"),
                      "n_pos": int(yt.sum()), "defect_rate": float(yt.mean()),
                      "val_auc": fit["best_val_auc"], "epochs_run": fit["epochs_run"],
                      "recon_src_warm": r_src,
                      "recon_test_before": r0, "recon_test_after": r1,
                      "recon_drop": r0 - r1, "feat_shift": shift,
                      "scores_base": s_base.tolist(), "scores_ttt": s_ttt.tolist(),
                      "wall_s": round(time.time() - t0, 1)})
        print(f"  [P5d ttt={ttt_steps}] test={tc}: base={auc_base:.3f} ttt={auc_ttt:.3f} "
              f"Δ={auc_ttt - auc_base:+.3f} | val={fit['best_val_auc']:.3f} | "
              f"recon {r0:.1f}->{r1:.1f} | shift={shift:.4f} ({round(time.time()-t0,1)}s)",
              flush=True)

    def mean4(key):
        return float(np.mean([f[key] for f in folds if f["test_coupon"] in NP4]))

    def pooled(key):
        sc, lb = [], []
        for f in folds:
            if f["test_coupon"] in NP4:
                m = coupons == f["test_coupon"]
                sc.append(np.array(f[f"scores_{key}"])); lb.append(labels[m])
        return float(roc_auc_score(np.concatenate(lb), np.concatenate(sc)))

    return {"exp": "p5d_ttt", "seed": args.seed, "ttt_steps": ttt_steps,
            "ttt_enc_lr": args.ttt_enc_lr, "ttt_dec_lr": args.ttt_dec_lr,
            "ttt_domain": args.ttt_domain,
            "nonPP4_fold_mean_base": mean4("auc_base"),
            "nonPP4_fold_mean_ttt": mean4("auc_ttt"),
            "nonPP4_delta_mean": mean4("delta"),
            "nonPP4_pooled_base": pooled("base"),
            "nonPP4_pooled_ttt": pooled("ttt"),
            "folds": [{k: v for k, v in f.items() if not k.startswith("scores_")} for f in folds]}


def parse_args():
    ap = argparse.ArgumentParser()
    # head 协议: 默认 P1 同超参 (lr=1e-3, epochs=80) == P4a 规范 baseline (0.579±0.007);
    # --head-lr 5e-4 --head-epochs 40 == P5/P5b LOOCV 脚本协议 (~0.51-0.53)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr", type=float, default=None,
                    help="覆盖 --lr (head 学习率), 兼容 P5/P5b 的 5e-4")
    ap.add_argument("--head-epochs", type=int, default=None,
                    help="覆盖 --epochs (head epochs), 兼容 P5/P5b 的 40")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--warm-steps", type=int, default=600)
    ap.add_argument("--warm-lr", type=float, default=1e-3)
    ap.add_argument("--ttt-steps", type=str, default="30", help="逗号分隔: 主预算 + 灵敏度")
    ap.add_argument("--ttt-batch", type=int, default=256)
    ap.add_argument("--ttt-enc-lr", type=float, default=1e-4)
    ap.add_argument("--ttt-dec-lr", type=float, default=1e-3)
    ap.add_argument("--ttt-domain", type=str, default="test", choices=["test", "val"],
                    help="test=主实验(在 test 折无标签上 TTT); val=对照(在 val 源域上 TTT)")
    ap.add_argument("--tag", type=str, default="", help="结果文件后缀, 如 weak/canonical (区分 head 协议)")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.head_lr is not None:
        args.lr = args.head_lr
    if args.head_epochs is not None:
        args.epochs = args.head_epochs
    if args.smoke:
        args.epochs = 2
        args.warm_steps = 20
    steps = [int(s) for s in args.ttt_steps.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed} ttt_steps={steps} domain={args.ttt_domain} "
          f"ttt_enc_lr={args.ttt_enc_lr} ttt_dec_lr={args.ttt_dec_lr} warm={args.warm_steps}")
    for st in steps:
        out = run(args, device, st)
        tag = f"ttt{st}_{args.ttt_domain}"
        if args.tag:
            tag += f"_{args.tag}"
        with open(RES / f"paut_p5d_{tag}_s{args.seed}_full.json", "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"[P5d ttt={st} domain={args.ttt_domain} seed={args.seed} tag={args.tag}] "
              f"nonPP4 逐折均值: base={out['nonPP4_fold_mean_base']:.3f} "
              f"ttt={out['nonPP4_fold_mean_ttt']:.3f} Δ={out['nonPP4_delta_mean']:+.3f} | "
              f"pooled base={out['nonPP4_pooled_base']:.3f} ttt={out['nonPP4_pooled_ttt']:.3f}")


if __name__ == "__main__":
    main()
