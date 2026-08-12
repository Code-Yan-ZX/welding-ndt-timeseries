#!/usr/bin/env python3
"""P4a: SSL 表征变体 LOOCV —— 打破"冻结 SSL 0.57-0.60"天花板的小步实验。

复用 P1 SSL 预训练编码器 (experiments/runs/ssl_ae/encoder.pt, d_model=128),
评测与 P1 同口径: 5 折 LOOCV, 非PP4 AUC (逐折均值 + pooled), 每折单独报告。

变体:
  baseline : 冻结编码器 + MLP 头 (复现 P1, 应 ≈0.572 逐折均值)
  finetune : 全参数微调 (H4) —— 编码器+头, 低 LR, 直击"冻结 SSL 欠表达" (oracle ~0.68)
  typehead : 冻结编码器 + 二分类头 + 缺陷类型头 (H2) —— 多任务强制学回波形态
  tta      : baseline 头训练后, TENT 式在 test 折未标注数据上最小化熵 (H3) —— 试件级适配

仅用 P0-P3 已有资产, 不修改其代码; 结果存新文件。
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from wndt.data.paut_dataset import PAUTSeriesDataset, make_weighted_sampler  # noqa: E402
from wndt.models.ssl_ae import MAEEncoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

DATA = REPO / "data/processed/paut"
RES = REPO / "experiments/results"
SSLCKPT = REPO / "experiments/runs/ssl_ae/encoder.pt"
COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NP4 = ["PP3", "PP5", "PP6", "PP7"]


class SSLTypeClassifier(nn.Module):
    """冻结/可训 SSL 编码器 + 二分类头 + 缺陷类型头 (0=clean, 1..6)。"""

    def __init__(self, encoder, d_model=128, n_types=7, freeze_encoder=True,
                 dropout=0.3):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 2),
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_types),
        )

    def forward(self, x, want_type=False):
        z = self.encoder(x)
        out = self.head(z)
        if want_type:
            return out, self.type_head(z)
        return out


class PAUTTypeDataset(PAUTSeriesDataset):
    """返回 (x, [label, type]) 以支持类型多任务; 采样权重仍用二分类 label。"""

    def __init__(self, *a, types_all=None, **kw):
        super().__init__(*a, **kw)
        self.type_labels = np.asarray(types_all[self.indices])

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, torch.stack([y, torch.tensor(self.type_labels[i],
                                               dtype=torch.long)])


def fold_splits(coupons, labels, test_coupon, val_frac=0.15, seed=42):
    test_idx = np.nonzero(coupons == test_coupon)[0].astype(np.int64)
    rest = np.nonzero(coupons != test_coupon)[0].astype(np.int64)
    y = labels[rest]
    strat = y if (np.bincount(y, minlength=2) >= 2).all() else None
    tr, va = train_test_split(rest, test_size=val_frac, random_state=seed,
                              shuffle=True, stratify=strat)
    return np.sort(tr), np.sort(va), np.sort(test_idx)


def fold_norm(ascans, train_idx):
    tr = np.array(ascans[train_idx], dtype=np.float32).reshape(-1, ascans.shape[-1])
    return tr.mean(0).astype(np.float32), (tr.std(0) + 1e-8).astype(np.float32)


def scores_of(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            logits = model(x)
            out.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_fold(model, train_ds, val_ds, device, *, epochs, lr, wd, batch_size,
               seed, patience=20, type_loss=False, enc_lr_mult=1.0):
    """训练折叠模型(冻结或微调编码器), val AUC 早停。返回 fit_info。"""
    set_seed(seed)
    if enc_lr_mult != 1.0:
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        other = [p for p in model.parameters()
                 if p.requires_grad and p not in set(enc_params)]
        params = [
            {"params": other, "lr": lr},
            {"params": enc_params, "lr": lr * enc_lr_mult},
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True,
                              sampler=make_weighted_sampler(train_ds.labels))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            y = y.to(device)
            opt.zero_grad()
            if type_loss:
                # y 为 (label, type) 元组? 简化: typehead 变体由外部构造 dataset
                logits, tlogits = model(x, want_type=True)
                loss = F.cross_entropy(logits, y[:, 0])
                loss = loss + 0.5 * F.cross_entropy(tlogits, y[:, 1])
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += loss.item()
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
    return {"best_val_auc": best_auc, "epochs_run": ep + 1}


def run_exp(exp, args, device):
    coupons = np.load(DATA / "meta_coupon.npy")
    labels = np.load(DATA / "meta_label.npy")
    types = np.load(DATA / "meta_defect_type.npy")
    ascans = np.load(DATA / "ascans.npy", mmap_mode="r")

    augment = None
    if args.augment:
        augment = {"beam_dropout": {"prob": 0.5, "max_drop_frac": 0.15}}

    folds = []
    for tc in COUPONS:
        tr, va, te = fold_splits(coupons, labels, tc, args.val_frac, args.seed)
        ts_mean, ts_std = fold_norm(ascans, tr)
        t0 = time.time()

        if exp == "typehead":
            # 类型监督: y = [label, type]
            train_ds = PAUTTypeDataset(DATA, tr, beam="bscan",
                                       norm_mode="per_timestep",
                                       ts_mean=ts_mean, ts_std=ts_std,
                                       types_all=types, augment=augment)
            val_ds = PAUTTypeDataset(DATA, va, beam="bscan",
                                     norm_mode="per_timestep",
                                     ts_mean=ts_mean, ts_std=ts_std,
                                     types_all=types)
            test_ds = PAUTTypeDataset(DATA, te, beam="bscan",
                                      norm_mode="per_timestep",
                                      ts_mean=ts_mean, ts_std=ts_std,
                                      types_all=types)
            enc = MAEEncoder(d_model=args.d_model).to(device)
            enc.load_state_dict(torch.load(args.ckpt, map_location=device)["encoder_state"])
            model = SSLTypeClassifier(enc, args.d_model, freeze_encoder=args.freeze).to(device)
            fit = train_fold(model, train_ds, val_ds, device, epochs=args.epochs,
                             lr=args.lr, wd=args.wd, batch_size=args.batch,
                             seed=args.seed, type_loss=True,
                             enc_lr_mult=args.enc_lr_mult)
        else:
            train_ds = PAUTSeriesDataset(DATA, tr, beam="bscan",
                                         norm_mode="per_timestep",
                                         ts_mean=ts_mean, ts_std=ts_std,
                                         augment=augment)
            val_ds = PAUTSeriesDataset(DATA, va, beam="bscan",
                                       norm_mode="per_timestep",
                                       ts_mean=ts_mean, ts_std=ts_std)
            test_ds = PAUTSeriesDataset(DATA, te, beam="bscan",
                                        norm_mode="per_timestep",
                                        ts_mean=ts_mean, ts_std=ts_std)
            enc = MAEEncoder(d_model=args.d_model).to(device)
            enc.load_state_dict(torch.load(args.ckpt, map_location=device)["encoder_state"])
            from wndt.models.ssl_ae import SSLClassifier
            model = SSLClassifier(enc, args.d_model,
                                  freeze_encoder=(exp in ("baseline", "tta"))).to(device)
            fit = train_fold(model, train_ds, val_ds, device, epochs=args.epochs,
                             lr=args.lr, wd=args.wd, batch_size=args.batch,
                             seed=args.seed, enc_lr_mult=args.enc_lr_mult)

        # H3: TTA —— 在 test 折未标注数据上做测试时适配 (不改权重/重算BN/熵最小化)
        if exp == "tta":
            tloader = DataLoader(test_ds, batch_size=32, shuffle=True,
                                 num_workers=2, pin_memory=True)
            if args.tta_mode == "bn":
                # BN 统计量重校准: 前向收集 running stats (权重不动, 头保留)
                model.eval()
                was_training_buffers = []
                for m in model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.reset_running_stats()
                        was_training_buffers.append(True)
                # 用 train 统计初始化? 直接收集 test 统计
                model.train()
                with torch.no_grad():
                    for x, _ in tloader:
                        x = x.to(device)
                        if x.dim() == 3:
                            x = x.unsqueeze(1)
                        model(x)
                model.eval()
            elif args.tta_mode == "tent":
                # TENT: 熵最小化, 只更新可训参数 (head)
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.SGD(params, lr=1e-3)
                model.train()
                for _ in range(args.tta_steps):
                    for x, _ in tloader:
                        x = x.to(device)
                        if x.dim() == 3:
                            x = x.unsqueeze(1)
                        opt.zero_grad()
                        logits = model(x)
                        p = F.softmax(logits, -1)
                        loss = -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean()
                        loss.backward()
                        opt.step()
                model.eval()
            else:
                raise ValueError(f"unknown tta_mode {args.tta_mode}")

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 num_workers=4, pin_memory=True)
        scores = scores_of(model, test_loader, device)
        yt = labels[te]
        auc = float(roc_auc_score(yt, scores)) if len(np.unique(yt)) == 2 else float("nan")
        folds.append({"test_coupon": tc, "auc": auc, "n_pos": int(yt.sum()),
                      "defect_rate": float(yt.mean()), "wall_s": round(time.time() - t0, 1),
                      "val_auc": fit["best_val_auc"], "scores": scores.tolist()})
        print(f"  [{exp}] test={tc}: AUC={auc:.3f} val_auc={fit['best_val_auc']:.3f} "
              f"({round(time.time()-t0,1)}s)", flush=True)

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
    summary = {"exp": exp, "nonPP4_fold_mean": float(np.mean(np4)),
               "nonPP4_pooled": pooled_auc,
               "folds": [{k: v for k, v in f.items() if k != "scores"} for f in folds]}
    print(f"[{exp}] nonPP4 逐折均值={np.mean(np4):.3f}  pooled={pooled_auc:.3f}")
    out = RES / f"paut_p4a_{exp}.json"
    # 存分数到单独文件, 主 json 只存摘要
    full = dict(summary)
    full["folds"] = folds
    with open(RES / f"paut_p4a_{exp}_full.json", "w") as fh:
        json.dump(full, fh, indent=2, ensure_ascii=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["baseline", "finetune", "typehead", "tta"])
    ap.add_argument("--folds", type=str, default="all", help="all 或逗号分隔子集, 如 PP3,PP7")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--freeze", type=int, default=1, help="typehead 是否冻结编码器")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta-steps", type=int, default=20)
    ap.add_argument("--enc-lr-mult", type=float, default=1.0,
                    help="编码器 LR = lr * mult (微调用 0.1)")
    ap.add_argument("--tta-mode", type=str, default="tent",
                    choices=["tent", "bn"])
    ap.add_argument("--augment", action="store_true",
                    help="训练集加 beam_dropout 增强")
    ap.add_argument("--ckpt", type=str, default=str(SSLCKPT),
                    help="SSL 预训练编码器路径")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} exp={args.exp}")
    if args.folds != "all":
        global COUPONS, NP4
        keep = args.folds.split(",")
        COUPONS = [c for c in COUPONS if c in keep]
        NP4 = [c for c in NP4 if c in keep]
    run_exp(args.exp, args, device)


if __name__ == "__main__":
    main()
