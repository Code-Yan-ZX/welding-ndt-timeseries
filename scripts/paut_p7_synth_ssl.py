#!/usr/bin/env python3
"""P7 (Synth-UT): 程序生成超声 B-scan 数据上 P1 风格 SSL 预训练 + 规范头 5 折 LOOCV 评估。

动机 (2026-08-18, 新方向探索): PAUT 真实数据天花板证伪在表征级 0.58, 当前 5 试件
/ 3000 位置规模太小, 必须靠外部数据源扩大预训练规模。本脚本跑一个端到端验证:
  Step 1: 程序生成 (scripts/synth_ultrasound.py) 的合成 B-scan (12000 样本,
          12 试件, 缺陷率 0.04-0.74 模仿真实 PP3-PP7)
  Step 2: MaskedAE 30% 掩码波束重建预训练 (P1 base 风格, 同 P6 --variant base)
  Step 3: 冻结编码器 + 规范头 (lr=1e-3/80ep/batch=128) 5 折 LOOCV by coupon
  Step 4: 报 全5折均值 + 剔除近零缺陷折 (def_rate<0.05) 均值 (后者对应 P4a 非PP4 口径)

目的: 验证 (a) 合成数据可学, (b) 评估管线跑通, (c) 纯合成预训练基线水平。
后续实验: (i) 真实+合成混合预训练对比, (ii) 换 CIVA/真仿真数据, (iii) 更大编码器。

Usage:
  python scripts/synth_ultrasound.py --n-coupons 12 --n-pos-per-coupon 1000 --seed 42
  python scripts/paut_p7_synth_ssl.py --epochs 40 --seed 42
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from wndt.models.ssl_ae import MaskedAE  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

SYNTH_DIR = REPO / "data/processed/synth_ut"
RUN_DIR = REPO / "experiments/runs/synth_ssl_p7"


def pretrain(X: np.ndarray, *, epochs: int, batch_size: int, d_model: int,
             mask_ratio: float, lr: float, seed: int, device: torch.device,
             out_dir: Path) -> Path:
    """P1 base 风格 SSL 预训练: 30% 掩码波束重建, Hubber loss。"""
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    Xn = np.ascontiguousarray(X.astype(np.float32))
    ds = TensorDataset(torch.from_numpy(Xn))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
    model = MaskedAE(d_model=d_model, mask_ratio=mask_ratio, noise_std=0.02).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(loader) * epochs

    def lr_at(step):
        warm = 200
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    log = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            recon, target, mask = model(x)
            loss = model.recon_loss(recon, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0); step += 1
        rec = {"epoch": epoch, "loss": tot / max(1, n), "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  pretrain epoch {epoch:3d} | recon {rec['loss']:.5f} | lr {rec['lr']:.2e}")
    enc_path = out_dir / "encoder.pt"
    torch.save({"encoder_state": model.encoder.state_dict(),
                "d_model": d_model, "mask_ratio": mask_ratio,
                "seed": seed, "epochs": epochs,
                "n_samples": int(X.shape[0]),
                "wall_s": round(time.time() - t0, 1)}, enc_path)
    with open(out_dir / "pretrain_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"  -> 预训练完成 | {epochs}ep | {time.time()-t0:.0f}s | {enc_path}")
    return enc_path


def fit_head(enc: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
             X_va: np.ndarray, y_va: np.ndarray, *,
             d_model: int, epochs: int, lr: float, batch_size: int,
             seed: int, device: torch.device) -> nn.Module:
    """冻结编码器 + 规范头 (lr=1e-3/80ep), val AUC 早停 (patience=20)。"""
    from wndt.models.ssl_ae import MAEEncoder  # noqa: F401
    head = nn.Sequential(
        nn.LayerNorm(d_model), nn.Dropout(0.3),
        nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(d_model, 2),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False)
    enc.eval()
    best_auc, best_state, patience = -1, None, 0
    set_seed(seed)
    for epoch in range(epochs):
        head.train()
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            with torch.no_grad():
                z = enc(x)
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # val
        head.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device)
                y = y.to(device)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                with torch.no_grad():
                    z = enc(x)
                logits = head(z)
                scores.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
                ys.append(y.cpu().numpy())
        scores = np.concatenate(scores); ys = np.concatenate(ys)
        if ys.min() == ys.max():
            auc = 0.5
        else:
            auc = roc_auc_score(ys, scores)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience > 20:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, best_auc


def evaluate(enc: nn.Module, head: nn.Module, X: np.ndarray, y: np.ndarray,
             device: torch.device, batch_size: int = 256) -> float:
    head.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, ys = [], []
    with torch.no_grad():
        for x, yy in loader:
            x = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            z = enc(x)
            logits = head(z)
            scores.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
            ys.append(yy.numpy())
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    if ys.min() == ys.max():
        return 0.5
    return float(roc_auc_score(ys, scores))


def loocv(enc: nn.Module, X: np.ndarray, y: np.ndarray, coupons: np.ndarray,
          *, d_model: int, epochs: int, lr: float, batch_size: int, seed: int,
          device: torch.device, exclude_near_zero: bool = False) -> list[dict]:
    """5 折 LOOCV by coupon。每折: 余 n-1 试件 → 15% val + 剩余 train → norm on train。"""
    from wndt.models.ssl_ae import MAEEncoder  # noqa: F401
    uniq = np.unique(coupons)
    results = []
    for tc in uniq:
        te_idx = np.nonzero(coupons == tc)[0]
        rest = np.nonzero(coupons != tc)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(rest)
        n_val = max(1, int(0.15 * len(rest)))
        va_idx = np.sort(rest[:n_val])
        tr_idx = np.sort(rest[n_val:])
        # norm on train only
        mean = X[tr_idx].mean(axis=0).astype(np.float32)
        std = (X[tr_idx].std(axis=0) + 1e-8).astype(np.float32)
        def norm(i): return (X[i] - mean) / std
        X_tr, y_tr = norm(tr_idx), y[tr_idx]
        X_va, y_va = norm(va_idx), y[va_idx]
        X_te, y_te = norm(te_idx), y[te_idx]
        head, val_auc = fit_head(enc, X_tr, y_tr, X_va, y_va,
                                 d_model=d_model, epochs=epochs, lr=lr,
                                 batch_size=batch_size, seed=seed, device=device)
        test_auc = evaluate(enc, head, X_te, y_te, device)
        def_rate = float(y[te_idx].mean())
        results.append({"test_coupon": int(tc), "n_test": int(len(te_idx)),
                        "def_rate": round(def_rate, 3),
                        "val_auc": round(val_auc, 4), "test_auc": round(test_auc, 4)})
        print(f"  fold coupon={int(tc):2d} n={len(te_idx):4d} def_rate={def_rate:.2f} "
              f"| val {val_auc:.4f} | test {test_auc:.4f}")
    if exclude_near_zero:
        kept = [r for r in results if r["def_rate"] >= 0.05]
    else:
        kept = results
    return results, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--out", type=Path, default=RUN_DIR)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--pretrain-batch", type=int, default=256)
    ap.add_argument("--head-epochs", type=int, default=80)
    ap.add_argument("--head-batch", type=int, default=128)
    ap.add_argument("--lr-pretrain", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    # 加载合成数据
    X = np.load(args.synth_dir / "ascans.npy")          # (N,49,512)
    y = np.load(args.synth_dir / "labels.npy")          # (N,)
    coupons = np.load(args.synth_dir / "coupons.npy")    # (N,)
    print(f"P7[Synth-UT] data: X={X.shape} | def_rate={y.mean():.3f} | "
          f"n_coupons={len(np.unique(coupons))}")
    if args.smoke:
        idx = np.linspace(0, len(X) - 1, 1024).astype(int)
        X = X[idx]; y = y[idx]; coupons = coupons[idx]
        args.pretrain_epochs = 2; args.head_epochs = 4

    # Step 1: 预训练
    enc_path = pretrain(X, epochs=args.pretrain_epochs, batch_size=args.pretrain_batch,
                        d_model=args.d_model, mask_ratio=args.mask_ratio,
                        lr=args.lr_pretrain, seed=args.seed, device=device,
                        out_dir=args.out)

    # Step 2: 加载冻结编码器
    from wndt.models.ssl_ae import MAEEncoder
    enc = MAEEncoder(d_model=args.d_model).to(device)
    enc.load_state_dict(torch.load(enc_path, map_location=device)["encoder_state"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    # Step 3: 5 折 LOOCV
    print("\nP7 5 折 LOOCV (按 coupon):")
    all_folds, non_trivial = loocv(enc, X, y, coupons, d_model=args.d_model,
                                    epochs=args.head_epochs, lr=args.lr_head,
                                    batch_size=args.head_batch, seed=args.seed,
                                    device=device)
    aucs_all = [f["test_auc"] for f in all_folds]
    aucs_nt = [f["test_auc"] for f in non_trivial]
    print(f"\n主指标 | 全 {len(aucs_all)} 折 test AUC: "
          f"mean={np.mean(aucs_all):.4f} ± {np.std(aucs_all):.4f} | {aucs_all}")
    print(f"副指标 | 非近零缺陷折 (def_rate>=0.05, {len(aucs_nt)} 折): "
          f"mean={np.mean(aucs_nt):.4f} ± {np.std(aucs_nt):.4f} | {aucs_nt}")

    summary = {
        "exp": "synth_ssl_p7", "seed": args.seed,
        "n_samples": int(X.shape[0]), "n_coupons": int(len(np.unique(coupons))),
        "defect_rate": float(y.mean()),
        "pretrain": {"epochs": args.pretrain_epochs, "batch": args.pretrain_batch,
                     "lr": args.lr_pretrain, "mask_ratio": args.mask_ratio,
                     "d_model": args.d_model},
        "head": {"epochs": args.head_epochs, "batch": args.head_batch, "lr": args.lr_head},
        "all_folds_mean_auc": float(np.mean(aucs_all)),
        "all_folds_std_auc": float(np.std(aucs_all)),
        "non_near_zero_folds_mean_auc": float(np.mean(aucs_nt)),
        "folds": all_folds,
    }
    out_json = REPO / "experiments/results" / f"paut_p7_synth_ssl_s{args.seed}_full.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
