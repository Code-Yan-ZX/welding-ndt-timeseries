#!/usr/bin/env python3
"""P7 (Synth-UT → Real): 程序生成超声 B-scan 预训练, 在真实 PAUT 5 折 LOOCV 规范头评估。

动机 (2026-08-18, 新方向探索): PAUT 真实数据天花板在表征级 0.58, 关键诊断
(P4-P6) = "缺陷强度与试件身份强耦合", 廉价杠杆全部证伪。P4/P6 报告点名翻盘路径
= "物理保真合成数据教缺陷回波物理"。本脚本测试这条路:
  Step 1: 程序生成 (scripts/synth_ultrasound.py) 合成 B-scan (12k-100k 样本)
  Step 2: MaskedAE 30% 掩码波束重建预训练 (P1 base 风格)
  Step 3: 冻结编码器, 在真实 PAUT 单视角数据 (3000 位置 / 5 试件) 上做规范头
          5 折 LOOCV (lr=1e-3/80ep), 报 **非PP4 逐折均值** (与 P4a baseline 0.579
          同口径可比)

P5 证伪的 "2D 高斯峰注入" vs 本工作:
  - P5 注入: 2D 高斯, 无衰减/散斑/底面/走时 → inj_acc 0.998 但下游 0.545<0.579
  - Synth-UT: 高斯包络正弦回波 + 衰减 + 散斑 + 底面 + 走时 → 物理保真

对照基线 (P4a): 真实数据预训练 → 真实 LOOCV = 0.579±0.007
P6 batch=128 真实预训练基线: 0.556
期望: 纯合成预训练若 ≥ 0.579 = 合成可替代真实 (天花板, 强证据);
      纯合成预训练若 [0.50, 0.58) = 合成可学但耦合到合成统计, 迁移有限;
      合成+真实混合预训练才是下一步。

Usage:
  python scripts/synth_ultrasound.py --n-coupons 12 --n-pos-per-coupon 1000 --seed 42
  python scripts/paut_p7_synth_to_real.py --pretrain-epochs 40 --seed 42
  # 混合模式 (推荐): 在合成+真实 联合数据上预训练, 在真实 LOOCV
  python scripts/paut_p7_synth_to_real.py --mix-real --pretrain-epochs 60 --seed 42
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

from wndt.models.ssl_ae import MAEEncoder, MaskedAE  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

SYNTH_DIR = REPO / "data/processed/synth_ut"
REAL_DIR = REPO / "data/processed/paut"
RUN_DIR = REPO / "experiments/runs/synth_to_real_p7"
COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NP4 = {"PP3", "PP5", "PP6", "PP7"}  # 排除近零缺陷试件 PP4


def pretrain(X: np.ndarray, *, epochs: int, batch_size: int, d_model: int,
             mask_ratio: float, lr: float, seed: int, device: torch.device,
             out_dir: Path) -> Path:
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
             seed: int, device: torch.device) -> tuple[nn.Module, float]:
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
        head.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device); y = y.to(device)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                with torch.no_grad():
                    z = enc(x)
                logits = head(z)
                scores.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
                ys.append(y.cpu().numpy())
        scores = np.concatenate(scores); ys = np.concatenate(ys)
        auc = 0.5 if ys.min() == ys.max() else float(roc_auc_score(ys, scores))
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
            scores.append(F.softmax(head(z), -1)[:, 1].cpu().numpy())
            ys.append(yy.numpy())
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    if ys.min() == ys.max():
        return 0.5
    return float(roc_auc_score(ys, scores))


def loocv_real(enc: nn.Module, *, d_model: int, epochs: int, lr: float,
               batch_size: int, seed: int, device: torch.device) -> list[dict]:
    """真实 PAUT 单视角 5 折 LOOCV by PP coupon, 报全折 + 非PP4 折。"""
    ascans = np.load(REAL_DIR / "ascans.npy")           # (3000, 49, 512)
    coupons = np.load(REAL_DIR / "meta_coupon.npy")     # str: 'PP3'..'PP7'
    labels = np.load(REAL_DIR / "meta_label.npy")
    coupons_str = np.asarray(coupons)
    results = []
    for tc in COUPONS:
        te = np.nonzero(coupons_str == tc)[0]
        rest = np.nonzero(coupons_str != tc)[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(rest)
        n_val = max(1, int(0.15 * len(rest)))
        va, tr = np.sort(rest[:n_val]), np.sort(rest[n_val:])
        mean = ascans[tr].mean(0).astype(np.float32)
        std = (ascans[tr].std(0) + 1e-8).astype(np.float32)
        Xtr = (ascans[tr] - mean) / std
        Xva = (ascans[va] - mean) / std
        Xte = (ascans[te] - mean) / std
        head, val_auc = fit_head(enc, Xtr, labels[tr], Xva, labels[va],
                                 d_model=d_model, epochs=epochs, lr=lr,
                                 batch_size=batch_size, seed=seed, device=device)
        test_auc = evaluate(enc, head, Xte, labels[te], device)
        def_rate = float(labels[te].mean())
        results.append({"test_coupon": tc, "n_test": int(len(te)),
                        "def_rate": round(def_rate, 3),
                        "val_auc": round(val_auc, 4),
                        "test_auc": round(test_auc, 4)})
        print(f"  fold {tc} n={len(te):4d} def_rate={def_rate:.2f} "
              f"| val {val_auc:.4f} | test {test_auc:.4f}")
    return results


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
    ap.add_argument("--mix-real", action="store_true",
                    help="混合: 合成+真实联合预训练 (推荐下一步)")
    ap.add_argument("--skip-pretrain", action="store_true",
                    help="跳过预训练, 直接加载已有 encoder.pt (默认路径 out/encoder.pt)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    # 加载合成数据
    Xs = np.load(args.synth_dir / "ascans.npy")
    print(f"P7[Synth→Real] 合成数据: Xs={Xs.shape}")
    if args.mix_real:
        Xr = np.load(REAL_DIR / "ascans.npy")
        print(f"  + 真实数据: Xr={Xr.shape} | 联合 {Xs.shape[0]+Xr.shape[0]} 样本")
        Xs = np.concatenate([Xs, Xr], axis=0)
    X = Xs
    if args.smoke:
        idx = np.linspace(0, len(X) - 1, 1024).astype(int)
        X = X[idx]
        args.pretrain_epochs = 2; args.head_epochs = 4

    # Step 1: 预训练 (合成 / 混合)
    if args.skip_pretrain:
        enc_path = args.out / "encoder.pt"
        if not enc_path.exists():
            raise FileNotFoundError(f"--skip-pretrain 但 {enc_path} 不存在, 请先跑 pretrain")
        print(f"P7 跳过预训练, 加载 {enc_path}")
    else:
        enc_path = pretrain(X, epochs=args.pretrain_epochs, batch_size=args.pretrain_batch,
                            d_model=args.d_model, mask_ratio=args.mask_ratio,
                            lr=args.lr_pretrain, seed=args.seed, device=device,
                            out_dir=args.out)

    # Step 2: 加载冻结编码器
    enc = MAEEncoder(d_model=args.d_model).to(device)
    enc.load_state_dict(torch.load(enc_path, map_location=device,
                                    weights_only=True)["encoder_state"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    # Step 3: 真实 PAUT 5 折 LOOCV
    print("\nP7 Synth→Real 5 折 LOOCV (真实 PAUT 数据):")
    folds = loocv_real(enc, d_model=args.d_model, epochs=args.head_epochs,
                       lr=args.lr_head, batch_size=args.head_batch, seed=args.seed,
                       device=device)
    all_aucs = [f["test_auc"] for f in folds]
    np4_aucs = [f["test_auc"] for f in folds if f["test_coupon"] in NP4]
    summary = {
        "exp": "synth_to_real_p7_mix" if args.mix_real else "synth_to_real_p7",
        "seed": args.seed,
        "pretrain": {"n_samples": int(X.shape[0]),
                     "epochs": args.pretrain_epochs, "batch": args.pretrain_batch,
                     "lr": args.lr_pretrain, "mask_ratio": args.mask_ratio,
                     "d_model": args.d_model, "mix_real": args.mix_real},
        "head": {"epochs": args.head_epochs, "batch": args.head_batch, "lr": args.lr_head},
        "all_folds_mean_auc": float(np.mean(all_aucs)),
        "all_folds_std_auc": float(np.std(all_aucs)),
        "nonPP4_mean_auc": float(np.mean(np4_aucs)),
        "nonPP4_std_auc": float(np.std(np4_aucs)),
        "folds": folds,
    }
    print(f"\n主指标 | 真实 PAUT 全 5 折 test AUC: mean={np.mean(all_aucs):.4f} "
          f"± {np.std(all_aucs):.4f} | {all_aucs}")
    print(f"主指标 | 非PP4 4 折 (P4a 同口径): mean={np.mean(np4_aucs):.4f} "
          f"± {np.std(np4_aucs):.4f} | {np4_aucs}")
    print(f"对照基线: P4a 真实预训练非PP4=0.579±0.007 | P6 base 0.556")
    out_json = REPO / "experiments/results" / (
        f"paut_p7_synth_to_real{'_mix' if args.mix_real else ''}_s{args.seed}_full.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
