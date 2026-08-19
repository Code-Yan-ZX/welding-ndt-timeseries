#!/usr/bin/env python3
"""P7 (Synth-UT → Real): 物理启发程序化超声 B-scan 预训练, 真实 PAUT LOOCV 规范头评估。

Protocol V2 (docs/M0_evaluation_protocol_v2.md) 修正 (M0-1.5):

  --protocol strict_inductive (默认):
      test coupon 的一切信息 (信号/统计量/无标签数据) 都不得进入预训练、
      normalization、validation 或模型选择。
      - 纯合成预训练: 合成 coupon 与真实 test coupon 无关, 预训练只跑一次
        即满足 strict (合成数据不含任何真实 test 信号);
      - --mix-real: 真实部分必须排除 test coupon —— 预训练在**每个 outer fold
        内**执行 (5 个 fold = 5 次预训练), 每次只用非 test coupon 真实数据。
      normalization 只在 train coupons 上计算。

  --protocol transductive_unlabeled (仅诊断/单独报告):
      允许使用 test coupon 的**无标签**信号参与预训练 (一次性预训练,
      含全部真实数据)。**不得**与 strict 结果合并成同一主指标,
      报告必须显式标注。

  Validation 按**完整 coupon** 分组 (每折取 1 个非 test coupon 作 val),
  禁止随机位置级 validation。

  每个结果 JSON 记录: protocol / pretrain_coupons / train_coupons /
  val_coupons / test_coupon / normalization_scope / seed / code_commit /
  run_type (smoke/full)。smoke 输出带 _smoke 后缀, 不覆盖 _full。

对照基线 (P4a): 真实数据预训练 → 真实 LOOCV = 0.579±0.007
P6 batch=128 真实预训练基线: 0.556

Usage:
  python scripts/synth_ultrasound.py --n-coupons 12 --n-pos-per-coupon 1000 --seed 42
  python scripts/paut_p7_synth_to_real.py --pretrain-epochs 40 --seed 42
  # 混合 (strict, 每 fold 内预训练)
  python scripts/paut_p7_synth_to_real.py --mix-real --pretrain-epochs 60 --seed 42
  # transductive 探索 (单独报告, 非严格跨试件结论)
  python scripts/paut_p7_synth_to_real.py --protocol transductive_unlabeled --mix-real --seed 42
  python scripts/paut_p7_synth_to_real.py --smoke   # 冒烟: 输出 _smoke.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


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


def load_real():
    ascans = np.load(REAL_DIR / "ascans.npy")           # (3000, 49, 512)
    coupons = np.load(REAL_DIR / "meta_coupon.npy")     # str: 'PP3'..'PP7'
    labels = np.load(REAL_DIR / "meta_label.npy")
    return ascans, np.asarray(coupons), labels


def coupon_val_split(rest_coupons, seed: int):
    """按完整 coupon 切 val: 取 1 个非 test coupon 作 val, 其余作 train。

    Protocol V2 §3: 禁止随机位置级 validation —— val 必须是完整 coupon。
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(list(rest_coupons)).tolist()
    val_coupon = shuffled[0]
    train_coupons = sorted(shuffled[1:])
    return train_coupons, [val_coupon]


def loocv_real(enc: nn.Module, *, d_model: int, epochs: int, lr: float,
               batch_size: int, seed: int, device: torch.device,
               ascans, coupons_str, labels) -> list[dict]:
    """真实 PAUT 5 折 LOOCV by PP coupon; val 按完整 coupon 分组。"""
    results = []
    for tc in COUPONS:
        te = np.nonzero(coupons_str == tc)[0]
        rest_coupons = [c for c in COUPONS if c != tc]
        train_coupons, val_coupons = coupon_val_split(rest_coupons, seed)
        tr = np.nonzero(np.isin(coupons_str, train_coupons))[0]
        va = np.nonzero(np.isin(coupons_str, val_coupons))[0]
        # normalization 只在 train coupons 上计算 (strict; transductive 也保持
        # 以 train 为准, 避免污染 val/head 选择)
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
                        "train_coupons": train_coupons,
                        "val_coupons": val_coupons,
                        "def_rate": round(def_rate, 3),
                        "val_auc": round(val_auc, 4),
                        "test_auc": round(test_auc, 4)})
        print(f"  fold {tc} n={len(te):4d} def_rate={def_rate:.2f} "
              f"| val(coupons {val_coupons}) {val_auc:.4f} | test {test_auc:.4f}")
    return results


def run_fold_pretrain(synth_X, real_X, real_coupons, test_coupon, *,
                      mix_real: bool, args, device, fold_dir: Path) -> nn.Module:
    """一次预训练, 返回冻结编码器。

    strict_inductive + mix_real: 真实部分排除 test_coupon —— 每 fold 重训;
    strict_inductive 纯合成 或 transductive: 一次预训练 (合成无 test 信号 /
     transductive 允许 test 无标签)。
    """
    X = synth_X
    if mix_real:
        keep = np.nonzero(real_coupons != test_coupon)[0] if args.protocol == "strict_inductive" \
            else np.arange(len(real_coupons))
        X = np.concatenate([X, real_X[keep]], axis=0)
    enc_path = pretrain(X, epochs=args.pretrain_epochs, batch_size=args.pretrain_batch,
                        d_model=args.d_model, mask_ratio=args.mask_ratio,
                        lr=args.lr_pretrain, seed=args.seed, device=device,
                        out_dir=fold_dir)
    enc = MAEEncoder(d_model=args.d_model).to(device)
    enc.load_state_dict(torch.load(enc_path, map_location=device,
                                   weights_only=True)["encoder_state"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc, int(X.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--out", type=Path, default=RUN_DIR)
    ap.add_argument("--protocol", choices=["strict_inductive", "transductive_unlabeled"],
                    default="strict_inductive",
                    help="strict_inductive=主协议 (test coupon 不进预训练/统计/选择); "
                         "transductive_unlabeled=允许 test 无标签信号, 仅诊断单独报告")
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
                    help="混合: 合成+真实联合预训练。strict 下每 fold 内重训并排除 test coupon")
    ap.add_argument("--skip-pretrain", action="store_true",
                    help="跳过预训练, 直接加载已有 encoder.pt (仅限非 mix strict 或 transductive)")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 小样本/少 epoch, 输出 _smoke.json (不覆盖 _full)")
    args = ap.parse_args()

    run_type = "smoke" if args.smoke else "full"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    code_commit = git_commit()
    code_dirty = git_dirty()

    # 加载合成数据
    Xs = np.load(args.synth_dir / "ascans.npy")
    print(f"P7[Synth→Real] 合成数据: Xs={Xs.shape} | protocol={args.protocol}")
    if args.smoke:
        idx = np.linspace(0, len(Xs) - 1, 1024).astype(int)
        Xs = Xs[idx]
        args.pretrain_epochs = 2; args.head_epochs = 4

    # 加载真实数据 (评估用)
    ascans, coupons_str, labels = load_real()

    if args.mix_real:
        Xr = ascans
        print(f"  + 真实数据: Xr={Xr.shape}")

    if args.protocol == "strict_inductive" and args.mix_real and not args.smoke:
        # strict + mix: 每 fold 内重训 (排除 test coupon 真实数据)
        fold_dirs = []
        folds = []
        pretrain_nsamples = []
        for tc in COUPONS:
            print(f"\n=== strict fold {tc}: 预训练 (真实排除 {tc}) ===")
            fold_dir = args.out / f"fold_{tc}_strict"
            enc, n = run_fold_pretrain(Xs, ascans, coupons_str, tc, mix_real=True,
                                       args=args, device=device, fold_dir=fold_dir)
            fold_dirs.append(str(fold_dir)); pretrain_nsamples.append(n)
            fold_res = loocv_real(enc, d_model=args.d_model, epochs=args.head_epochs,
                                  lr=args.lr_head, batch_size=args.head_batch,
                                  seed=args.seed, device=device,
                                  ascans=ascans, coupons_str=coupons_str, labels=labels)
            # 只保留本折 (tc) 的结果
            folds.append([r for r in fold_res if r["test_coupon"] == tc][0])
        folds = [dict(f, pretrain_coupons=[c for c in COUPONS if c != f["test_coupon"]],
                      pretrain_n_samples=pretrain_nsamples[i]) for i, f in enumerate(folds)]
    else:
        # 纯合成 strict 或 transductive: 一次预训练
        if args.mix_real:
            # transductive_unlabeled + mix: 全部真实数据 (含 test 无标签) 参与预训练
            X = np.concatenate([Xs, Xr], axis=0)
            print(f"  [transductive] 联合 {X.shape[0]} 样本 (含 test 无标签) 一次性预训练")
        else:
            X = Xs
        if args.smoke and args.mix_real:
            X = np.concatenate([Xs, Xr[:256]], axis=0)
        enc, _ = run_fold_pretrain(X, np.empty((0,)), np.array([]), None,
                                   mix_real=False, args=args, device=device,
                                   fold_dir=args.out)
        folds = loocv_real(enc, d_model=args.d_model, epochs=args.head_epochs,
                           lr=args.lr_head, batch_size=args.head_batch,
                           seed=args.seed, device=device,
                           ascans=ascans, coupons_str=coupons_str, labels=labels)
        for f in folds:
            f["pretrain_coupons"] = (
                list(COUPONS) if args.protocol == "transductive_unlabeled"
                else [])  # 纯合成: 真实 pretain coupons 为空
            f["pretrain_n_samples"] = int(X.shape[0])

    all_aucs = [f["test_auc"] for f in folds]
    np4_aucs = [f["test_auc"] for f in folds if f["test_coupon"] in NP4]
    summary = {
        "exp": "synth_to_real_p7_mix" if args.mix_real else "synth_to_real_p7",
        "protocol": args.protocol,
        "run_type": run_type,
        "seed": args.seed,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "normalization_scope": "train_coupons",
        "pretrain": {"n_samples": int(sum(f.get("pretrain_n_samples", 0) for f in folds) // max(1, len(folds))),
                     "epochs": args.pretrain_epochs, "batch": args.pretrain_batch,
                     "lr": args.lr_pretrain, "mask_ratio": args.mask_ratio,
                     "d_model": args.d_model, "mix_real": args.mix_real,
                     "per_fold_repretrain": bool(args.protocol == "strict_inductive" and args.mix_real)},
        "head": {"epochs": args.head_epochs, "batch": args.head_batch, "lr": args.lr_head},
        "all_folds_mean_auc": float(np.mean(all_aucs)),
        "all_folds_std_auc": float(np.std(all_aucs)),
        "nonPP4_mean_auc": float(np.mean(np4_aucs)),
        "nonPP4_std_auc": float(np.std(np4_aucs)),
        "folds": folds,
    }
    print(f"\n[{args.protocol} {run_type}] 主指标 | 真实 PAUT 全 5 折 test AUC: "
          f"mean={np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f} | {all_aucs}")
    print(f"[{args.protocol} {run_type}] 主指标 | 非PP4 4 折 (P4a 同口径): "
          f"mean={np.mean(np4_aucs):.4f} ± {np.std(np4_aucs):.4f} | {np4_aucs}")
    print(f"对照基线: P4a 真实预训练非PP4=0.579±0.007 | P6 base 0.556")
    suffix = "_smoke" if run_type == "smoke" else "_full"
    out_json = REPO / "experiments/results" / (
        f"paut_p7_synth_to_real{'_mix' if args.mix_real else ''}"
        f"_{args.protocol}_s{args.seed}{suffix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
