#!/usr/bin/env python3
"""M0-2B 严格跨试件 LOOCV 评估（外部超声预训练迁移判断）。

严格 coupon-level LOOCV（Protocol V2，参考 ``scripts/paut_p7_synth_to_real.py``
的 coupon-level validation 思路，**不调用** ``paut_loocv.py`` 随机位置级
validation 的旧 ``fold_splits()``）：

- outer test：一个完整 coupon；
- inner validation：剩余 coupons 中一个完整 coupon；
- train：其余三个完整 coupons；
- 归一化只用 train coupons；目标域 SSL 只用 train coupons；
- 分类头训练只用 train coupons；validation 只用于模型选择；
- test coupon 在训练 / SSL / 统计量 / 模型选择阶段完全不可见。

四个条件（相同模型结构 / mask ratio / 优化器 / 总 optimizer steps / 头协议）：
- E0 scratch                 : 随机初始化共享 encoder，冻结，只训分类头（sanity）
- E1 target_ssl              : 每折仅在本折 train coupons 上 SSL，冻结，训头
- E2 external_ssl            : 外部（ML-NDT + NDT_ML_Flaw）预训练一次，复用于 5 折
- E3 external_then_target    : E2 外部 ckpt -> 每折继续目标域 SSL -> 冻结训头

下游头（规范协议）：mean-pooled encoder 特征 -> MLP 头（lr 1e-3 / ≤80 ep /
class-balanced / val coupon 驱动早停）。encoder 冻结，第一轮不做 fine-tune。

主指标：``non-PP4 mean AUC = mean(PP3, PP5, PP6, PP7)``；同时报告逐折 AUC、
non-PP4 std、pooled AUC（仅参考）、PP4（不纳入主均值）。

smoke 输出带 ``_smoke`` 后缀，不覆盖正式结果。

Usage:
  python scripts/m0_2b_loocv.py --exp all --seed 42
  python scripts/m0_2b_loocv.py --exp e1 --seed 42
  python scripts/m0_2b_loocv.py --exp e2 --seed 42
  python scripts/m0_2b_loocv.py --smoke            # 冒烟（20 SSL steps / 1 head ep）
  python scripts/m0_2b_loocv.py --exp combine      # 合并已生成的各条件结果
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
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa: E402

from wndt.data.ultrasound_pretrain import (  # noqa: E402
    COUPONS, NP4, load_paut, paut_fold_split, penelope_fold_stats, penelope_transform,
)
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

from m0_2b_pretrain import (  # noqa: E402
    DEFAULT_CONFIG, PretrainOpts, build_model, external_ckpt_path,
    load_checkpoint, pretrain_external, pretrain_target, target_ckpt_path,
)

RESULTS_DIR = REPO / "experiments" / "results"
EXPS = ["e0", "e1", "e2", "e3"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen(path: Path, cfg, device) -> nn.Module:
    return freeze(load_checkpoint(path, device))


# ---------------------------------------------------------------------------
# 各条件 encoder 获取（缺失时自动跑对应预训练）
# ---------------------------------------------------------------------------
def _opts(cmd, seed, steps, cfg, fold=None, init_path=None):
    return PretrainOpts(
        cmd=cmd, seed=seed, steps=steps,
        batch_size=int(cfg.pretrain.batch_size),
        steps_per_epoch=int(cfg.pretrain.steps_per_epoch),
        fold=fold, init_path=init_path,
    )


def exp_checkpoint_path(exp: str, fold: str, seed: int, cfg, smoke: bool) -> Path | None:
    """各条件下该折 encoder 的 checkpoint 路径（E0 无预训练 -> None）。"""
    p = cfg.pretrain
    if exp == "e0":
        return None
    if exp == "e1":
        steps = 20 if smoke else int(p.e1_target_steps)
        return target_ckpt_path(seed, fold, steps)
    if exp == "e2":
        steps = 20 if smoke else int(p.e2_external_steps)
        return external_ckpt_path(seed, steps)
    if exp == "e3":
        ext_steps = 16 if smoke else int(p.e3_external_steps)
        ext_path = external_ckpt_path(seed, ext_steps)
        tgt_steps = 4 if smoke else int(p.e3_target_steps)
        return target_ckpt_path(seed, fold, tgt_steps, init_tag=ext_path.stem)
    raise ValueError(f"unknown exp {exp}")


def get_encoder(exp: str, fold: str, seed: int, cfg, device, smoke: bool) -> nn.Module:
    path = exp_checkpoint_path(exp, fold, seed, cfg, smoke)
    if path is None:
        return freeze(build_model(cfg).to(device))
    if not path.exists():
        if exp == "e1":
            steps = 20 if smoke else int(cfg.pretrain.e1_target_steps)
            pretrain_target(_opts("target", seed, steps, cfg, fold=fold), cfg, device)
        elif exp == "e2":
            steps = 20 if smoke else int(cfg.pretrain.e2_external_steps)
            pretrain_external(_opts("external", seed, steps, cfg), cfg, device)
        elif exp == "e3":
            ext_steps = 16 if smoke else int(cfg.pretrain.e3_external_steps)
            tgt_steps = 4 if smoke else int(cfg.pretrain.e3_target_steps)
            ext_path = external_ckpt_path(seed, ext_steps)
            if not ext_path.exists():
                pretrain_external(_opts("external", seed, ext_steps, cfg), cfg, device)
            pretrain_target(_opts("target", seed, tgt_steps, cfg, fold=fold,
                                  init_path=ext_path), cfg, device)
        else:
            raise ValueError(f"unknown exp {exp}")
    return load_frozen(path, cfg, device)


def exp_optimizer_steps(exp: str, cfg, smoke: bool) -> int:
    """各条件总 SSL optimizer steps（E1/E2/E3 可比；E0=0）。"""
    if smoke:
        return 20
    p = cfg.pretrain
    if exp == "e0":
        return 0
    if exp == "e1":
        return int(p.e1_target_steps)
    if exp == "e2":
        return int(p.e2_external_steps)
    if exp == "e3":
        return int(p.e3_external_steps) + int(p.e3_target_steps)
    raise ValueError(exp)


# ---------------------------------------------------------------------------
# 特征提取 + 下游头
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_all(model: nn.Module, X: np.ndarray, device, batch_size: int = 256) -> torch.Tensor:
    """冻结 encoder 的 mean-pooled 特征：``X (n, 512, 64) -> (n, D)``。"""
    model.eval()
    feats = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device, non_blocking=True)
        feats.append(model.encode_pooled(xb).cpu())
    return torch.cat(feats, dim=0)


def make_head(d_model: int, dropout: float = 0.3) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_model), nn.Dropout(dropout),
        nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_model, 2),
    )


def fit_head(ftr: torch.Tensor, ytr: torch.Tensor,
             fva: torch.Tensor, yva: torch.Tensor, *,
             d_model: int, epochs: int, lr: float, batch_size: int,
             patience: int, class_balance: bool, seed: int,
             device) -> tuple[nn.Module, float, int]:
    """冻结-encoder 下游二分类头：val coupon AUC 驱动早停（模型选择）。"""
    head = make_head(d_model).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    tr_ds = TensorDataset(ftr, ytr)
    if class_balance and len(torch.unique(ytr)) >= 2:
        counts = torch.bincount(ytr.long())
        weights = 1.0 / counts[ytr.long()].float()
        sampler = WeightedRandomSampler(weights, num_samples=len(ytr), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(fva, yva), batch_size=256, shuffle=False)

    set_seed(seed)
    best_auc, best_state, patience_cnt, epochs_run = -1.0, None, 0, 0
    for epoch in range(epochs):
        head.train()
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        ys, ss = [], []
        with torch.no_grad():
            for x, y in va_loader:
                logits = head(x.to(device))
                ys.append(y.numpy())
                ss.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
        ys = np.concatenate(ys); ss = np.concatenate(ss)
        val_auc = 0.5 if ys.min() == ys.max() else float(roc_auc_score(ys, ss))
        epochs_run = epoch + 1
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, best_auc, epochs_run


@torch.no_grad()
def score_test(head: nn.Module, fte: torch.Tensor, device) -> np.ndarray:
    head.eval()
    logits = head(fte.to(device))
    return F.softmax(logits, -1)[:, 1].cpu().numpy()


# ---------------------------------------------------------------------------
# 单条件 5 折 LOOCV
# ---------------------------------------------------------------------------
def run_exp(exp: str, seed: int, cfg, device, smoke: bool) -> dict:
    ascans, coupons, labels = load_paut()
    h = cfg.head
    head_epochs = 1 if smoke else int(h.epochs)
    ssl_steps_total = exp_optimizer_steps(exp, cfg, smoke)

    folds = []
    all_scores, all_labels = [], []
    for tc in COUPONS:
        tr_idx, va_idx, te_idx, train_coupons, val_coupon = paut_fold_split(
            coupons, tc, seed)
        mean, std = penelope_fold_stats(ascans, tr_idx)
        X_all = penelope_transform(ascans, np.arange(len(ascans)), mean, std)
        ckpt_path = exp_checkpoint_path(exp, tc, seed, cfg, smoke)
        enc = get_encoder(exp, tc, seed, cfg, device, smoke)
        assert enc.arch_signature() == build_model(cfg).arch_signature(), (
            f"{exp}/{tc}: encoder structure mismatch!")
        ftr_all = encode_all(enc, X_all, device)
        ftr, ytr = ftr_all[tr_idx], torch.from_numpy(labels[tr_idx].astype(np.int64))
        fva, yva = ftr_all[va_idx], torch.from_numpy(labels[va_idx].astype(np.int64))
        fte, yte = ftr_all[te_idx], labels[te_idx]

        t0 = time.time()
        head, val_auc, epochs_run = fit_head(
            ftr, ytr, fva, yva, d_model=int(cfg.model.d_model),
            epochs=head_epochs, lr=float(h.lr), batch_size=int(h.batch_size),
            patience=int(h.patience), class_balance=bool(h.class_balance),
            seed=seed, device=device)
        wall = round(time.time() - t0, 1)

        scores = score_test(head, fte, device)
        test_auc = float(roc_auc_score(yte, scores))
        pr_auc = float(average_precision_score(yte, scores))
        n_pos = int(yte.sum())
        folds.append({
            "test_coupon": tc, "train_coupons": train_coupons,
            "val_coupon": val_coupon, "n_test": int(len(te_idx)),
            "n_pos": n_pos, "defect_rate": round(float(yte.mean()), 4),
            "val_auc": round(float(val_auc), 4),
            "test_auc": round(test_auc, 4), "pr_auc": round(pr_auc, 4),
            "checkpoint": None, "optimizer_steps": ssl_steps_total,
            "epochs_run": epochs_run, "wall_s": wall,
        })
        all_scores.append(scores); all_labels.append(yte)
        print(f"  fold {tc} train={train_coupons} val={val_coupon} | "
              f"n={len(te_idx)} def={n_pos} | val_auc={val_auc:.4f} "
              f"test_auc={test_auc:.4f} pr_auc={pr_auc:.4f} ({wall}s)")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    aucs = [f["test_auc"] for f in folds]
    np4 = [f["test_auc"] for f in folds if f["test_coupon"] in NP4]
    return {
        "exp": exp, "seed": seed,
        "run_type": "smoke" if smoke else "full",
        "code_commit": git_commit(),
        "head": {"lr": float(h.lr), "epochs": head_epochs,
                 "batch_size": int(h.batch_size),
                 "class_balance": bool(h.class_balance)},
        "optimizer_steps_total": ssl_steps_total,
        "folds": folds,
        "all_folds_mean_auc": round(float(np.mean(aucs)), 4),
        "all_folds_std_auc": round(float(np.std(aucs)), 4),
        "nonPP4_mean_auc": round(float(np.mean(np4)), 4) if np4 else None,
        "nonPP4_std_auc": round(float(np.std(np4)), 4) if np4 else None,
        "pooled_auc": round(float(roc_auc_score(all_labels, all_scores)), 4),
        "pp4_auc": round(float([f["test_auc"] for f in folds
                                if f["test_coupon"] == "PP4"][0]), 4),
    }


def per_exp_path(exp: str, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return RESULTS_DIR / f"m0_2b_{exp}_seed{seed}{suffix}.json"


# ---------------------------------------------------------------------------
# 合并 + 决策
# ---------------------------------------------------------------------------
def stopping_decision(combined: dict) -> dict:
    """按停止判据给建议（E3 vs E1 的 non-PP4 mean AUC 差）。"""
    e1 = combined["conditions"]["e1"]["nonPP4_mean_auc"]
    e3 = combined["conditions"]["e3"]["nonPP4_mean_auc"]
    e2 = combined["conditions"]["e2"]["nonPP4_mean_auc"]
    delta_e3 = e3 - e1 if e1 is not None and e3 is not None else None
    delta_e2 = e2 - e1 if e1 is not None and e2 is not None else None
    if delta_e3 is None:
        verdict = "n/a (smoke)"
    elif delta_e3 >= 0.02:
        # 且至少 3 个非 PP4 折不退化
        a1 = {f["test_coupon"]: f["test_auc"]
              for f in combined["conditions"]["e1"]["folds"]}
        a3 = {f["test_coupon"]: f["test_auc"]
              for f in combined["conditions"]["e3"]["folds"]}
        n_ok = sum(1 for c in NP4 if a3[c] >= a1[c])
        verdict = (f"E3-E1 nonPP4 >= +0.02 ({delta_e3:+.4f}), 非PP4不退化折 "
                   f"{n_ok}/4 -> 建议 seeds 43/44 + ML-NDT-only / NDT_ML_Flaw-only 消融"
                   if n_ok >= 3 else
                   f"E3-E1 nonPP4 >= +0.02 ({delta_e3:+.4f}) 但仅 {n_ok}/4 非PP4折不退化"
                   f" -> 不建议扩大")
    elif delta_e3 >= 0.01:
        verdict = f"E3-E1 nonPP4 在 0.01–0.02 ({delta_e3:+.4f}) -> 建议只补 seed 43"
    else:
        verdict = (f"E3-E1 nonPP4 <= 0.01 ({delta_e3:+.4f}) -> 第一轮没有足够迁移信号；"
                   f"停止扩大模型/预训练数据，转入涡流公开数据基线")
    return {"e3_minus_e1_nonpp4": delta_e3, "e2_minus_e1_nonpp4": delta_e2,
            "decision": verdict}


def combine(seed: int, smoke: bool) -> dict:
    conds = {}
    for exp in EXPS:
        p = per_exp_path(exp, seed, smoke)
        conds[exp] = json.loads(p.read_text()) if p.exists() else None
    combined = {
        "exp": "m0_2b", "seed": seed,
        "run_type": "smoke" if smoke else "full",
        "code_commit": git_commit(),
        "conditions": conds,
        "transfer": stopping_decision({"conditions": conds}),
    }
    json_path = RESULTS_DIR / f"m0_2b_seed{seed}{'_smoke' if smoke else ''}.json"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    write_markdown(combined, json_path.with_suffix(".md"))
    print(f"-> {json_path}")
    return combined


def write_markdown(combined: dict, path: Path) -> None:
    conds = combined["conditions"]
    L = [f"# M0-2B 外部超声预训练迁移 LOOCV（seed {combined['seed']}）",
         "",
         "严格 coupon-level LOOCV（test=1 coupon / val=1 coupon / train=3 coupons），",
         "归一化与目标域 SSL 只读 train coupons；主指标 = **非PP4 逐折均值** "
         "(PP3, PP5, PP6, PP7)；pooled AUC 仅参考；PP4 单独报告。",
         "",
         "| 条件 | PP3 | PP4 | PP5 | PP6 | PP7 | 全5折 mean±std | 非PP4 mean±std | pooled |",
         "|---|---|---|---|---|---|---|---|---|"]
    for exp in EXPS:
        r = conds[exp]
        if r is None:
            continue
        pf = {f["test_coupon"]: f["test_auc"] for f in r["folds"]}
        row = [exp] + [f"{pf[c]:.4f}" for c in COUPONS]
        row.append(f"{r['all_folds_mean_auc']:.4f}±{r['all_folds_std_auc']:.4f}")
        row.append(f"{r['nonPP4_mean_auc']:.4f}±{r['nonPP4_std_auc']:.4f}")
        row.append(f"{r['pooled_auc']:.4f}")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## 每折明细", ""]
    for exp in EXPS:
        r = conds[exp]
        if r is None:
            continue
        L.append(f"### {exp}  (SSL steps={r['optimizer_steps_total']}, "
                 f"head lr={r['head']['lr']}, ep≤{r['head']['epochs']}, "
                 f"class_balance={r['head']['class_balance']})")
        L.append("| test | train coupons | val | n | pos | def_rate | val_auc | "
                 "test_auc | PR-AUC | steps | ep | 耗时s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in r["folds"]:
            L.append(f"| {f['test_coupon']} | {','.join(f['train_coupons'])} | "
                     f"{f['val_coupon']} | {f['n_test']} | {f['n_pos']} | "
                     f"{f['defect_rate']:.3f} | {f['val_auc']:.4f} | "
                     f"{f['test_auc']:.4f} | {f['pr_auc']:.4f} | "
                     f"{f['optimizer_steps']} | {f['epochs_run']} | {f['wall_s']} |")
        L.append("")
    tr = combined["transfer"]
    L += ["## 迁移判断（E3 vs E1）",
          "",
          f"- E3 − E1 non-PP4 mean AUC = "
          f"{(tr['e3_minus_e1_nonpp4'] if tr['e3_minus_e1_nonpp4'] is not None else float('nan')):+.4f}",
          f"- E2 − E1 non-PP4 mean AUC = "
          f"{(tr['e2_minus_e1_nonpp4'] if tr['e2_minus_e1_nonpp4'] is not None else float('nan')):+.4f}",
          f"- 建议：{tr['decision']}",
          "",
          "历史 P4a `0.579±0.007` 仅作参考，不作为 E1 匹配对照；正式迁移判断"
          "必须比较新协议下的 E3 与 E1。"]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=EXPS + ["all", "combine"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-2B LOOCV exp={args.exp} seed={args.seed} smoke={args.smoke} "
          f"device={device}")

    if args.exp == "combine":
        combine(args.seed, args.smoke)
        return
    exps = EXPS if args.exp == "all" else [args.exp]
    for exp in exps:
        print(f"\n===== {exp} (seed {args.seed}) =====")
        res = run_exp(exp, args.seed, cfg, device, args.smoke)
        p = per_exp_path(exp, args.seed, args.smoke)
        p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"-> {p}")
    if args.exp == "all":
        combine(args.seed, args.smoke)


if __name__ == "__main__":
    main()
