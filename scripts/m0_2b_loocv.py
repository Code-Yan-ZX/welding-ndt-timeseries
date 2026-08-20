#!/usr/bin/env python3
"""M0-2B 严格跨试件 LOOCV 评估（外部超声预训练迁移判断，deterministic v2）。

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

**deterministic v2（seed 职责分离）**：``--split-seed`` 只控制 coupon 划分；
``--data-seed`` 只控制数据采样；``--model-seed``（``--seed`` 为别名）只控制
模型初始化 / MAE mask / dropout / 分类头初始化 / 训练随机性。所有结果写入
带 ``det_v2`` 后缀的新文件，不覆盖旧 seed42 结果。

迁移判断（最终口径，见 ``aggregate``）：
- E3 用于判断“外部预训练后继续目标域 MAE”是否有效（E3 vs E2，seed42）；
- E2 用于判断“外部 encoder 直接迁移”是否有效（E2 vs E0 主对照，E2 vs E1 次对照）；
- 三个 model seed 平均 E2−E0 ≥ +0.01 且 ≥2/3 seed 为正 -> 保留 external
  encoder（可复现的小幅迁移证据）；否则结束公开超声迁移实验。

smoke 输出带 ``_smoke`` 后缀，不覆盖正式结果。

Usage:
  python scripts/m0_2b_loocv.py --exp all --model-seed 42
  python scripts/m0_2b_loocv.py --exp e1 --model-seed 42
  python scripts/m0_2b_loocv.py --exp e2 --model-seed 42
  python scripts/m0_2b_loocv.py --smoke            # 冒烟（20 SSL steps / 1 head ep）
  python scripts/m0_2b_loocv.py --exp combine      # 合并已生成的各条件结果
  python scripts/m0_2b_loocv.py --aggregate        # 三种子(42/43/44)聚合 + 迁移判据
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
DET_TAG = "det_v2"
DEFAULT_MODEL_SEEDS = [42, 43, 44]


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
def _opts(cmd, model_seed, split_seed, data_seed, steps, cfg, fold=None,
          init_path=None):
    return PretrainOpts(
        cmd=cmd, split_seed=split_seed, data_seed=data_seed,
        model_seed=model_seed, steps=steps,
        batch_size=int(cfg.pretrain.batch_size),
        steps_per_epoch=int(cfg.pretrain.steps_per_epoch),
        fold=fold, init_path=init_path,
    )


def exp_checkpoint_path(exp: str, fold: str, model_seed: int, cfg,
                        smoke: bool) -> Path | None:
    """各条件下该折 encoder 的 checkpoint 路径（E0 无预训练 -> None）。

    路径全部带 det_v2 目录/后缀，与旧 seed42 checkpoint 隔离。
    """
    p = cfg.pretrain
    if exp == "e0":
        return None
    if exp == "e1":
        steps = 20 if smoke else int(p.e1_target_steps)
        return target_ckpt_path(model_seed, fold, steps)
    if exp == "e2":
        steps = 20 if smoke else int(p.e2_external_steps)
        return external_ckpt_path(model_seed, steps)
    if exp == "e3":
        ext_steps = 16 if smoke else int(p.e3_external_steps)
        ext_path = external_ckpt_path(model_seed, ext_steps)
        tgt_steps = 4 if smoke else int(p.e3_target_steps)
        return target_ckpt_path(model_seed, fold, tgt_steps, init_tag=ext_path.stem)
    raise ValueError(f"unknown exp {exp}")


def get_encoder(exp: str, fold: str, model_seed: int, cfg, device, smoke: bool,
                split_seed: int, data_seed: int) -> nn.Module:
    """返回冻结 encoder。E0 随机 encoder 在 **set_seed(model_seed) 后**构建
    （同一 model_seed 下 5 折共用同一个随机 encoder）。E1/E2/E3 缺 ckpt 时
    现场跑对应预训练（数据采样用 data_seed / split_seed）。"""
    path = exp_checkpoint_path(exp, fold, model_seed, cfg, smoke)
    if path is None:
        set_seed(model_seed)                   # E0：构建前必须 set_seed
        return freeze(build_model(cfg).to(device))
    if not path.exists():
        if exp == "e1":
            steps = 20 if smoke else int(cfg.pretrain.e1_target_steps)
            pretrain_target(_opts("target", model_seed, split_seed, data_seed,
                                  steps, cfg, fold=fold), cfg, device)
        elif exp == "e2":
            steps = 20 if smoke else int(cfg.pretrain.e2_external_steps)
            pretrain_external(_opts("external", model_seed, split_seed, data_seed,
                                    steps, cfg), cfg, device)
        elif exp == "e3":
            ext_steps = 16 if smoke else int(cfg.pretrain.e3_external_steps)
            tgt_steps = 4 if smoke else int(cfg.pretrain.e3_target_steps)
            ext_path = external_ckpt_path(model_seed, ext_steps)
            if not ext_path.exists():
                pretrain_external(_opts("external", model_seed, split_seed,
                                        data_seed, ext_steps, cfg), cfg, device)
            pretrain_target(_opts("target", model_seed, split_seed, data_seed,
                                  tgt_steps, cfg, fold=fold, init_path=ext_path),
                            cfg, device)
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
    """分类头。**调用方必须先在 make_head() 前 set_seed(model_seed)**，
    保证同一 fold 下 E0/E1/E2/E3 的初始化一致、且只由 model_seed 决定。"""
    return nn.Sequential(
        nn.LayerNorm(d_model), nn.Dropout(dropout),
        nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_model, 2),
    )


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker 确定性：继承主进程 seed（由 generator 派生）。"""
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    import random
    random.seed(ws)


def fit_head(ftr: torch.Tensor, ytr: torch.Tensor,
             fva: torch.Tensor, yva: torch.Tensor, *,
             d_model: int, epochs: int, lr: float, batch_size: int,
             patience: int, class_balance: bool, seed: int,
             device) -> tuple[nn.Module, float, int]:
    """冻结-encoder 下游二分类头：val coupon AUC 驱动早停（模型选择）。

    ``seed`` = model_seed：**在 make_head / WeightedRandomSampler /
    DataLoader 构建前 set_seed**，保证头初始化与训练随机性只由 model_seed
    决定（同一 fold 的 E0/E1/E2/E3 头初始化一致；checkpoint 是否存在不影响
    头初始化）。
    """
    set_seed(seed)
    head = make_head(d_model).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    tr_ds = TensorDataset(ftr, ytr)
    dl_gen = torch.Generator().manual_seed(seed)
    if class_balance and len(torch.unique(ytr)) >= 2:
        counts = torch.bincount(ytr.long())
        weights = 1.0 / counts[ytr.long()].float()
        sampler = WeightedRandomSampler(weights, num_samples=len(ytr), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True,
                               generator=dl_gen, worker_init_fn=_seed_worker)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True,
                               generator=dl_gen, worker_init_fn=_seed_worker)
    va_loader = DataLoader(TensorDataset(fva, yva), batch_size=256, shuffle=False)

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
def run_exp(exp: str, model_seed: int, cfg, device, smoke: bool,
            split_seed: int, data_seed: int) -> dict:
    ascans, coupons, labels = load_paut()
    h = cfg.head
    head_epochs = 1 if smoke else int(h.epochs)
    ssl_steps_total = exp_optimizer_steps(exp, cfg, smoke)

    folds = []
    all_scores, all_labels = [], []
    for tc in COUPONS:
        tr_idx, va_idx, te_idx, train_coupons, val_coupon = paut_fold_split(
            coupons, tc, split_seed)
        mean, std = penelope_fold_stats(ascans, tr_idx)
        X_all = penelope_transform(ascans, np.arange(len(ascans)), mean, std)
        ckpt_path = exp_checkpoint_path(exp, tc, model_seed, cfg, smoke)
        enc = get_encoder(exp, tc, model_seed, cfg, device, smoke,
                          split_seed, data_seed)
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
            seed=model_seed, device=device)
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
        "exp": exp, "model_seed": model_seed,
        "split_seed": split_seed, "data_seed": data_seed,
        "det_version": DET_TAG,
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


def per_exp_path(exp: str, model_seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return RESULTS_DIR / f"m0_2b_{exp}_seed{model_seed}_{DET_TAG}{suffix}.json"


# ---------------------------------------------------------------------------
# 合并 + 单 seed 决策（参考）
# ---------------------------------------------------------------------------
def stopping_decision(combined: dict) -> dict:
    """单 seed 决策（参考）：报告 E3−E1 / E2−E1 / E2−E0 / E3−E2。

    **最终迁移判据由 ``aggregate``（三种子）决定**：此处仅记录单 seed 数值，
    明确标注"单 seed 不能下结论"。
    """
    conds = combined["conditions"]
    v = {}
    for key, a, b in (("e3_minus_e1", "e3", "e1"),
                      ("e2_minus_e1", "e2", "e1"),
                      ("e2_minus_e0", "e2", "e0"),
                      ("e3_minus_e2", "e3", "e2")):
        xa = conds.get(a) and conds[a]["nonPP4_mean_auc"]
        xb = conds.get(b) and conds[b]["nonPP4_mean_auc"]
        v[key] = (xa - xb) if (xa is not None and xb is not None) else None
    v["decision"] = (
        "单 seed 仅供参考：最终迁移判据看三种子聚合 "
        "（平均 E2−E0 ≥ +0.01 且 ≥2/3 seed 为正）与 E3−E2（seed42）。"
    )
    return v


def combine(model_seed: int, smoke: bool) -> dict:
    conds = {}
    for exp in EXPS:
        p = per_exp_path(exp, model_seed, smoke)
        conds[exp] = json.loads(p.read_text()) if p.exists() else None
    combined = {
        "exp": "m0_2b", "model_seed": model_seed,
        "det_version": DET_TAG,
        "run_type": "smoke" if smoke else "full",
        "code_commit": git_commit(),
        "conditions": conds,
        "transfer": stopping_decision({"conditions": conds}),
    }
    json_path = RESULTS_DIR / f"m0_2b_seed{model_seed}_{DET_TAG}{'_smoke' if smoke else ''}.json"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    write_markdown(combined, json_path.with_suffix(".md"))
    print(f"-> {json_path}")
    return combined


def write_markdown(combined: dict, path: Path) -> None:
    conds = combined["conditions"]
    ms = combined["model_seed"]
    L = [f"# M0-2B 外部超声预训练迁移 LOOCV（deterministic v2，model_seed {ms}）",
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
    L += ["## 迁移判断（单 seed 参考）",
          "",
          f"- E3 − E1 non-PP4 mean AUC = "
          f"{(tr['e3_minus_e1'] if tr['e3_minus_e1'] is not None else float('nan')):+.4f}",
          f"- E2 − E1 non-PP4 mean AUC = "
          f"{(tr['e2_minus_e1'] if tr['e2_minus_e1'] is not None else float('nan')):+.4f}",
          f"- E2 − E0 non-PP4 mean AUC = "
          f"{(tr['e2_minus_e0'] if tr['e2_minus_e0'] is not None else float('nan')):+.4f}",
          f"- E3 − E2 non-PP4 mean AUC = "
          f"{(tr['e3_minus_e2'] if tr['e3_minus_e2'] is not None else float('nan')):+.4f}",
          f"- 建议：{tr['decision']}",
          "",
          "历史 P4a `0.579±0.007` 仅作参考，不作为 E1 匹配对照；最终迁移判据"
          "见三种子聚合（`--aggregate`）。"]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# 三种子聚合 + 最终迁移判据
# ---------------------------------------------------------------------------
def _np4_by_coupon(res: dict) -> dict[str, float]:
    return {f["test_coupon"]: f["test_auc"] for f in res["folds"]}


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else float("nan")


def aggregate(model_seeds: list[int], smoke: bool) -> dict:
    """三种子（42/43/44）聚合 + 最终迁移判据。

    判据（任务规定）：
    - 平均 E2−E0 ≥ +0.01 **且** ≥2/3 seed 的 E2−E0 > 0 -> 结论：外部超声 SSL
      对 PAUT 存在**可复现的小幅直接迁移价值**，保留 external encoder 为未来
      合作单位真实 UT 数据的初始化；但不扩大公开超声模型与数据。
    - 否则 -> 外部直接迁移正信号不能跨初始化稳定复现，结束公开超声迁移实验。
    - E3（仅 seed42）：E3−E2 明显为负 -> 停止“外部预训练后继续目标域 MAE”。
    """
    per = {}
    for ms in model_seeds:
        p = RESULTS_DIR / f"m0_2b_seed{ms}_{DET_TAG}{'_smoke' if smoke else ''}.json"
        per[ms] = json.loads(p.read_text()) if p.exists() else None

    def m(exp: str) -> list[float]:
        vals = []
        for ms in model_seeds:
            c = per[ms] and per[ms]["conditions"].get(exp)
            if c and c["nonPP4_mean_auc"] is not None:
                vals.append(c["nonPP4_mean_auc"])
        return vals

    e0s, e1s, e2s = m("e0"), m("e1"), m("e2")
    d20 = [a - b for a, b in zip(e2s, e0s)]
    d21 = [a - b for a, b in zip(e2s, e1s)]
    # E3 只有 seed42
    e3_42 = per.get(42) and per[42]["conditions"].get("e3")
    e2_42 = per.get(42) and per[42]["conditions"].get("e2")

    # 每 coupon 三种子 mean±std（E0/E1/E2）
    coupon_stats: dict[str, dict] = {}
    for c in sorted(NP4):
        vals_by_exp = {}
        for exp in ("e0", "e1", "e2"):
            vals = [per[ms]["conditions"][exp]["folds"] for ms in model_seeds
                    if per[ms] and per[ms]["conditions"].get(exp)]
            aucs = [f["test_auc"] for v in vals for f in v if f["test_coupon"] == c]
            vals_by_exp[exp] = {"mean": _mean(aucs), "std": _std(aucs),
                                "vals": [round(x, 4) for x in aucs]}
        coupon_stats[c] = vals_by_exp

    # E2 收益是否被 PP7 单折主导：非 PP7 三折(E2−E0) vs 全四折(E2−E0)
    def mean_nonpp4_minus_p7(exp: str, ms: int) -> float:
        r = per[ms]["conditions"][exp]
        np4c = {f["test_coupon"]: f["test_auc"] for f in r["folds"]
                if f["test_coupon"] in NP4}
        vals = [v for c, v in np4c.items() if c != "PP7"]
        return float(np.mean(vals)) if vals else float("nan")

    dominated_by_p7 = False
    if len(e2s) and len(e0s):
        deltas_all = [a - b for a, b in zip(e2s, e0s)]
        deltas_nonp7 = []
        for ms in model_seeds:
            if per[ms]:
                deltas_nonp7.append(mean_nonpp4_minus_p7("e2", ms)
                                   - mean_nonpp4_minus_p7("e0", ms))
        avg_all = float(np.mean(deltas_all))
        avg_nonp7 = float(np.mean(deltas_nonp7))
        # PP7 单折主导：去掉 PP7 后平均 E2−E0 变为负（或接近 0）
        dominated_by_p7 = avg_nonp7 <= 0.0 and avg_all > 0.0
    else:
        avg_all = avg_nonp7 = float("nan")

    n_pos = sum(1 for d in d20 if d > 0)
    avg_d20 = _mean(d20)
    avg_d21 = _mean(d21)
    criterion_met = (avg_d20 >= 0.01) and (n_pos >= max(1, len(model_seeds) * 2 // 3))
    if len(e2s) >= 2 and criterion_met:
        verdict = ("外部超声 SSL 对 PAUT 存在**小幅但可复现的直接迁移价值**："
                   f"平均 E2−E0={avg_d20:+.4f}（≥+0.01）且 {n_pos}/{len(e2s)} "
                   "seed 为正。可将 external encoder 保留为未来合作单位真实 UT "
                   "数据的初始化；**不**继续扩大公开超声模型和数据。")
    else:
        verdict = ("外部直接迁移正信号不能跨初始化稳定复现"
                   f"（平均 E2−E0={avg_d20:+.4f}，正 seed 数 {n_pos}/{len(e2s)}）："
                   "**结束公开超声迁移实验**，不再扩大公开超声模型和数据。")

    # E3 判断（seed42）：E3−E2 明显为负 -> 目标域继续 MAE 有害
    e3_judgment = "n/a (E3 未运行)"
    if e3_42 and e2_42 and e3_42["nonPP4_mean_auc"] is not None and \
            e2_42["nonPP4_mean_auc"] is not None:
        d32 = e3_42["nonPP4_mean_auc"] - e2_42["nonPP4_mean_auc"]
        d31 = None
        e1_42 = per[42]["conditions"].get("e1")
        if e1_42 and e1_42["nonPP4_mean_auc"] is not None:
            d31 = e3_42["nonPP4_mean_auc"] - e1_42["nonPP4_mean_auc"]
        per_fold = {f["test_coupon"]: f["test_auc"] for f in e3_42["folds"]}
        per_fold_e2 = {f["test_coupon"]: f["test_auc"] for f in e2_42["folds"]}
        fold_changes = {c: round(per_fold[c] - per_fold_e2[c], 4)
                        for c in COUPONS}
        harmful = d32 < -0.01
        e3_judgment = {
            "e3_minus_e2": round(d32, 4),
            "e3_minus_e1": round(d31, 4) if d31 is not None else None,
            "per_fold_changes_vs_e2": fold_changes,
            "target_continue_mae_harmful": harmful,
            "note": "E3−E2 明显为负 -> 停止“外部预训练后继续目标域 MAE”路线，"
                    "不继续调参（此结论独立于 E2 是否成立）。" if harmful else
                    "E3−E2 未明显为负，目标域继续 MAE 路线需结合三种子评估。",
        }

    out = {
        "exp": "m0_2b_det_v2_aggregate", "det_version": DET_TAG,
        "run_type": "smoke" if smoke else "full",
        "code_commit": git_commit(),
        "model_seeds": list(model_seeds),
        "split_seed": per[model_seeds[0]]["conditions"]["e0"]["split_seed"],
        "data_seed": per[model_seeds[0]]["conditions"]["e0"]["data_seed"],
        "conditions_three_seed": {
            "e0": {"mean": _mean(e0s), "std": _std(e0s),
                   "seeds": [round(x, 4) for x in e0s]},
            "e1": {"mean": _mean(e1s), "std": _std(e1s),
                   "seeds": [round(x, 4) for x in e1s]},
            "e2": {"mean": _mean(e2s), "std": _std(e2s),
                   "seeds": [round(x, 4) for x in e2s]},
        },
        "avg_e2_minus_e0": round(avg_d20, 4),
        "avg_e2_minus_e1": round(avg_d21, 4),
        "e2_minus_e0_per_seed": [round(x, 4) for x in d20],
        "e2_minus_e1_per_seed": [round(x, 4) for x in d21],
        "n_seeds_e2_gt_e0": n_pos,
        "avg_e2_minus_e0_excluding_pp7": round(avg_nonp7, 4),
        "e2_gain_dominated_by_pp7": dominated_by_p7,
        "criterion_met": criterion_met,
        "verdict": verdict,
        "coupon_three_seed": coupon_stats,
        "e3_seed42": e3_judgment,
    }
    json_path = RESULTS_DIR / f"m0_2b_{DET_TAG}_aggregate{'_smoke' if smoke else ''}.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_aggregate_markdown(out, json_path.with_suffix(".md"))
    print(f"-> {json_path}")
    return out


def write_aggregate_markdown(agg: dict, path: Path) -> None:
    L = ["# M0-2B 外部超声预训练迁移 —— 三种子聚合（deterministic v2）",
         "",
         f"- model seeds: {agg['model_seeds']}（split_seed={agg['split_seed']}, "
         f"data_seed={agg['data_seed']}）",
         f"- 主指标 = 非PP4 逐折均值（PP3/PP5/PP6/PP7）；pooled AUC 仅参考；"
         f"PP4 单独报告",
         "",
         "## 三种子汇总（non-PP4 mean±std）",
         "",
         "| 条件 | 三种子 mean±std | 各 seed |",
         "|---|---|---|"]
    for exp in ("e0", "e1", "e2"):
        c = agg["conditions_three_seed"][exp]
        L.append(f"| {exp} | {c['mean']:.4f}±{c['std']:.4f} | "
                 f"{', '.join(f'{s:.4f}' for s in c['seeds'])} |")
    L += ["",
          "| 对比 | 平均 Δ | 各 seed Δ |",
          "|---|---|---|",
          f"| **E2 − E0** | **{agg['avg_e2_minus_e0']:+.4f}** | "
          f"{', '.join(f'{x:+.4f}' for x in agg['e2_minus_e0_per_seed'])} |",
          f"| E2 − E1 | {agg['avg_e2_minus_e1']:+.4f} | "
          f"{', '.join(f'{x:+.4f}' for x in agg['e2_minus_e1_per_seed'])} |",
          "",
          f"- E2−E0 为正的 seed 数：{agg['n_seeds_e2_gt_e0']}/{len(agg['model_seeds'])}",
          f"- 平均 E2−E0（剔除 PP7 折）：{agg['avg_e2_minus_e0_excluding_pp7']:+.4f}",
          f"- E2 收益是否被 PP7 单折主导：{'是' if agg['e2_gain_dominated_by_pp7'] else '否'}",
          f"- 是否满足 +0.01 & 2/3 seed 判据：{'是' if agg['criterion_met'] else '否'}",
          "",
          "## 四种非PP4 coupon 三种子 mean±std",
          "",
          "| coupon | E0 | E1 | E2 |",
          "|---|---|---|---|"]
    for c in sorted(agg["coupon_three_seed"]):
        row = [f"**{c}**"]
        for exp in ("e0", "e1", "e2"):
            v = agg["coupon_three_seed"][c][exp]
            row.append(f"{v['mean']:.4f}±{v['std']:.4f}")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## 结论",
          "",
          f"{agg['verdict']}"]
    if isinstance(agg["e3_seed42"], dict):
        ej = agg["e3_seed42"]
        L += ["",
              "## E3（seed42 确定性复跑）",
              "",
              f"- E3 − E2 = {ej['e3_minus_e2']:+.4f}",
              f"- E3 − E1 = {(ej['e3_minus_e1'] if ej['e3_minus_e1'] is not None else float('nan')):+.4f}",
              f"- 每折变化（vs E2）：{ej['per_fold_changes_vs_e2']}",
              f"- 目标域继续 MAE 是否仍有害：{'是' if ej['target_continue_mae_harmful'] else '否'}",
              f"- {ej['note']}"]
    L.append("")
    L.append("> 措辞纪律：单 seed 不能写“证明”；多种子成立只能写“提供可复现的"
             "小幅迁移证据”；不得写成“突破 PAUT 天花板”除非确实显著超过匹配协议"
             "基线；E3 失败只否定目标域继续 MAE，不自动否定 E2；不得用 pooled "
             "AUC 替代逐折主指标。")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=EXPS + ["all", "combine"], default="all")
    ap.add_argument("--seed", type=int, default=None, dest="model_seed",
                    help="[deprecated alias] = --model-seed")
    ap.add_argument("--model-seed", type=int, default=42,
                    help="只控制模型初始化/训练随机性")
    ap.add_argument("--data-seed", type=int, default=42,
                    help="只控制数据采样（抽帧/裁窗/SSL 样本顺序）")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="只控制 coupon train/val/test 划分")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="三种子(42/43/44)聚合 + 最终迁移判据")
    ap.add_argument("--model-seeds", type=int, nargs="+", default=None,
                    help="聚合用 model seeds（默认 42 43 44）")
    args = ap.parse_args()
    if args.model_seed is None:
        args.model_seed = 42
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"M0-2B LOOCV exp={args.exp} model_seed={args.model_seed} "
          f"data_seed={args.data_seed} split_seed={args.split_seed} "
          f"smoke={args.smoke} device={device}")

    if args.aggregate:
        seeds = args.model_seeds or DEFAULT_MODEL_SEEDS
        aggregate(seeds, args.smoke)
        return
    if args.exp == "combine":
        combine(args.model_seed, args.smoke)
        return
    exps = EXPS if args.exp == "all" else [args.exp]
    for exp in exps:
        print(f"\n===== {exp} (model_seed {args.model_seed}) =====")
        res = run_exp(exp, args.model_seed, cfg, device, args.smoke,
                      args.split_seed, args.data_seed)
        p = per_exp_path(exp, args.model_seed, args.smoke)
        p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"-> {p}")
    if args.exp == "all":
        combine(args.model_seed, args.smoke)


if __name__ == "__main__":
    main()
