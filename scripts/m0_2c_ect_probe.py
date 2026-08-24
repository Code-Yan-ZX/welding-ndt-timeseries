#!/usr/bin/env python3
"""M0-2C ECT 下游评估：冻结 encoder，clean vs flaw 二分类（规范头）。

协议（transductive_unlabeled representation probe）：
- SSL 使用**全部** 2780 views（无划分）；本脚本对冻结表征做 group probe；
  **不得把 transductive 结果写成严格 cross-group 泛化**（报告与 JSON 显式标记）。
- 单元 = 扫描（695）：每扫描 4 个 view 的 mean-pooled encoder 特征取均值
  -> 每条扫描 1 个 (128,) 特征；标签 = clean vs flaw（flaw=1）。
- 划分：按 **config/specimen proxy**（specimen_id）做 5 折
  ``StratifiedGroupKFold``；**同一配置组绝不跨 fold**；fold 内再按组切
  inner val（~20% 组）驱动早停。
- 指标：fold mean ROC-AUC / PR-AUC / balanced accuracy；输出逐 seed 结果，
  聚合时比较 P→E 与 E 的逐 seed 配对差值。
- 每折训练前输出审计块：train/val/test 记录数、配置组数、clean/flaw 数量、
  8 类标签分布、material/sensor 分布；**train、val、test 必须同时有正负样本**，
  否则该 fold 无效并停止。
- 当前阶段 ECT 主任务只做 clean vs flaw 二分类；**8 类只输出分布，不训练**。

Usage:
  python scripts/m0_2c_ect_probe.py --cond E  --seed 42 --steps 10000
  python scripts/m0_2c_ect_probe.py --cond PE --seed 42 --steps 2000 --tag pilot
  python scripts/m0_2c_ect_probe.py --cond E --seed 42 --steps 100 --smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from wndt.data.adapters.eddycus import EddyCusAdapter  # noqa: E402
from wndt.data.eddycus_pretrain import (  # noqa: E402
    build_view_index, ect_view_summary, read_view_ds,
)
from wndt.models.ssl_ae import MAEEncoder  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

from m0_2b_loocv import fit_head  # noqa: E402
from m0_2c_ect_pretrain import (  # noqa: E402
    DEFAULT_CONFIG, ckpt_path, git_commit,
)

RESULTS_DIR = REPO / "experiments" / "results"
RUN_TYPE = "transductive_unlabeled"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# 扫描级特征（4 view 特征均值；与训练完全相同的读取/归一化/下采样路径）
# ---------------------------------------------------------------------------
@torch.no_grad()
def scan_features(enc: nn.Module, adapter: EddyCusAdapter, view_index,
                  device) -> tuple[np.ndarray, list[int]]:
    enc.eval()
    by_scan: dict[int, list[int]] = defaultdict(list)
    for vi, v in enumerate(view_index):
        by_scan[v.rec_index].append(vi)
    scan_rec_indices = sorted(by_scan)
    feats = np.zeros((len(scan_rec_indices), 128), dtype=np.float32)
    for si, rec_idx in enumerate(scan_rec_indices):
        zs = []
        for vi in by_scan[rec_idx]:
            grid, _valid = read_view_ds(adapter, rec_idx, view_index[vi].freq_key)
            x = torch.from_numpy(grid[None]).to(device)
            zs.append(enc(x).cpu().numpy()[0])
        feats[si] = np.mean(zs, axis=0)
    return feats, scan_rec_indices


# ---------------------------------------------------------------------------
# fold 审计（训练前必跑：正负样本齐备否则停止）
# ---------------------------------------------------------------------------
def audit_split(rows: list[dict], train_idx, val_idx, test_idx,
                tag: str) -> dict:
    """输出 + 断言每部分同时有正负样本；返回审计块。"""
    out = {}
    for part, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sub = [rows[i] for i in idx]
        pos = sum(1 for r in sub if r["flaw"])
        neg = len(sub) - pos
        groups = len({r["specimen_id"] for r in sub})
        types = Counter(r["defect_type"] for r in sub)
        mats = Counter(r["material"] for r in sub)
        sens = Counter(r["sensor"] for r in sub)
        out[part] = {
            "n_records": len(sub),
            "n_config_groups": groups,
            "flaw": pos, "clean": neg,
            "defect_type_dist": dict(types),
            "material_dist": dict(mats),
            "sensor_dist": dict(sens),
        }
        if pos == 0 or neg == 0:
            raise ValueError(
                f"{tag}/{part}: 缺正负样本 (flaw={pos}, clean={neg})，fold 无效，停止。")
    return out


def inner_val_split(train_idx, groups, flaws, seed, val_frac: float = 0.2):
    """train 内按配置组再切 inner val（~val_frac 组），**保证正负样本齐备**。

    clean 组与 flaw 组分别取 ~20%（各至少 1 组）进 val，其余进 train——
    只要外层 SGKFold 的 train 同时含两类（分层保证），val/train 必同时含
    正负样本。确定性：只由 ``seed`` 决定（rng）。
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_groups = groups[train_idx]
    clean_groups = sorted({g for g, f in zip(train_groups, flaws[train_idx])
                           if f == 0})
    flaw_groups = sorted({g for g, f in zip(train_groups, flaws[train_idx])
                          if f == 1})
    rng = np.random.default_rng(seed)
    n_val_clean = max(1, round(len(clean_groups) * val_frac))
    n_val_flaw = max(1, round(len(flaw_groups) * val_frac))
    val_clean = set(rng.choice(clean_groups, min(n_val_clean, len(clean_groups)),
                               replace=False).tolist())
    val_flaw = set(rng.choice(flaw_groups, min(n_val_flaw, len(flaw_groups)),
                              replace=False).tolist())
    val_groups = val_clean | val_flaw
    val_mask = np.array([g in val_groups for g in train_groups])
    va = train_idx[val_mask]
    tr = train_idx[~val_mask]
    return tr, va


def best_threshold(y_val, s_val) -> float:
    """val 上最大化 macro-F1 的阈值（PAUT 协议一致）。"""
    ys = np.asarray(y_val)
    ss = np.asarray(s_val)
    if len(np.unique(ys)) < 2:
        return 0.5
    cands = np.unique(np.concatenate([ss, [0.5]]))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        pred = (ss > t).astype(int)
        tp = np.sum((pred == 1) & (ys == 1))
        fp = np.sum((pred == 1) & (ys == 0))
        fn = np.sum((pred == 0) & (ys == 1))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def run_probe(cond: str, model_seed: int, steps: int, tag: str | None,
              cfg, device, smoke: bool) -> dict:
    p = ckpt_path(cond, model_seed, steps, tag)
    assert p.exists(), f"checkpoint 不存在: {p}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    enc = MAEEncoder(d_model=int(cfg.model.d_model),
                     in_channels=int(cfg.model.in_channels)).to(device)
    enc.load_state_dict(ck["state_dict"]["encoder"])
    for prm in enc.parameters():
        prm.requires_grad = False
    enc.eval()
    print(f"[probe {cond} s{model_seed}] 加载 {p}，冻结 encoder")

    adapter = EddyCusAdapter()
    view_index = build_view_index(adapter)
    feats, scan_rec_indices = scan_features(enc, adapter, view_index, device)
    v0_by_rec = {v.rec_index: v for v in view_index}
    rows = []
    for si, rec_idx in enumerate(scan_rec_indices):
        v0 = v0_by_rec[rec_idx]
        rows.append({
            "rec_index": rec_idx, "record_id": v0.record_id,
            "specimen_id": v0.specimen_id, "flaw": v0.flaw,
            "defect_type": v0.defect_type, "material": v0.material,
            "sensor": v0.sensor,
        })
    y = np.array([1 if r["flaw"] else 0 for r in rows], dtype=np.int64)
    groups = np.array([r["specimen_id"] for r in rows])
    n_groups = len(set(groups.tolist()))
    print(f"[probe {cond} s{model_seed}] scans={len(rows)} groups={n_groups} "
          f"flaw={int(y.sum())} clean={int((1 - y).sum())}")

    h = cfg.head
    head_epochs = 1 if smoke else int(h.epochs)

    folds = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=model_seed)
    split_iter = sgkf.split(np.zeros(len(rows)), y, groups)
    for fold_i, (tr_outer, te_outer) in enumerate(split_iter):
        tr_outer = np.asarray(tr_outer, dtype=np.int64)
        te_outer = np.asarray(te_outer, dtype=np.int64)
        # inner val：train 内按配置组切 ~20% 组（组纯度保持；保证正负齐备）
        tr_inner, va_inner = inner_val_split(
            tr_outer, groups, y, seed=model_seed * 100 + fold_i)
        audit = audit_split(rows, tr_inner, va_inner, te_outer,
                            f"fold{fold_i}")
        # fit_head 内部用 DataLoader(num_workers>0) -> 必须传 CPU 张量（同 m0_2b）
        ftr = torch.from_numpy(feats[tr_inner])
        ytr = torch.from_numpy(y[tr_inner])
        fva = torch.from_numpy(feats[va_inner])
        yva = torch.from_numpy(y[va_inner])
        fte = torch.from_numpy(feats[te_outer]).to(device)
        yte = y[te_outer]

        t0 = time.time()
        head, val_auc, epochs_run = fit_head(
            ftr, ytr, fva, yva, d_model=int(cfg.model.d_model),
            epochs=head_epochs, lr=float(h.lr), batch_size=int(h.batch_size),
            patience=int(h.patience), class_balance=bool(h.class_balance),
            seed=model_seed, device=device)
        wall = round(time.time() - t0, 1)
        with torch.no_grad():
            logits = head(fte)
            scores = F.softmax(logits, -1)[:, 1].cpu().numpy()
            val_scores = F.softmax(head(fva.to(device)), -1)[:, 1].cpu().numpy()
        thr = best_threshold(y[va_inner], val_scores)
        auc = float(roc_auc_score(yte, scores))
        pr = float(average_precision_score(yte, scores))
        bacc = float(balanced_accuracy_score(yte, (scores > thr).astype(int)))
        folds.append({
            "fold": fold_i,
            "n_test": int(len(te_outer)), "n_test_flaw": int(yte.sum()),
            "n_test_clean": int(len(te_outer) - yte.sum()),
            "n_test_groups": len({groups[i] for i in te_outer}),
            "val_auc": round(float(val_auc), 4), "thr": round(thr, 4),
            "roc_auc": round(auc, 4), "pr_auc": round(pr, 4),
            "balanced_acc": round(bacc, 4), "epochs_run": epochs_run,
            "wall_s": wall,
            "audit": audit,
        })
        print(f"  fold{fold_i} n_test={len(te_outer)} (flaw={int(yte.sum())}) "
              f"groups={folds[-1]['n_test_groups']} | val_auc={val_auc:.4f} "
              f"roc_auc={auc:.4f} pr_auc={pr:.4f} bacc={bacc:.4f} ({wall}s)")

    res = {
        "exp": "m0_2c_ect_probe", "cond": cond, "model_seed": model_seed,
        "data_seed": int(cfg.pretrain.data_seed),
        "run_type": RUN_TYPE,
        "smoke": bool(smoke), "ckpt": str(p),
        "n_scans": len(rows), "n_groups": n_groups,
        "n_flaw": int(y.sum()), "n_clean": int(len(rows) - y.sum()),
        "8class_dist_scans": dict(Counter(r["defect_type"] for r in rows)),
        "head": {"lr": float(h.lr), "epochs": head_epochs,
                 "batch_size": int(h.batch_size),
                 "class_balance": bool(h.class_balance)},
        "folds": folds,
        "fold_mean_roc_auc": round(float(np.mean([f["roc_auc"] for f in folds])), 4),
        "fold_mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in folds])), 4),
        "fold_mean_balanced_acc": round(float(np.mean([f["balanced_acc"] for f in folds])), 4),
        "pooled_roc_auc": None,     # 主指标 = fold mean；pooled 不在此处计
        "note": "transductive_unlabeled representation probe：SSL 使用全部 ECT "
                "无标注视图后冻结；group 5 折按 config/specimen proxy，同一配置组"
                "绝不跨 fold；不得写成严格 cross-group 泛化。",
        "code_commit": git_commit(), "code_dirty": git_dirty(),
    }
    return res


def per_exp_path(cond: str, model_seed: int, steps: int, tag: str | None,
                 smoke: bool) -> Path:
    tag_s = f"_{tag}" if tag else ""
    suffix = ("_smoke" if (smoke and tag != "smoke") else "")
    return RESULTS_DIR / f"m0_2c_ect_probe_{cond}_seed{model_seed}_s{steps}{tag_s}{suffix}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=("E", "PE"))
    ap.add_argument("--seed", type=int, default=42, dest="model_seed")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps is None:
        args.steps = 100 if args.smoke else int(cfg.pretrain.steps)
    if args.smoke and args.tag is None:
        args.tag = "smoke"          # 与 pretrain --smoke 的 tag 一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run_probe(args.cond, args.model_seed, args.steps, args.tag,
                    cfg, device, args.smoke)
    out = per_exp_path(args.cond, args.model_seed, args.steps, args.tag, args.smoke)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
