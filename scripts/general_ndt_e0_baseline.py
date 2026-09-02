#!/usr/bin/env python
"""General NDT Foundation — E0 scratch-supervised 严格基线 (PENELOPE, Phase 2A → E 阶段).

协议 (对齐 P4a 规范头, phase1_experiment_protocol.md §E0):
- 划分: coupon LOOCV; 每折 test = 1 coupon, rest 按标签 85/15 分层切 train/val
  (P4a fold_splits 约定, val 有稳定早停信号); 主指标 = 非PP4 逐折均值±std。
- 训练: 骨干 = general_ndt ModalAdapter + PatchTransformer (与 E1/E2 同架构),
  随机初始化 + 线性头; AdamW lr=1e-3 / wd=1e-4 / ≤80ep / cosine / grad clip 1.0 /
  class-weighted sampler (1-ratio/ratio) / val AUC 早停 (patience=20)。
- 归一化: per-sample z-score (无泄漏; 与 SSL 管线一致)。
- seed 职责分离: data_seed 固定 (划分/采样), model_seed ∈ {0,1,2} (初始化/训练随机性)。
- 汇报: 每 seed 每折 AUROC/Macro-F1/bal_acc; 主指标 = 非PP4 逐折均值±std (跨折跨 seed)。

用法:
  python scripts/general_ndt_e0_baseline.py [--config configs/general_ndt_e0.yaml]
                                            [--smoke] [--seeds 0 1 2]
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
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.adapters.base import ModalAdapter          # noqa: E402
from general_ndt.datasets.collate import collate_general_ndt  # noqa: E402
from general_ndt.datasets.registry import build_dataset     # noqa: E402
from general_ndt.datasets.schema import GeneralNDTSample    # noqa: E402
from general_ndt.models.backbone import PatchTransformer     # noqa: E402
from general_ndt.ssl.token_masks import token_valid_mask     # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NON_PP4 = ["PP3", "PP5", "PP6", "PP7"]      # 主指标逐折集合 (PP4 近零缺陷剔除)


# ---------------------------------------------------------------------------
# 模型: ModalAdapter → PatchTransformer → 线性头 (CLS pooled)
# ---------------------------------------------------------------------------
class ScratchClassifier(nn.Module):
    """E0 scratch-supervised: 随机初始化骨干 + 线性分类头。"""

    def __init__(self, d_model=128, patch_len=16, patch2d=16, n_layers_enc=4,
                 n_heads=4, n_modalities=8, n_sensors=32, dropout=0.1, n_classes=2):
        super().__init__()
        self.patch_len = patch_len
        self.patch2d = patch2d
        self.adapter = ModalAdapter(d_model=d_model, patch_len=patch_len, patch2d=patch2d,
                                    n_modalities=n_modalities, n_sensors=n_sensors)
        self.encoder = PatchTransformer(d_model=d_model, n_layers=n_layers_enc,
                                        n_heads=n_heads, dropout=dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["x"]
        tokens, grid = self.adapter(
            x, batch["shape_kind"], batch["modalities"],
            batch.get("sampling_rates"), batch.get("sensor_ids"),
        )
        tv_np = token_valid_mask(batch, self.patch_len, self.patch2d)
        tv = torch.from_numpy(tv_np).to(x.device)
        h = self.encoder(tokens, valid_mask=tv, grid=grid)
        return self.head(h[:, 0])


# ---------------------------------------------------------------------------
# 数据管线
# ---------------------------------------------------------------------------
class _NDTDataset(torch.utils.data.Dataset):
    """包 GeneralNDTSample; collate + per-sample z-score (与 SSL 管线一致)。"""

    def __init__(self, samples, normalize: bool = True):
        self.samples = list(samples)
        self.normalize = normalize
        self.labels = np.asarray([int(s.label) if s.label is not None else -1
                                  for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    @staticmethod
    def collate(items):
        nb = collate_general_ndt(items)
        if True:  # per-sample z-score (无泄漏)
            sig, vm = nb.padded_signal, nb.valid_mask
            for b in range(nb.batch_size):
                sel = sig[b, :, vm[b].astype(bool)]
                if sel.size == 0:
                    continue
                mu, sd = float(sel.mean()), float(sel.std())
                if sd < 1e-8:
                    sd = 1.0
                sig[b, :, vm[b].astype(bool)] = (sig[b, :, vm[b].astype(bool)] - mu) / sd
        return {
            "x": torch.from_numpy(nb.padded_signal).float(),
            "padded_signal": nb.padded_signal,
            "valid_mask": nb.valid_mask,
            "shape_kind": nb.shape_kind,
            "modalities": nb.modalities,
            "sampling_rates": [s.sampling_rate for s in items],
            "sensor_ids": [s.sensor_id for s in items],
            "shapes": nb.shapes,
            "sample_ids": nb.sample_ids,
            "specimen_ids": nb.specimen_ids,
            "labels": nb.labels,
        }


def e0_folds(samples, val_frac: float, data_seed: int):
    """coupon LOOCV: test = 1 coupon (非PP4 主集合), rest 85/15 分层 train/val。

    返回 [(train_idx, val_idx, test_idx, test_coupon), ...] (test 逐折主集合)。
    """
    idx_by_coupon = {c: [i for i, s in enumerate(samples) if s.specimen_id == c]
                     for c in COUPONS}
    labels = np.asarray([int(s.label) if s.label is not None else -1 for s in samples])
    folds = []
    for test_c in NON_PP4:
        test_idx = idx_by_coupon[test_c]
        rest = [i for c in COUPONS if c != test_c for i in idx_by_coupon[c]]
        rest = np.asarray(rest)
        y_rest = labels[rest]
        strat = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
        tr, va = train_test_split(rest, test_size=val_frac, random_state=data_seed,
                                  shuffle=True, stratify=strat)
        folds.append((sorted(tr.tolist()), sorted(va.tolist()),
                      sorted(test_idx), test_c))
    return folds


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def weighted_sampler(labels, seed):
    """P4a 官方采样: weight = (1-ratio, ratio), replacement=True。"""
    ratio = float(np.mean(labels == 0))
    w = np.zeros(len(labels), dtype=np.float64)
    w[labels == 0] = 1.0 - ratio
    w[labels == 1] = ratio
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def run_fold(model, samples, tr_idx, va_idx, te_idx, device, cfg, model_seed):
    """训练一折 (scratch supervised), 返回 test 折指标。"""
    torch.manual_seed(model_seed)
    train_ds = _NDTDataset([samples[i] for i in tr_idx])
    val_ds = _NDTDataset([samples[i] for i in va_idx])
    test_ds = _NDTDataset([samples[i] for i in te_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              sampler=weighted_sampler(train_ds.labels, cfg["train"]["data_seed"]),
                              collate_fn=_NDTDataset.collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            collate_fn=_NDTDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             collate_fn=_NDTDataset.collate)

    epochs = int(cfg["train"]["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    clip = float(cfg["train"].get("grad_clip", 1.0))
    patience = int(cfg["train"].get("patience", 20))
    crit = nn.CrossEntropyLoss()

    best_auc, best_state, bad = -1.0, None, 0
    val_labels = np.asarray(val_ds.labels)
    for ep in range(epochs):
        model.train()
        for tb in train_loader:
            tb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tb.items()}
            y = torch.as_tensor(tb["labels"], device=device)
            logits = model(tb)
            opt.zero_grad()
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
        sched.step()
        # val AUC
        model.eval()
        v_scores = []
        with torch.no_grad():
            for tb in val_loader:
                tb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tb.items()}
                logits = model(tb)
                v_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        v_scores = np.concatenate(v_scores)
        v_auc = float(roc_auc_score(val_labels, v_scores)) if len(set(val_labels)) == 2 else 0.0
        if v_auc > best_auc + 1e-4:
            best_auc, bad = v_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # test 评估
    model.eval()
    t_scores, t_labels = [], []
    with torch.no_grad():
        for tb in test_loader:
            tb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tb.items()}
            logits = model(tb)
            t_scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            t_labels.extend(tb["labels"])
    t_scores = np.concatenate(t_scores)
    t_labels = np.asarray(t_labels)
    y_pred = (t_scores >= 0.5).astype(int)
    return {
        "test_coupon": None,  # 由调用方填
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "best_val_auc": best_auc, "epochs_run": ep + 1,
        "auroc": float(roc_auc_score(t_labels, t_scores)) if len(set(t_labels)) > 1 else float("nan"),
        "macro_f1": float(f1_score(t_labels, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(t_labels, y_pred)),
        "test_positive_rate": float(t_labels.mean()),
    }


def make_model(cfg, model_seed):
    torch.manual_seed(model_seed)
    m = cfg["model"]
    return ScratchClassifier(
        d_model=int(m["d_model"]), patch_len=int(m["patch_len"]), patch2d=int(m["patch2d"]),
        n_layers_enc=int(m["n_layers_enc"]), n_heads=int(m["n_heads"]),
        n_modalities=int(m.get("n_modalities", 8)), n_sensors=int(m.get("n_sensors", 32)),
        dropout=float(m.get("dropout", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "general_ndt_e0.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        cfg["train"]["epochs"] = 5
        cfg["train"]["model_seeds"] = [0]
        cfg["dataset_config"]["sample_limit"] = 200
    model_seeds = args.seeds or [int(s) for s in cfg["train"]["model_seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e0] device={device} seeds={model_seeds} data_seed={cfg['train']['data_seed']}")

    samples = build_dataset(cfg["dataset"], cfg.get("dataset_config", {}))
    if not samples:
        print("[e0] 空数据集", file=sys.stderr)
        return 2
    labels = np.asarray([int(s.label) if s.label is not None else -1 for s in samples])
    print(f"[e0] loaded {len(samples)} samples; "
          f"pos_rate={labels.mean():.4f} per_coupon="
          + {c: int(labels[[i for i, s in enumerate(samples) if s.specimen_id == c]].sum())
             for c in COUPONS}.__str__())

    folds = e0_folds(samples, float(cfg["train"]["val_frac"]), int(cfg["train"]["data_seed"]))
    t0 = time.time()
    all_results = []          # 每 seed 每折
    for ms in model_seeds:
        for tr_idx, va_idx, te_idx, test_c in folds:
            model = make_model(cfg, ms).to(device)
            r = run_fold(model, samples, tr_idx, va_idx, te_idx, device, cfg, ms)
            r["test_coupon"] = test_c
            r["model_seed"] = ms
            all_results.append(r)
            print(f"  seed={ms} test={test_c}: auroc={r['auroc']:.4f} "
                  f"bal_acc={r['balanced_acc']:.4f} f1={r['macro_f1']:.4f} "
                  f"val_auc={r['best_val_auc']:.4f} ep={r['epochs_run']} "
                  f"pos_rate={r['test_positive_rate']:.3f}")

    # 聚合: 主指标 = 非PP4 逐折均值±std (跨折跨 seed)
    auroc_by_seed = {}
    for ms in model_seeds:
        auroc_by_seed[ms] = [r["auroc"] for r in all_results if r["model_seed"] == ms]
    per_seed_mean = {ms: float(np.mean([a for a in v if not np.isnan(a)]))
                     for ms, v in auroc_by_seed.items()}
    all_auroc = [r["auroc"] for r in all_results if not np.isnan(r["auroc"])]
    summary = {
        "config": str(args.config),
        "dataset": cfg["dataset"],
        "protocol": "E0 scratch-supervised, coupon LOOCV, 非PP4 逐折均值",
        "model_seeds": model_seeds,
        "data_seed": int(cfg["train"]["data_seed"]),
        "n_folds_per_seed": len(folds),
        "per_seed_mean_auroc": per_seed_mean,
        "auroc_mean": float(np.mean(all_auroc)),
        "auroc_std": float(np.std(all_auroc)),
        "per_fold_auroc": [r["auroc"] for r in all_results],
        "results": all_results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"\n[e0] per-seed 非PP4 逐折均值: { {k: round(v, 4) for k, v in per_seed_mean.items()} }")
    print(f"[e0] AUROC = {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f} "
          f"(非PP4 逐折均值, {len(all_auroc)} 折×seed)")
    out = args.out or (Path(cfg["output_dir"]) / "e0_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[e0] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
