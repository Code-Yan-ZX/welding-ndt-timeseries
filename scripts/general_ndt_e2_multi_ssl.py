#!/usr/bin/env python
"""General NDT Foundation — E2 多源 SSL: PENELOPE(超声) + EddyCus(涡流) 联合预训练 + 冻结探针。

协议 (严格, 对齐项目纪律):
- **per-fold 严格预训练**: 对每个 test coupon T (非PP4), 联合预训练只使用
  4 个非 T 的 PENELOPE coupon + 全部 EddyCus (均无标签) —— test coupon 信号不进预训练。
- **模态平衡采样**: 1:1 确定性交替 (train_multi), PENELOPE 曝光 3000 步 (= E1 相同),
  EddyCus 小源过采样 3000 步 → 隔离"多源"效应。
- 冻结 encoder 后, 在 E0/E1 同划分 (rest 85/15 分层 + test=coupon) 上做 PENELOPE
  logistic 探针。
- 主指标 = 非PP4 逐折均值 ± std (4 折 × 3 seed), 对照 E1 0.5680 / E0 0.5254。
- seed 职责分离: model_seed / data_seed。

用法:
  python scripts/general_ndt_e2_multi_ssl.py [--config configs/general_ndt_e2_multi_ssl.yaml]
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
import yaml
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.registry import build_dataset                    # noqa: E402
from general_ndt.evaluation.probe import logistic_probe                    # noqa: E402
from general_ndt.models.mae import MaskedAutoencoder                       # noqa: E402
from general_ndt.trainers.ssl_trainer import SSLTrainer                    # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NON_PP4 = ["PP3", "PP5", "PP6", "PP7"]


def make_model(cfg, model_seed):
    torch.manual_seed(model_seed)
    m = cfg["model"]
    return MaskedAutoencoder(
        d_model=int(m["d_model"]), patch_len=int(m["patch_len"]), patch2d=int(m["patch2d"]),
        n_layers_enc=int(m["n_layers_enc"]), n_heads=int(m["n_heads"]),
        d_decoder=int(m["d_decoder"]), n_layers_dec=int(m["n_layers_dec"]),
        mask_ratio=float(m["mask_ratio"]), n_modalities=int(m.get("n_modalities", 8)),
        n_sensors=int(m.get("n_sensors", 32)), dropout=float(m.get("dropout", 0.0)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "general_ndt_e2_multi_ssl.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        cfg["ssl"]["n_steps"] = 60
        cfg["ssl"]["model_seeds"] = [0]
        cfg["dataset_config"]["sample_limit"] = 300
        cfg["eddycus_config"]["sample_limit"] = 60
    model_seeds = args.seeds or [int(s) for s in cfg["ssl"]["model_seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[e2] device={device} seeds={model_seeds} data_seed={cfg['ssl']['data_seed']} "
          f"n_steps={cfg['ssl']['n_steps']}")

    samples = build_dataset(cfg["dataset"], cfg.get("dataset_config", {}))
    if not samples:
        print("[e2] 空 PENELOPE", file=sys.stderr)
        return 2
    eddy = build_dataset("eddycus", cfg.get("eddycus_config", {}))
    if not eddy:
        print("[e2] 空 EddyCus", file=sys.stderr)
        return 2
    labels = np.asarray([int(s.label) if s.label is not None else -1 for s in samples])
    print(f"[e2] PENELOPE {len(samples)} samples (pos_rate={labels.mean():.4f}); "
          f"EddyCus {len(eddy)} samples (无标签, exploratory_flat_1d)")

    val_frac = float(cfg["probe"].get("val_frac", 0.15))
    data_seed = int(cfg["ssl"]["data_seed"])
    idx_by_coupon = {c: [i for i, s in enumerate(samples) if s.specimen_id == c]
                     for c in COUPONS}
    t0 = time.time()
    all_results = []
    for test_c in NON_PP4:
        te_idx = idx_by_coupon[test_c]
        rest = np.asarray([i for c in COUPONS if c != test_c for i in idx_by_coupon[c]])
        y_rest = labels[rest]
        strat = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
        tr, va = train_test_split(rest, test_size=val_frac, random_state=data_seed,
                                  shuffle=True, stratify=strat)
        tr_idx, va_idx = sorted(tr.tolist()), sorted(va.tolist())
        pretrain_pen = [samples[i] for i in sorted(set(tr_idx) | set(va_idx))]  # 4 非 test coupon
        for ms in model_seeds:
            model = make_model(cfg, ms).to(device)
            trainer_cfg = dict(cfg["ssl"])
            trainer_cfg["model_seed"] = ms
            trainer_cfg["data_seed"] = data_seed
            trainer = SSLTrainer(model, trainer_cfg, device=device)
            # 联合预训练: PENELOPE(非 test) + EddyCus(全部), 1:1 交替
            ckpt = trainer.train_multi(
                [("penelope", pretrain_pen), ("eddycus", eddy)],
                n_steps=int(cfg["ssl"]["n_steps"]),
                batch_size=int(cfg["ssl"]["batch_size"]),
                log_every=int(cfg["ssl"].get("log_every", 100)),
                ckpt_every=int(cfg["ssl"].get("ckpt_every", 10**9)),
                output_dir=Path(cfg["output_dir"]) / f"ckpts/{test_c}_seed{ms}",
            )
            # 冻结特征 (仅 PENELOPE eval 集) + logistic 探针
            eval_samples = [samples[i] for i in tr_idx + va_idx + te_idx]
            feats, ids = trainer.extract_features(eval_samples, batch_size=64)
            feat_by_id = dict(zip(ids, feats))
            F = np.stack([feat_by_id[samples[i].sample_id] for i in tr_idx + te_idx])
            y_sub = np.asarray([labels[i] for i in tr_idx + te_idx])
            tr_pos = list(range(len(tr_idx)))
            te_pos = list(range(len(tr_idx), len(tr_idx) + len(te_idx)))
            r = logistic_probe(F, y_sub, tr_pos, te_pos)
            r["test_coupon"] = test_c
            r["model_seed"] = ms
            r["ckpt"] = str(ckpt)
            all_results.append(r)
            print(f"  seed={ms} test={test_c}: auroc={r['auroc']:.4f} "
                  f"bal_acc={r['balanced_acc']:.4f} f1={r['macro_f1']:.4f} "
                  f"pos_rate={labels[te_idx].mean():.3f}")

    auroc_by_seed = {ms: [r["auroc"] for r in all_results if r["model_seed"] == ms]
                     for ms in model_seeds}
    per_seed_mean = {str(ms): float(np.mean([a for a in v if not np.isnan(a)]))
                     for ms, v in auroc_by_seed.items()}
    all_auroc = [r["auroc"] for r in all_results if not np.isnan(r["auroc"])]
    summary = {
        "config": str(args.config),
        "dataset": cfg["dataset"],
        "protocol": "E2 per-fold strict multi-source MAE (PENELOPE+EddyCus) + frozen probe",
        "model_seeds": model_seeds,
        "data_seed": data_seed,
        "n_folds_per_seed": len(NON_PP4),
        "per_seed_mean_auroc": per_seed_mean,
        "auroc_mean": float(np.mean(all_auroc)),
        "auroc_std": float(np.std(all_auroc)),
        "per_fold_auroc": [r["auroc"] for r in all_results],
        "results": all_results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"\n[e2] per-seed 非PP4 逐折均值: { {k: round(v, 4) for k, v in per_seed_mean.items()} }")
    print(f"[e2] AUROC = {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f} "
          f"(非PP4 逐折均值, {len(all_auroc)} 折×seed)")
    out = args.out or (Path(cfg["output_dir"]) / "e2_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[e2] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
