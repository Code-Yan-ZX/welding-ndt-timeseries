#!/usr/bin/env python
"""General NDT Foundation — 冻结编码 linear probe (严格逐折, Phase 2A)。

用法:
  python scripts/general_ndt_probe.py --ckpt experiments/runs/general_ndt_mae_smoke/mae_step300.pt
                                      [--config configs/general_ndt_mae_smoke.yaml]
                                      [--exclude-specimens PP4]

- leave-one-specimen 划分 (coupon LOOCV), 主指标 = 非PP4 逐折均值±std (记录每折, 不用 pooled)。
- 冻结 encoder 的 CLS pooled 表征 + logistic probe (规范头协议 class_weight balanced)。
- checkpoint 数据集指纹与当前数据不一致时报错。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.registry import build_dataset                       # noqa: E402
from general_ndt.evaluation.probe import leave_one_specimen_split, logistic_probe  # noqa: E402
from general_ndt.models.mae import MaskedAutoencoder                           # noqa: E402
from general_ndt.trainers.ssl_trainer import SSLTrainer, dataset_fingerprint   # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "general_ndt_mae_smoke.yaml"


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--exclude-specimens", type=str, default="PP4",
                    help="逗号分隔, 从逐折主结果中排除的 coupon (默认 PP4)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    samples = build_dataset(cfg["dataset"], cfg.get("dataset_config", {}))
    if not samples:
        print("[probe] 空数据集", file=sys.stderr)
        return 2
    fp = dataset_fingerprint(samples)
    print(f"[probe] loaded {len(samples)} samples; fingerprint={fp}")

    m = cfg["model"]
    model = MaskedAutoencoder(
        d_model=int(m["d_model"]), patch_len=int(m["patch_len"]), patch2d=int(m["patch2d"]),
        n_layers_enc=int(m["n_layers_enc"]), n_heads=int(m["n_heads"]),
        d_decoder=int(m["d_decoder"]), n_layers_dec=int(m["n_layers_dec"]),
        mask_ratio=float(m["mask_ratio"]), n_modalities=int(m.get("n_modalities", 8)),
        n_sensors=int(m.get("n_sensors", 32)), dropout=float(m.get("dropout", 0.0)),
    )
    tr = cfg["train"]
    trainer = SSLTrainer(
        model, dict(tr),
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    trainer.load_checkpoint(args.ckpt, expected_fingerprint=fp)
    print(f"[probe] checkpoint loaded: {args.ckpt}")

    feats, ids = trainer.extract_features(samples, batch_size=32)
    print(f"[probe] features: {feats.shape}")

    # 严格逐折: leave-one-specimen (默认排除 PP4)
    exclude = [x for x in args.exclude_specimens.split(",") if x] or None
    specimens = sorted({s.specimen_id for s in samples})
    test_specs = [sp for sp in specimens if sp not in (exclude or [])]
    folds = leave_one_specimen_split(samples, test_specimens=test_specs)
    labels = np.asarray([int(s.label) if s.label is not None else -1 for s in samples])
    if not folds:
        print("[probe] 无有效折", file=sys.stderr)
        return 2

    results = []
    for train_idx, val_idx, test_idx in folds:
        test_sp = {samples[i].specimen_id for i in test_idx}
        r = logistic_probe(feats, labels, train_idx, test_idx)
        r["test_specimen"] = sorted(test_sp)
        r["n_train"] = len(train_idx)
        r["n_test"] = len(test_idx)
        results.append(r)
        print(f"  fold test={sorted(test_sp)}: "
              f"auroc={r['auroc']:.4f} bal_acc={r['balanced_acc']:.4f} f1={r['macro_f1']:.4f}")

    auroc_vals = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    summary = {
        "dataset": cfg["dataset"],
        "ckpt": str(args.ckpt),
        "fingerprint": fp,
        "n_folds": len(results),
        "excluded_specimens": exclude or [],
        "per_fold": results,
        "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
        "auroc_std": float(np.std(auroc_vals)) if auroc_vals else float("nan"),
        "per_fold_auroc": [r["auroc"] for r in results],
    }
    print(f"\n[probe] AUROC = {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f} "
          f"(n_folds={summary['n_folds']}, 逐折均值)")
    out_path = args.out or (Path(cfg["output_dir"]) / "probe_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[probe] results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
