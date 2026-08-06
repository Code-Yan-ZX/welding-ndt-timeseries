#!/usr/bin/env python
"""SAW (Submerged Arc Welding) training entrypoint.

Reuses the generic ClassificationTrainer + existing encoder/heads. Supports:
  - encoder_only : PatchTST-style encoder + linear head (from-scratch baseline)
  - moment       : pretrained MOMENT time-series foundation model wrapper
                   (frozen feature extractor + head, or fine-tuned)

Usage:
  python scripts/saw_train.py --config configs/saw_encoder.yaml
  python scripts/saw_train.py --config configs/saw_moment.yaml --seed 42

Writes experiments/results/saw_<model>_seed<seed>.json
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

from wndt.data.saw_dataset import SAWSeriesDataset  # noqa: E402
from wndt.eval.metrics import majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

log = get_logger("saw_train")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    known, extra = ap.parse_known_args()
    overrides: dict = {}
    it = iter(extra)

    def _parse_val(val: str):
        if val.lower() in ("true", "false"):
            return val.lower() == "true"
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val

    for tok in it:
        assert tok.startswith("--"), f"unexpected token {tok}"
        key = tok[2:]
        val = next(it, None)
        assert val is not None and not val.startswith("--"), f"missing value for {tok}"
        overrides[key] = _parse_val(val)
    return known, overrides


def build_model(cfg, device, n_channels, seq_len):
    name = cfg.model.name
    m = cfg.model
    n_classes = int(m.get("n_classes", 2))
    if name == "encoder_only":
        from wndt.models.heads import EncoderOnly
        return EncoderOnly(d_model=m.d_model, n_heads=m.n_heads,
                           enc_layers=m.enc_layers, patch_len=m.patch_len,
                           seq_len=seq_len, n_vars=n_channels,
                           dropout=m.dropout, n_classes=n_classes).to(device)
    if name == "moment":
        from wndt.models.moment_model import MomentClassifier
        return MomentClassifier(n_classes=n_classes, n_channels=n_channels,
                                seq_len=seq_len, ckpt=m.get("ckpt", "AutonLab/MOMENT-1-large"),
                                freeze=m.get("freeze", True), dropout=m.dropout).to(device)
    raise ValueError(f"unknown model {name}")


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    if args.smoke:
        overrides.setdefault("train.epochs", 1)
        cfg = load_config(args.config, overrides)

    seed = args.seed if args.seed is not None else cfg.get("seed_list", [42])[0]
    set_seed(seed)

    processed_dir = args.repo_root / cfg.data.get("processed_dir", "data/processed/saw")
    norm_mode = cfg.data.get("norm_mode", "global")
    splits = np.load(processed_dir / "splits.npz", allow_pickle=True)
    train_ds = SAWSeriesDataset(processed_dir, splits["train"], norm_mode)
    val_ds = SAWSeriesDataset(processed_dir, splits["val"], norm_mode)
    test_ds = SAWSeriesDataset(processed_dir, splits["test"], norm_mode)
    n_channels, seq_len = train_ds.n_channels, train_ds.seq_len
    n_classes = int(cfg.model.get("n_classes", 2))

    if args.smoke:
        log.warning("SMOKE mode: subsampling")
        for ds, n in ((train_ds, 256), (val_ds, 128), (test_ds, 128)):
            sel = np.linspace(0, len(ds.indices) - 1, min(n, len(ds))).astype(int)
            ds.indices = ds.indices[sel]
            ds.labels = ds.labels[sel]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.model.name
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"saw_{model_name}_seed{seed}{tag}" + ("_smoke" if args.smoke else "")
    run_dir = args.repo_root / "experiments/runs" / f"saw_{model_name}" / run_name
    log.info("run: %s | device: %s | C=%d L=%d classes=%d norm=%s",
             run_name, device, n_channels, seq_len, n_classes, norm_mode)

    from wndt.train.trainer_cls import ClassificationTrainer
    model = build_model(cfg, device, n_channels, seq_len)
    tr = cfg.train
    trainer = ClassificationTrainer(model, device=device, run_dir=run_dir,
                                    lr=float(tr.lr), weight_decay=float(tr.weight_decay),
                                    batch_size=tr.batch_size, epochs=tr.epochs,
                                    warmup_steps=tr.get("warmup_steps", 300),
                                    patience=tr.get("patience", 8),
                                    grad_clip=float(tr.get("grad_clip", 1.0)),
                                    weighted_sampler=tr.get("weighted_sampler", True),
                                    num_workers=tr.get("num_workers", 4), seed=seed)

    t0 = time.time()
    fit_info = trainer.fit(train_ds, val_ds)
    test_metrics = trainer.evaluate(test_ds, "test")
    val_metrics_final = trainer.evaluate(val_ds, "val")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    result = {
        "dataset": "saw", "model": model_name, "seed": seed,
        "norm_mode": norm_mode, "smoke": bool(args.smoke),
        "n_channels": n_channels, "seq_len": seq_len, "n_classes": n_classes,
        "config": dict(cfg),
        "val_metrics": val_metrics_final, "test_metrics": test_metrics,
        "val_macro_f1_best": fit_info.get("val_macro_f1_best"),
        "epochs_run": fit_info.get("epochs_run"),
        "train_wall_s": round(fit_info.get("wall_s", time.time() - t0), 1),
        "n_params_trainable": n_train, "n_params_total": n_total,
        "majority_baseline_test": majority_baseline(test_ds.labels),
    }
    results_dir = args.repo_root / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log.info("results -> %s", out_path)
    log.info("TEST: %s", test_metrics)


if __name__ == "__main__":
    main()
