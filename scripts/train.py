#!/usr/bin/env python
"""Unified training entrypoint for all OUR models.

Usage:
  python scripts/train.py --config configs/itformer_probe.yaml [--seed 42]
  python scripts/train.py --config configs/simple_dl.yaml --model.name lstm
  python scripts/train.py --config configs/itformer_qa_qwen3_1p7b.yaml --train.lr 1e-4

Writes experiments/runs/<model>/<seed>/{best_*.pt, train_log.json}
and   experiments/results/<model>_seed<seed>.json
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

from wndt.data.dataset import WeldCycleDataset  # noqa: E402
from wndt.data.splits import load_split_idx  # noqa: E402
from wndt.eval.metrics import majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

log = get_logger("train")


def parse_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--tag", type=str, default="",
                        help="extra run-name tag (e.g. lr sweep label)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subsets + 1 epoch for pipeline testing")
    known, extra = parser.parse_known_args()
    overrides: dict = {}
    it = iter(extra)

    def _parse_val(val: str):
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
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


def build_model(cfg, device):
    name = cfg.model.name
    if name == "itformer_qa":
        from wndt.models.time_language_model import ITFormerTLM
        m = cfg.model
        return ITFormerTLM(m.llm_path, prefix_num=m.prefix_num, d_model=m.d_model,
                           n_heads=m.n_heads, it_layers=m.it_layers,
                           enc_layers=m.enc_layers, patch_len=m.patch_len,
                           dropout=m.dropout)
    if name == "itformer_probe":
        from wndt.models.heads import ITFormerProbe
        m = cfg.model
        return ITFormerProbe(prefix_num=m.prefix_num, d_model=m.d_model,
                             n_heads=m.n_heads, it_layers=m.it_layers,
                             enc_layers=m.enc_layers, patch_len=m.patch_len,
                             dropout=m.dropout).to(device)
    if name == "encoder_only":
        from wndt.models.heads import EncoderOnly
        m = cfg.model
        return EncoderOnly(d_model=m.d_model, n_heads=m.n_heads,
                           enc_layers=m.enc_layers, patch_len=m.patch_len,
                           dropout=m.dropout).to(device)
    if name in ("mlp", "lstm", "gru"):
        from wndt.models.simple_dl import MLPClassifier, RNNClassifier
        m = cfg.model
        if name == "mlp":
            return MLPClassifier(dropout=m.dropout).to(device)
        return RNNClassifier(kind=name, hidden=m.hidden, n_layers=m.n_layers,
                             dropout=m.dropout).to(device)
    if name == "dlinear":
        from wndt.models.dlinear import DLinearClassifier
        return DLinearClassifier(kernel_size=cfg.model.kernel_size).to(device)
    if name == "timesnet":
        from wndt.models.timesnet import TimesNetClassifier
        m = cfg.model
        return TimesNetClassifier(top_k=m.top_k, e_layers=m.e_layers,
                                  d_model=m.d_model, d_ff=m.d_ff).to(device)
    raise ValueError(f"unknown model {name}")


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    if args.smoke:
        overrides.setdefault("train.epochs", 1)
        cfg = load_config(args.config, overrides)

    seed = args.seed if args.seed is not None else cfg.get("seed_list", [42])[0]
    set_seed(seed)

    processed_dir = args.repo_root / cfg.data.get("processed_dir", "data/processed")
    norm_mode = cfg.data.get("norm_mode", "global")
    idx = load_split_idx(processed_dir)
    train_ds = WeldCycleDataset(processed_dir, idx["train"], norm_mode)
    val_ds = WeldCycleDataset(processed_dir, idx["val"], norm_mode)
    test_ds = WeldCycleDataset(processed_dir, idx["test"], norm_mode)
    if args.smoke:
        # even-spaced picks so runs/both classes are mixed (first-N rows would
        # come from a single all-bad run)
        log.warning("SMOKE mode: subsampling datasets")
        for ds, n in ((train_ds, 512), (val_ds, 256), (test_ds, 256)):
            sel = np.linspace(0, len(ds.indices) - 1, n).astype(int)
            ds.indices = ds.indices[sel]
            ds.labels = ds.labels[sel]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.model.name
    llm_tag = ""
    if model_name == "itformer_qa":
        llm_tag = Path(cfg.model.llm_path).name.lower() + "_"
    tag = f"_{args.tag}" if args.tag else ""
    run_name = (f"{model_name}_{llm_tag}seed{seed}{tag}"
                + ("_smoke" if args.smoke else ""))
    run_dir = args.repo_root / "experiments/runs" / model_name / run_name
    log.info("run: %s | device: %s | norm: %s", run_name, device, norm_mode)

    t0 = time.time()
    if model_name == "itformer_qa":
        from wndt.train.trainer_qa import QATrainer
        model = build_model(cfg, device)
        tr = cfg.train
        trainer = QATrainer(model, device=device, run_dir=run_dir, lr=float(tr.lr),
                            weight_decay=float(tr.weight_decay),
                            batch_size=tr.batch_size, accum_steps=tr.get("accum_steps", 1),
                            epochs=tr.epochs, warmup_steps=tr.get("warmup_steps", 300),
                            patience=tr.get("patience", 2),
                            grad_clip=float(tr.get("grad_clip", 1.0)),
                            weighted_sampler=tr.get("weighted_sampler", True),
                            num_workers=tr.get("num_workers", 2), seed=seed)
    else:
        from wndt.train.trainer_cls import ClassificationTrainer
        model = build_model(cfg, device)
        tr = cfg.train
        trainer = ClassificationTrainer(model, device=device, run_dir=run_dir,
                                        lr=float(tr.lr),
                                        weight_decay=float(tr.weight_decay),
                                        batch_size=tr.batch_size, epochs=tr.epochs,
                                        warmup_steps=tr.get("warmup_steps", 300),
                                        patience=tr.get("patience", 8),
                                        grad_clip=float(tr.get("grad_clip", 1.0)),
                                        weighted_sampler=tr.get("weighted_sampler", True),
                                        num_workers=tr.get("num_workers", 4), seed=seed)

    fit_info = trainer.fit(train_ds, val_ds)
    test_metrics = trainer.evaluate(test_ds, "test")
    val_metrics_final = trainer.evaluate(val_ds, "val")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    result = {
        "model": model_name, "llm": cfg.model.get("llm_path", None),
        "seed": seed, "norm_mode": norm_mode, "smoke": bool(args.smoke),
        "config": dict(cfg),
        "val_metrics": val_metrics_final, "test_metrics": test_metrics,
        "val_macro_f1_best": fit_info.get("val_macro_f1_best"),
        "epochs_run": fit_info.get("epochs_run"),
        "train_wall_s": round(fit_info.get("wall_s", time.time() - t0), 1),
        "peak_vram_gb": round(fit_info.get("peak_vram_gb", 0.0), 2),
        "n_params_trainable": n_train, "n_params_total": n_total,
        "majority_baseline_test": majority_baseline(test_ds.labels),
    }
    results_dir = args.repo_root / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log.info("results written to %s", out_path)
    log.info("TEST: %s", test_metrics)


if __name__ == "__main__":
    main()
