#!/usr/bin/env python
"""PAUT (phased-array ultrasonic) training entrypoint.

Reuses the generic ClassificationTrainer + existing encoder head.  Input is a
per-position PAUT A-scan.  ``encoder_only`` trains on per-beam samples
(N_pos * 49, SAW-scale) and is evaluated position-level by taking the MAX
per-beam P(defect) over the 49-beam aperture - physically "a position is
defective if any beam sees an echo", and it avoids the train/eval
distribution mismatch of feeding the max-envelope to a beam-trained model.
``ssf`` trains/evals on the full (49, 512) B-scan (one prediction per
position).  Early stopping uses val AUC (threshold-free) because the val coupon
(PP6, 76 % defect) and test coupon (PP7, 14 %) have very different defect
rates, which biases threshold-based metrics.

Usage:
  python scripts/paut_train.py --config configs/paut_encoder.yaml
  python scripts/paut_train.py --config configs/paut_ssf.yaml --seed 42

Writes experiments/results/paut_<model>_seed<seed>.json
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

from wndt.data.paut_dataset import PAUTSeriesDataset  # noqa: E402
from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

log = get_logger("paut_train")


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
    if name == "ssf":
        from wndt.models.ssf import SSFClassifier
        return SSFClassifier(n_beams=n_channels, seq_len=seq_len,
                             d_model=m.d_model, dropout=m.dropout,
                             n_classes=n_classes).to(device)
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def position_scores(model, ascans, indices, ts_mean, ts_std, device, mode):
    """Position-level defect scores (one per position).

    mode="bscan": run the full (49, T) B-scan through the model, P(defect).
    """
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    for pi, gi in enumerate(indices):
        full = np.array(ascans[gi], dtype=np.float32)        # (49, T)
        full = (full - ts_mean) / ts_std
        x = torch.from_numpy(full).unsqueeze(0).to(device)   # (1, 49, T)
        scores[pi] = float(torch.softmax(model(x), 1)[0, 1].item())
    return scores


def best_threshold(y_val, s_val):
    """Threshold maximizing macro-F1 on the val scores (calibration for the
    val/test defect-rate mismatch: PP6 is 76 % defect, PP7 is 14 %, so a fixed
    0.5 cut is badly miscalibrated)."""
    from sklearn.metrics import f1_score
    cand = np.unique(s_val)
    if len(cand) > 200:
        cand = np.quantile(s_val, np.linspace(0.01, 0.99, 200))
    best_t, best_f = 0.5, -1.0
    for t in cand:
        f = f1_score(y_val, (s_val > t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    if args.smoke:
        overrides.setdefault("train.epochs", 1)
        cfg = load_config(args.config, overrides)

    seed = args.seed if args.seed is not None else cfg.get("seed_list", [42])[0]
    set_seed(seed)

    processed = args.repo_root / cfg.data.get("processed_dir", "data/processed/paut")
    norm_mode = cfg.data.get("norm_mode", "per_timestep")
    splits = np.load(processed / "splits.npz", allow_pickle=True)
    model_name = cfg.model.name
    # All models are position-level (the defect label is position-level; we have
    # no per-beam labels, so per-beam training would mislabel the ~47 echo-free
    # beams at a defect position).  encoder/ssf consume the full (49, T) B-scan:
    #   - encoder: EncoderOnly's VarAttention over the 49 channels learns to
    #     spotlight the echo-bearing beam (attention-MIL over the aperture);
    #   - ssf: the spectral-spatial-frequency CNN sees all beams at once.
    # Both produce one prediction per position -> consistent position-level eval.
    beam_tr = beam_va = beam_te = "bscan"
    eval_mode = "bscan"
    train_ds = PAUTSeriesDataset(processed, splits["train"], beam=beam_tr, norm_mode=norm_mode)
    val_ds = PAUTSeriesDataset(processed, splits["val"], beam=beam_va, norm_mode=norm_mode)
    test_ds = PAUTSeriesDataset(processed, splits["test"], beam=beam_te, norm_mode=norm_mode)
    n_channels, seq_len = train_ds.n_channels, train_ds.seq_len
    n_classes = int(cfg.model.get("n_classes", 2))

    if args.smoke:
        log.warning("SMOKE mode: subsampling")
        for ds, n in ((train_ds, 512), (val_ds, 256), (test_ds, 256)):
            if ds.beam == "expand":
                orig = ds.labels[::ds.n_beams]                      # (n_pos,)
                sel = np.linspace(0, len(ds.indices) - 1,
                                  min(n, len(ds.indices))).astype(int)
                ds.indices = ds.indices[sel]
                ds.labels = np.repeat(orig[sel], ds.n_beams)
            else:
                n = min(n, len(ds))
                sel = np.linspace(0, len(ds) - 1, n).astype(int)
                ds.indices = ds.indices[sel % len(ds.indices)]
                ds.labels = ds.labels[sel]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"paut_{model_name}_seed{seed}{tag}" + ("_smoke" if args.smoke else "")
    run_dir = args.repo_root / "experiments/runs" / f"paut_{model_name}" / run_name
    log.info("run: %s | device: %s | beam tr/va/te=%s/%s/%s | C=%d L=%d | "
             "train=%d val=%d test=%d", run_name, device, beam_tr, beam_va, beam_te,
             n_channels, seq_len, len(train_ds), len(val_ds), len(test_ds))

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
                                    num_workers=tr.get("num_workers", 4), seed=seed,
                                    monitor=tr.get("monitor", "auc"))

    t0 = time.time()
    fit_info = trainer.fit(train_ds, val_ds)

    # position-level evaluation: tune threshold on val (macro-F1), apply to test.
    # AUC is threshold-free and is the primary cross-coupon metric; the
    # val-tuned threshold gives calibrated f1/acc despite the val/test
    # defect-rate mismatch (PP6 76 % vs PP7 14 %).
    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    labels_all = np.load(processed / "meta_label.npy")
    with open(processed / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
    val_scores = position_scores(model, ascans, splits["val"], ts_mean, ts_std,
                                 device, eval_mode)
    test_scores = position_scores(model, ascans, splits["test"], ts_mean, ts_std,
                                  device, eval_mode)
    y_val = labels_all[splits["val"]]
    y_test = labels_all[splits["test"]]
    thr = best_threshold(y_val, val_scores)
    test_metrics = compute_metrics(y_test, (test_scores > thr).astype(int), test_scores)
    val_metrics_final = compute_metrics(y_val, (val_scores > thr).astype(int), val_scores)
    test_metrics["threshold"] = thr
    log.info("[test pos] thr=%.3f %s", thr, test_metrics)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    result = {
        "dataset": "paut", "model": model_name, "seed": seed,
        "norm_mode": norm_mode, "smoke": bool(args.smoke),
        "beam_train": beam_tr, "eval_mode": eval_mode,
        "n_channels": n_channels, "seq_len": seq_len, "n_classes": n_classes,
        "config": dict(cfg),
        "val_metrics": val_metrics_final, "test_metrics": test_metrics,
        "val_macro_f1_best": fit_info.get("val_macro_f1_best"),
        "epochs_run": fit_info.get("epochs_run"),
        "train_wall_s": round(fit_info.get("wall_s", time.time() - t0), 1),
        "n_params_trainable": n_train, "n_params_total": n_total,
        "majority_baseline_test": majority_baseline(labels_all[splits["test"]]),
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
