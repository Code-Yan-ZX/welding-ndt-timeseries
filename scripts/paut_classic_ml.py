#!/usr/bin/env python
"""Classic-ML baselines (RF / XGBoost) on handcrafted A-scan features for PAUT.

Operates on the per-position max-over-beams envelope (env.npy, shape (N, 512))
of the 71-degree PAUT group.  Features per position:
  - time-domain stats (mean/std/rms/peak/crest/skew/kurt/...)  [14]
  - FFT spectral features (band energies / centroid / entropy / dom freq /
    rolloff / peak mag)                                        [9]
  - PAUT-specific: peak amplitude, peak depth (argmax), near-field vs
    backwall energy ratio, count of threshold crossings        [~5]
Fits on train (PP3/4/5), evaluates once on test (PP7); val (PP6) reported too.
Mirrors the schema of saw_classic_ml.py for a direct comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from wndt.features.handcrafted import _time_features, _spectral_features  # noqa: E402
from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402

# Original A-scan: 3500 samples @ 40 ns -> 140 us window; downsampled to 512 by
# max-pooling, so the 512 bins still span 140 us -> effective fs = 512/140e-6.
FS = 512 / 140e-6
NEAR_SPLIT = 256   # first 256 bins ~ 0-70 mm (defect / fill zone); rest ~ backwall


def extract_paut_features(env: np.ndarray) -> np.ndarray:
    """env: (N, 512) -> (N, F) float32."""
    x = env.astype(np.float32)
    feats = [_time_features(x), _spectral_features(x, fs=FS)]
    # PAUT-specific
    mx = x.max(axis=1)
    argmax = x.argmax(axis=1).astype(np.float32) / x.shape[1]
    energy = (x ** 2).sum(axis=1) + 1e-12
    near = (x[:, :NEAR_SPLIT] ** 2).sum(axis=1)
    far = (x[:, NEAR_SPLIT:] ** 2).sum(axis=1)
    near_ratio = near / (energy + 1e-12)
    far_ratio = far / (energy + 1e-12)
    thr = 0.1 * mx[:, None]
    crossings = (x > thr).sum(axis=1).astype(np.float32) / x.shape[1]
    feats.append(np.stack([mx, argmax, near_ratio, far_ratio, crossings], axis=1))
    out = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    processed = REPO / "data/processed/paut"
    env = np.load(processed / "env.npy")
    labels = np.load(processed / "meta_label.npy").astype(int)
    splits = np.load(processed / "splits.npz", allow_pickle=True)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print("extracting PAUT features ...")
    X = extract_paut_features(env)
    print(f"features: {X.shape} | fs={FS/1e6:.3f} MHz")
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    results_dir = REPO / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost as xgb
        have_xgb = True
    except Exception:
        have_xgb = False

    models = {"classic_rf": RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                    class_weight="balanced", random_state=42)}
    if have_xgb:
        models["classic_xgb"] = xgb.XGBClassifier(n_estimators=500, max_depth=6,
                                                   learning_rate=0.1, n_jobs=-1,
                                                   eval_metric="logloss",
                                                   tree_method="hist",
                                                   random_state=42)

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(Xs[tr], labels[tr])
        score_te = clf.predict_proba(Xs[te])[:, 1]
        pred_te = (score_te > 0.5).astype(int)
        test_m = compute_metrics(labels[te], pred_te, score_te)
        score_va = clf.predict_proba(Xs[va])[:, 1]
        val_m = compute_metrics(labels[va], (score_va > 0.5).astype(int), score_va)
        res = {"dataset": "paut", "model": name, "seed": 42, "n_classes": 2,
               "n_features": int(X.shape[1]),
               "val_metrics": val_m, "test_metrics": test_m,
               "train_wall_s": round(time.time() - t0, 1),
               "majority_baseline_test": majority_baseline(labels[te])}
        out = results_dir / f"paut_{name}_seed42.json"
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"{name}: val(f1m {val_m['f1_macro']:.4f} auc {val_m.get('auc',0):.4f}) | "
              f"TEST acc {test_m['acc']:.4f} f1bin {test_m['f1_bin']:.4f} "
              f"f1m {test_m['f1_macro']:.4f} auc {test_m.get('auc',0):.4f} -> {out}")


if __name__ == "__main__":
    main()
