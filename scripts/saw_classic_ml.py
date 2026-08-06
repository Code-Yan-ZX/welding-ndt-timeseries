#!/usr/bin/env python
"""Classic-ML baselines (RF / XGBoost / SVM) on handcrafted features for SAW.

Per channel (current_a/b, voltage_a/b): time-domain stats + FFT spectral
features (reusing wndt.features.handcrafted). Cross-channel: pairwise Pearson
correlations. ~70 features. Fits on train (PP3/4/5), selects on val (PP6),
evaluates once on test (PP7).
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

FS = 5000.0


def extract_saw_features(waves: np.ndarray) -> np.ndarray:
    """waves: (N, 4, 512) -> (N, F) float32."""
    n, C, L = waves.shape
    feats = []
    for c in range(C):
        x = waves[:, c, :]
        feats.append(_time_features(x))
        feats.append(_spectral_features(x, fs=FS))
    # pairwise correlations across channels
    centered = waves - waves.mean(axis=2, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(axis=2)) + 1e-12
    corr_feats = []
    for i in range(C):
        for j in range(i + 1, C):
            dot = (centered[:, i, :] * centered[:, j, :]).sum(axis=1)
            corr = dot / (norm[:, i] * norm[:, j])
            corr_feats.append(corr)
    if corr_feats:
        feats.append(np.stack(corr_feats, axis=1))
    out = np.concatenate(feats, axis=1).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def main():
    processed = Path("data/processed/saw")
    waves = np.load(processed / "waves.npy", mmap_mode="r")
    labels = np.load(processed / "meta_label.npy")
    splits = np.load(processed / "splits.npz", allow_pickle=True)
    print("extracting features ...")
    X = extract_saw_features(np.asarray(waves))
    y = labels.astype(int)
    print(f"features: {X.shape}")
    tr, va, te = splits["train"], splits["val"], splits["test"]
    # standardize on train
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
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
                                                   tree_method="hist")
    # SVM on 100k samples is O(n^2)-ish and very slow; subsample to 20k for a
    # comparable-in-time RBF-SVM baseline.
    svm_sub = 20000

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(Xs[tr], y[tr])
        if name == "classic_svm":
            score = clf.decision_function(Xs[te])
        elif name == "classic_xgb":
            score = clf.predict_proba(Xs[te])[:, 1]
        else:
            score = clf.predict_proba(Xs[te])[:, 1]
        pred = (score > 0.5).astype(int) if name != "classic_svm" else (score > 0).astype(int)
        test_m = compute_metrics(y[te], pred, score)
        va_score = (clf.decision_function(Xs[va]) if name == "classic_svm"
                    else clf.predict_proba(Xs[va])[:, 1])
        va_pred = (va_score > 0.5).astype(int) if name != "classic_svm" else (va_score > 0).astype(int)
        val_m = compute_metrics(y[va], va_pred, va_score)
        res = {"dataset": "saw", "model": name, "seed": 42, "n_classes": 2,
               "val_metrics": val_m, "test_metrics": test_m,
               "train_wall_s": round(time.time() - t0, 1),
               "majority_baseline_test": majority_baseline(y[te])}
        out = results_dir / f"saw_{name}_seed42.json"
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"{name}: val(f1m {val_m['f1_macro']:.4f} auc {val_m.get('auc',0):.4f}) | "
              f"TEST acc {test_m['acc']:.4f} f1bin {test_m['f1_bin']:.4f} f1m {test_m['f1_macro']:.4f} "
              f"auc {test_m.get('auc',0):.4f} -> {out}")


if __name__ == "__main__":
    main()
