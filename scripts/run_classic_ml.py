#!/usr/bin/env python
"""Classic ML baselines: Random Forest / XGBoost / SVM on handcrafted features.

Features are computed from RAW waves; StandardScaler is fit on TRAIN features
only. SVM's C is selected on val macro-F1 (inner selection).

Usage: python scripts/run_classic_ml.py [--models rf,xgb,svm]
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from wndt.data.splits import load_split_idx  # noqa: E402
from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402
from wndt.features.handcrafted import extract_features  # noqa: E402
from wndt.utils.logging import get_logger  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

log = get_logger("classic_ml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,xgb,svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    args = ap.parse_args()
    set_seed(args.seed)

    proc = args.repo_root / "data/processed"
    idx = load_split_idx(proc)
    waves = np.load(proc / "waves.npy", mmap_mode="r")
    labels = np.asarray(np.load(proc / "meta_label.npy", mmap_mode="r"))

    log.info("extracting features ...")
    t0 = time.time()
    X_all = extract_features(np.asarray(waves))
    log.info("features %s in %.1fs", X_all.shape, time.time() - t0)

    X_tr, y_tr = X_all[idx["train"]], labels[idx["train"]]
    X_va, y_va = X_all[idx["val"]], labels[idx["val"]]
    X_te, y_te = X_all[idx["test"]], labels[idx["test"]]

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    wanted = [m.strip() for m in args.models.split(",")]
    results_dir = args.repo_root / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_base = args.repo_root / "experiments/runs/classic_ml"

    for name in wanted:
        t0 = time.time()
        log.info("=== %s ===", name)
        if name == "rf":
            model = RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                           random_state=args.seed)
            model.fit(X_tr, y_tr)
        elif name == "xgb":
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1,
                                      max_depth=6, tree_method="hist",
                                      device="cpu", random_state=args.seed,
                                      eval_metric="logloss")
            model.fit(X_tr, y_tr)
        elif name == "svm":
            # probability=False: Platt calibration costs as much as a second
            # SVM fit; decision_function scores are used for AUC instead
            best_c, best_f1, best_model = None, -1.0, None
            for C in (0.1, 1.0, 10.0):
                m = SVC(kernel="rbf", gamma="scale", C=C, random_state=args.seed,
                        probability=False)
                m.fit(X_tr, y_tr)
                val_m = compute_metrics(y_va, m.predict(X_va),
                                        m.decision_function(X_va))
                log.info("  SVM C=%s val: %s", C, val_m)
                if val_m["f1_macro"] > best_f1:
                    best_c, best_f1, best_model = C, val_m["f1_macro"], m
            model = best_model
            log.info("  selected C=%s", best_c)
        else:
            raise ValueError(name)

        wall = time.time() - t0
        y_pred = model.predict(X_te)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
            va_score = model.predict_proba(X_va)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_te)
            va_score = model.decision_function(X_va)
        else:
            y_score = va_score = None
        test_m = compute_metrics(y_te, y_pred, y_score)
        val_m = compute_metrics(y_va, model.predict(X_va), va_score)
        log.info("[%s] TEST %s | wall %.1fs", name, test_m, wall)

        run_dir = run_base / f"{name}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "model": f"classic_{name}", "llm": None, "seed": args.seed,
            "norm_mode": "raw+features", "smoke": False,
            "config": {"n_features": int(X_tr.shape[1])},
            "val_metrics": val_m, "test_metrics": test_m,
            "val_macro_f1_best": val_m["f1_macro"], "epochs_run": None,
            "train_wall_s": round(wall, 1), "peak_vram_gb": 0.0,
            "n_params_trainable": 0, "n_params_total": 0,
            "majority_baseline_test": majority_baseline(y_te),
        }
        with open(results_dir / f"classic_{name}_seed{args.seed}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
