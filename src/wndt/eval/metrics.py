"""Metrics shared by all models: accuracy, binary F1 (positive=good),
macro F1, AUC, plus the majority-class baseline."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_bin": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
    return out


def majority_baseline(y_true: np.ndarray) -> dict[str, float]:
    """Predict the majority class everywhere (reported as a reference row)."""
    y_true = np.asarray(y_true).astype(int)
    maj = int(np.round(y_true.mean()))          # 1 if pos-rate > 0.5 else 0
    y_pred = np.full_like(y_true, maj)
    return compute_metrics(y_true, y_pred)
