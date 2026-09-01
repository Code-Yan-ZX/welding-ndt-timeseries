"""严格划分 + 冻结表征 linear probe (最小实现)。

- leave_one_specimen_split: 按 specimen_id 做 leave-one-out 折 (禁止同试件跨 split)。
- logistic_probe: 冻结表征上训逻辑回归 (CPU 可跑), 返回 AUROC/Macro-F1/balanced acc。
严格原则 (实验协议 §〇):
  主指标 = 逐折均值±std; 同试件切片绝不随机跨 train/test。
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from general_ndt.datasets.schema import GeneralNDTSample


def leave_one_specimen_split(
    samples: Sequence[GeneralNDTSample],
    test_specimens: Iterable[str] | None = None,
) -> list[tuple[list[int], list[int], list[int]]]:
    """按 specimen_id 生成 LOOCV 折: [(train_idx, val_idx, test_idx), ...]。

    每折: test = 一个 specimen; val = 另一个 specimen; train = 其余。
    未指定 test_specimens 时逐个 specimen 作 test (非PP4 过滤由调用方决定)。
    """
    specimens = sorted({s.specimen_id for s in samples if s.specimen_id})
    if test_specimens is not None:
        specimens = [sp for sp in specimens if sp in set(test_specimens)]

    folds = []
    for test_sp in specimens:
        test_idx = [i for i, s in enumerate(samples) if s.specimen_id == test_sp]
        rest = [i for i, s in enumerate(samples) if s.specimen_id != test_sp]
        if not rest:
            continue
        # val = rest 中样本数最多的 specimen
        val_sp = max(
            {s.specimen_id for i in rest for s in [samples[i]]},
            key=lambda sp: sum(1 for i in rest if samples[i].specimen_id == sp),
        )
        val_idx = [i for i in rest if samples[i].specimen_id == val_sp]
        train_idx = [i for i in rest if samples[i].specimen_id != val_sp]
        if not train_idx:
            continue
        folds.append((train_idx, val_idx, test_idx))
    return folds


def logistic_probe(
    features: np.ndarray, labels: np.ndarray, train_idx: Sequence[int], test_idx: Sequence[int]
) -> dict:
    """冻结表征上的线性 probe (逻辑回归, 规范头协议: class_weight balanced)。

    输入 features: (N, d); labels: (N,) int。
    返回 AUROC / Macro-F1 / balanced acc。
    """
    Xtr, ytr = features[train_idx], labels[train_idx]
    Xte, yte = features[test_idx], labels[test_idx]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    y_prob = clf.predict_proba(Xte)
    y_pred = clf.predict(Xte)
    out = {
        "auroc": float(roc_auc_score(yte, y_prob[:, 1])) if len(set(yte)) > 1 else float("nan"),
        "macro_f1": float(f1_score(yte, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(yte, y_pred)),
        "acc": float(accuracy_score(yte, y_pred)),
    }
    return out


def summarize_folds(fold_results: list[dict], metric: str = "auroc") -> dict:
    """逐折结果 → 均值±std (主指标汇报格式)。"""
    vals = [r[metric] for r in fold_results if r.get(metric) is not None and not np.isnan(r[metric])]
    if not vals:
        return {metric: float("nan"), f"{metric}_std": float("nan"), "n_folds": 0}
    return {
        metric: float(np.mean(vals)),
        f"{metric}_std": float(np.std(vals)),
        "n_folds": len(vals),
        "per_fold": vals,
    }
