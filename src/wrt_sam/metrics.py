"""评估指标: Recall / Precision / AUC(PR 曲线下面积) / IoU, 对齐 WRT-SAM Table 1-7.

两种聚合口径 (论文未说明, 两种都报):
  micro: 全部图像像素拼接后计算 (大面积图主导)
  macro: 先逐图计算再取均值
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _cls(recall: float, precision: float, iou: float) -> dict:
    return {"recall": recall, "precision": precision, "iou": iou}


def image_metrics(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> dict:
    """单图指标. pred 为 sigmoid 概率图 (0..1), gt 为 0/1."""
    b = pred > thresh
    g = gt > 0.5
    inter = np.logical_and(b, g).sum()
    union = np.logical_or(b, g).sum()
    tp = inter
    fp = np.logical_and(b, ~g).sum()
    fn = np.logical_and(~b, g).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou = inter / union if union > 0 else 1.0
    return {"recall": recall, "precision": precision, "iou": iou,
            "auc": average_precision_score(g.ravel(), pred.ravel()) if g.any() else float("nan")}


def aggregate(results: list[dict], preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """micro + macro 聚合."""
    macro = {k: float(np.nanmean([r[k] for r in results])) for k in ("recall", "precision", "iou", "auc")}
    p = np.concatenate([x.ravel() for x in preds])
    g = np.concatenate([x.ravel() for x in gts])
    b = p > 0.5
    gg = g > 0.5
    inter = np.logical_and(b, gg).sum()
    union = np.logical_or(b, gg).sum()
    tp, fp, fn = inter, np.logical_and(b, ~gg).sum(), np.logical_and(~b, gg).sum()
    micro = {
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "iou": float(inter / max(union, 1)),
        "auc": float(average_precision_score(gg.astype(int), p)),
    }
    return {"micro": micro, "macro": macro}
