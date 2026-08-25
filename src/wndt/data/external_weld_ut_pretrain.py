"""M0-3 真实焊缝多源 SSL 数据（外部 FMC views + PAUT 目标域 views）。

核心约定（P-long 与 W→P **完全一致**，只差阶段 1 的数据源）：

1. **view**：FMC 每个 transmit event = 1 个 view（Rx×time 二维物理结构），
   view 继承原 specimen/group_id；PAUT 每个 train-coupon 位置 = 1 个 view
   （49×512 B-scan）。split/group 永远按物理试件/折划分。
2. **变长输入**：FMC 按 (Rx, T) bucket，同尺寸才同批；重建 loss 只统计
   masked∩valid（padding/缺失点由 valid mask 排除）。
3. **超大 FMC 等比例下采样（预先声明）**：``S = max(ceil(H/256), ceil(W/1024))``，
   S>1 时最近邻索引采样。
4. **归一化**：FMC 每 view 独立 median/MAD robust z-score；PAUT 用
   ``fold_norm``（per-timestep，只由 train coupons 计算，strict 无泄漏）。
5. **mask 计划**：block=16×16，mask_ratio=0.3，由 ``(model_seed, step, 样本序)``
   确定性生成（复用 ``eddycus_pretrain.sample_block_masks``）。
6. **数据顺序**：FMC 由 ``data_seed`` 确定性选 bucket + batch 内 view；
   PAUT 由 ``data_seed`` 确定性采样 train 位置。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from wndt.data.adapters.external_weld_ut import (
    DATA_ROOT, ExternalWeldUTAdapter, WeldUTView, build_view_index, read_view,
)
from wndt.data.eddycus_pretrain import stable_hash, sample_block_masks
from wndt.data.ultrasound_pretrain import (
    COUPONS, PROCESSED_PAUT, load_paut, paut_fold_split,
)

MAX_FMC_H = 256            # FMC 等比例下采样阈值（预先声明，与 config 一致）
MAX_FMC_W = 1024
BLOCK = 16
DEFAULT_MASK_RATIO = 0.3


def downsample_scale(H: int, W: int) -> int:
    """等比例下采样倍数：S = max(ceil(H/256), ceil(W/1024))，S>=1。"""
    return max(1, math.ceil(H / MAX_FMC_H), math.ceil(W / MAX_FMC_W))


def downsample_grid(grid: np.ndarray, valid: np.ndarray, S: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """最近邻索引采样（等比例）；valid mask 同步映射。"""
    if S <= 1:
        return grid, valid
    H, W = grid.shape[-2], grid.shape[-1]
    H2, W2 = math.ceil(H / S), math.ceil(W / S)
    rows = np.floor(np.arange(H2) * H / H2).astype(np.int64)
    cols = np.floor(np.arange(W2) * W / W2).astype(np.int64)
    return grid[:, rows][:, :, cols], valid[rows][:, cols]


@dataclass
class ExtView:
    """外部 FMC view 元数据（下采样后尺寸）。"""
    source: str
    tx: int
    rx: int
    t: int
    ds_rx: int
    ds_t: int
    S: int
    group_id: str
    record_id: str
    mat_relpath: str


def build_ext_views(data_root: Path = DATA_ROOT,
                    sources: Sequence[str] = ("A", "B", "C")) -> list[ExtView]:
    """外部 FMC 全部 transmit views（含下采样后尺寸）。"""
    out: list[ExtView] = []
    for v in build_view_index(data_root, sources):
        S = downsample_scale(v.rx, v.t)
        out.append(ExtView(
            source=v.source, tx=v.tx, rx=v.rx, t=v.t,
            ds_rx=math.ceil(v.rx / S), ds_t=math.ceil(v.t / S), S=S,
            group_id=v.group_id, record_id=v.record_id,
            mat_relpath=v.mat_relpath))
    out.sort(key=lambda x: (x.source, x.tx))
    return out


def read_ext_view_ds(data_root: Path, v: ExtView) -> tuple[np.ndarray, np.ndarray]:
    """read_view + 等比例下采样 -> ``(1, ds_rx, ds_t) float32`` + valid。"""
    grid, valid = read_view(data_root, WeldUTView(
        source=v.source, tx=v.tx, rx=v.rx, t=v.t, group_id=v.group_id,
        record_id=v.record_id, mat_relpath=v.mat_relpath,
        defect_type=None, material=""))
    return downsample_grid(grid, valid, v.S)


def ext_view_summary(views: Sequence[ExtView]) -> dict[str, Any]:
    """外部 view 审计摘要（训练前输出）。"""
    from collections import Counter
    return {
        "n_views": len(views),
        "n_sources": len({v.source for v in views}),
        "n_groups": len({v.group_id for v in views}),
        "per_source": {s: sum(1 for v in views if v.source == s)
                       for s in sorted({v.source for v in views})},
        "size_buckets": dict(Counter(f"{v.ds_rx}x{v.ds_t}" for v in views)),
    }


def ext_bucket_plan(data_seed: int, views: Sequence[ExtView],
                    steps: int, batch_size: int
                    ) -> list[tuple[tuple[int, int], list[int]]]:
    """每 step 的采样计划：``[( (ds_rx, ds_t), [view 索引...] ), ...]``。

    每 step 确定性选一个 bucket，batch 内从该 bucket 有放回抽 batch_size 个
    view；只由 ``data_seed`` 决定（与 model_seed 无关）。
    """
    buckets: dict[tuple[int, int], list[int]] = {}
    for vi, v in enumerate(views):
        buckets.setdefault((v.ds_rx, v.ds_t), []).append(vi)
    bucket_names = sorted(buckets)
    plan = []
    for step in range(steps):
        key = bucket_names[stable_hash(data_seed, "m3bucket", step) % len(bucket_names)]
        views_in = buckets[key]
        idx = [views_in[stable_hash(data_seed, "m3view", step, j) % len(views_in)]
               for j in range(batch_size)]
        plan.append((key, idx))
    return plan


# ---------------------------------------------------------------------------
# PAUT 目标域（每折，strict：SSL 只读本折 train coupons）
# ---------------------------------------------------------------------------
def paut_fold_ssl_inputs(test_coupon: str, split_seed: int
                         ) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """本折 PAUT SSL 输入：train coupons 的归一化 B-scan + 索引。

    返回 ``(X_train, train_idx, train_coupons, val_coupon)``，其中
    ``X_train`` = ``(n_train, 49, 512) float32``，per-timestep 归一化
    只由 train coupons 计算（strict，无泄漏）。
    """
    ascans, coupons, _labels = load_paut()
    tr_idx, _va, _te, train_coupons, val_coupon = paut_fold_split(
        coupons, test_coupon, split_seed)
    X = paut_ssl_norm(ascans, tr_idx)
    return X, tr_idx, train_coupons, val_coupon


def paut_ssl_norm(ascans: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """(n,49,512) -> per-timestep 归一化，统计量只由 train_idx 计算。"""
    tr = np.asarray(ascans[train_idx], dtype=np.float32).reshape(-1, ascans.shape[-1])
    mean = tr.mean(0).astype(np.float32)
    std = (tr.std(0) + 1e-8).astype(np.float32)
    x = np.asarray(ascans[train_idx], dtype=np.float32)
    return (x - mean[None, None]) / std[None, None]


def paut_ssl_sample_plan(data_seed: int, n: int, steps: int, batch_size: int
                         ) -> np.ndarray:
    """PAUT SSL 采样计划：``(steps, batch_size)`` train 位置索引。

    只由 ``(data_seed, n, steps, batch_size)`` 决定（与 model_seed 无关），
    P-long/W→P 的 PAUT 阶段样本顺序完全一致。
    """
    key = stable_hash(data_seed, "m3paut_plan", int(n), int(steps), int(batch_size))
    rng = np.random.default_rng(key % (2 ** 32))
    return rng.integers(0, int(n), size=(int(steps), int(batch_size)))
