"""M0-2C ECT SSL 数据与评估（EddyCus I/Q 双通道视图，P1 MAEEncoder 续训）。

核心约定（E 与 P→E **完全一致**，只差 encoder 初始化）：

1. **view**：每条 (scan, frequency) = 1 个 view；695 个有信号扫描 × 4 频率 =
   2780 views。split/group **永远按 scan 的物理配置**（specimen_id 配置代理），
   frequency 不是独立物理样本。
2. **原生栅格**：按 spatial track/sample 散点重建 ``(2, H, W)``；栅格缺失点
   置 0 作 sentinel，**是否参与 loss 由 valid mask 决定**（recon loss 只统计
   masked∩valid，padding/缺失点绝不进）。同 batch 按最终尺寸 bucket（同尺寸
   才同批，无需批内插值；"padding" = 缺失点由 valid mask 处理）。
3. **超大网格等比例下采样（预先声明，E/P→E 一致）**：
   ``S = max(ceil(H/256), ceil(W/768))``；S>1 时按最近邻索引采样到
   ``(ceil(H/S), ceil(W/S))``（如 202×1067→101×534，501×560→251×280）。
4. **归一化**：每个 (scan, frequency) 每个 I/Q 通道在 valid 像素上
   median/MAD robust z-score（``(x-med)/(1.4826*MAD+1e-6)``），E/P→E 相同，
   不调参、不按 train/val 划分（无泄漏概念，逐 view 独立）。
5. **mask 计划**：block=16×16，mask_ratio=0.3；掩码由
   ``(model_seed, step, 样本序)`` 确定性生成（``sample_block_masks``），
   E/P→E 完全一致。
6. **数据顺序**：view 按 ``(record_id, freq_key)`` 排序；每 step 由
   ``data_seed`` 确定性选 bucket + batch 内 view（``ect_bucket_plan``），
   E/P→E 完全一致。

``downsample_scale`` / ``read_view_ds`` / ``sample_block_masks`` /
``ect_bucket_plan`` 均被训练、probe 与测试复用，保证三处路径完全一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from wndt.data.adapters.eddycus import EddyCusAdapter

FREQ_KEYS = ("f1", "f2", "f3", "f4")
MAX_GRID_H = 256          # 等比例下采样阈值（预先声明）
MAX_GRID_W = 768          # 等比例下采样阈值（预先声明）
BLOCK = 16                # 2D block mask 尺寸
DEFAULT_MASK_RATIO = 0.3


def stable_hash(*parts: Any) -> int:
    import hashlib
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return int(h.hexdigest(), 16)


# ---------------------------------------------------------------------------
# 下采样规则（固定、预先声明；E 与 P→E 完全一致）
# ---------------------------------------------------------------------------
def downsample_scale(H: int, W: int) -> int:
    """等比例下采样倍数：S = max(ceil(H/256), ceil(W/768))，S>=1。"""
    return max(1, math.ceil(H / MAX_GRID_H), math.ceil(W / MAX_GRID_W))


def downsample_grid(grid: np.ndarray, valid: np.ndarray, S: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """最近邻索引采样（等比例）；valid mask 同步映射；E/P→E 一致。

    ``grid (2,H,W)`` -> ``(2,H2,W2)``；``valid (H,W)`` -> ``(H2,W2)``。
    被采样的源点若缺失（valid=False）则目标点仍缺失（valid=False），
    后续 loss 排除。
    """
    if S <= 1:
        return grid, valid
    H, W = grid.shape[-2], grid.shape[-1]
    H2, W2 = math.ceil(H / S), math.ceil(W / S)
    rows = np.floor(np.arange(H2) * H / H2).astype(np.int64)
    cols = np.floor(np.arange(W2) * W / W2).astype(np.int64)
    return grid[:, rows][:, :, cols], valid[rows][:, cols]


# ---------------------------------------------------------------------------
# view 索引（2780 视图的元数据；split/group 键 = 扫描物理配置）
# ---------------------------------------------------------------------------
@dataclass
class ECTView:
    rec_index: int            # 在 adapter.records() 中的索引（有信号记录）
    freq_key: str             # f1..f4
    record_id: str
    specimen_id: str          # 物理配置代理（split/group 键）
    defect_instance_id: str | None
    flaw: bool                # True = 缺陷
    defect_type: str          # 8 类
    material: str
    sensor: str
    H: int                    # 原生 track 数
    W: int                    # 原生每 track 采样
    ds_H: int                 # 下采样后
    ds_W: int
    S: int                    # 下采样倍数


def build_view_index(adapter: EddyCusAdapter,
                     freq_keys: Sequence[str] = FREQ_KEYS) -> list[ECTView]:
    """695 有信号扫描 × 4 频率 = 2780 views（按 record_id, freq 排序）。"""
    views: list[ECTView] = []
    for i in adapter.signal_indices():
        rec = adapter.records()[i]
        trk, smp = adapter.read_spatial(i)
        H, W = int(trk.max()), int(smp.max())
        S = downsample_scale(H, W)
        d = rec.domain
        for fk in freq_keys:
            views.append(ECTView(
                rec_index=i, freq_key=fk, record_id=rec.record_id,
                specimen_id=rec.specimen_id,
                defect_instance_id=rec.defect_instance_id,
                flaw=rec.defect_present, defect_type=rec.defect_type,
                material=str(d.get("material_type", "")),
                sensor=str(rec.geometry.get("sensor_type", "")),
                H=H, W=W, ds_H=math.ceil(H / S), ds_W=math.ceil(W / S), S=S,
            ))
    views.sort(key=lambda v: (v.record_id, v.freq_key))
    return views


# ---------------------------------------------------------------------------
# 读取 + 归一化 + 下采样（probe 与训练共用，保证完全一致）
# ---------------------------------------------------------------------------
def robust_normalize_1d(vals: np.ndarray) -> np.ndarray:
    """median/MAD robust z-score（valid 像素上）。"""
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad + 1e-6
    return (vals - med) / scale


def read_view(adapter: EddyCusAdapter, rec_index: int, freq_key: str
              ) -> tuple[np.ndarray, np.ndarray]:
    """读取并归一化一个 view -> ``(2,H,W) float32`` + ``(H,W) bool valid``。

    散点重建到原生栅格；缺失点置 0 sentinel（valid=False）；每个 I/Q 通道
    在 valid 像素上 median/MAD robust 归一化。loss 只统计 masked∩valid。
    """
    t = adapter.read_frequency(rec_index, freq_key)      # iq (N,2)
    iq = np.asarray(t["iq"], dtype=np.float32)
    trk, smp = adapter.read_spatial(rec_index)
    H, W = int(trk.max()), int(smp.max())
    grid = np.full((2, H, W), 0.0, dtype=np.float32)
    r = trk - 1
    c = smp - 1
    for ch in range(2):
        grid[ch, r, c] = robust_normalize_1d(iq[:, ch])
    valid = np.zeros((H, W), dtype=bool)
    valid[r, c] = True
    # 缺失点（padding / 栅格缺失）置 0 作 sentinel；**是否参与 loss 由
    # valid mask 决定**（recon loss 只统计 masked∩valid，绝不进 padding/缺失点）。
    return grid, valid


def read_view_ds(adapter: EddyCusAdapter, rec_index: int, freq_key: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """read_view + 预先声明的等比例下采样（E/P→E/probe 完全一致）。"""
    grid, valid = read_view(adapter, rec_index, freq_key)
    S = downsample_scale(grid.shape[-2], grid.shape[-1])
    return downsample_grid(grid, valid, S)


# ---------------------------------------------------------------------------
# 2D block mask（确定性；E/P→E 完全一致）
# ---------------------------------------------------------------------------
def block_mask(H: int, W: int, mask_ratio: float, generator: torch.Generator,
               block: int = BLOCK) -> torch.Tensor:
    """16×16 block mask：``(1,1,H,W)`` float，0=masked，1=可见。

    掩码块数 = max(1, round(H*W/(block²) * mask_ratio))；块级随机置零后
    最近邻放大到像素级（padding 区块由 valid mask 在 loss 中排除）。
    """
    bh, bw = math.ceil(H / block), math.ceil(W / block)
    n_blocks = bh * bw
    n_mask = max(1, int(n_blocks * mask_ratio))
    perm = torch.randperm(n_blocks, generator=generator)[:n_mask]
    m = torch.ones(n_blocks)
    m[perm] = 0.0
    m = m.reshape(1, 1, bh, bw)
    return F.interpolate(m.float(), size=(H, W), mode="nearest")


def sample_block_masks(H: int, W: int, n: int, model_seed: int, step: int,
                       mask_ratio: float = DEFAULT_MASK_RATIO,
                       block: int = BLOCK) -> torch.Tensor:
    """一个 batch 的 block mask：``(B,1,H,W)``。

    每个样本的掩码只由 ``(model_seed, step, 样本序)`` 决定（stable_hash ->
    torch.Generator），E/P→E 的掩码计划完全一致，且不依赖全局 RNG 顺序。
    """
    masks = []
    for j in range(n):
        seed = stable_hash(model_seed, "mask", step, j) & 0x7FFFFFFFFFFFFFFF
        g = torch.Generator().manual_seed(seed)
        masks.append(block_mask(H, W, mask_ratio, g, block))   # (1,1,H,W)
    return torch.stack(masks, dim=0).squeeze(1)                # (B,1,H,W)


# ---------------------------------------------------------------------------
# 确定性数据顺序（只由 data_seed 决定；E/P→E 完全一致）
# ---------------------------------------------------------------------------
def ect_bucket_plan(data_seed: int, view_index: Sequence[ECTView],
                    steps: int, batch_size: int
                    ) -> list[tuple[tuple[int, int], list[int]]]:
    """每 step 的采样计划：``[( (H,W), [view 索引...] ), ...]``。

    - 每 step 确定性选一个 bucket（最终尺寸 (ds_H, ds_W)）；
    - batch 内从该 bucket 确定性（有放回）抽 ``batch_size`` 个 view；
    - 只由 ``data_seed`` 决定，与 model_seed 无关 -> E/P→E 数据顺序一致。
    """
    buckets: dict[tuple[int, int], list[int]] = {}
    for vi, v in enumerate(view_index):
        buckets.setdefault((v.ds_H, v.ds_W), []).append(vi)
    bucket_names = sorted(buckets)                    # 确定性顺序
    plan = []
    for step in range(steps):
        key = bucket_names[stable_hash(data_seed, "bucket", step) % len(bucket_names)]
        views = buckets[key]
        idx = [views[stable_hash(data_seed, "view", step, j) % len(views)]
               for j in range(batch_size)]
        plan.append((key, idx))
    return plan


def ect_view_summary(view_index: Sequence[ECTView]) -> dict[str, Any]:
    """全数据集审计摘要（训练前输出：view/扫描/组/类/material/sensor 分布）。"""
    from collections import Counter
    n_scan = len({v.rec_index for v in view_index})
    groups = Counter(v.specimen_id for v in view_index)
    types = Counter(v.defect_type for v in view_index)
    mats = Counter(v.material for v in view_index)
    sens = Counter(v.sensor for v in view_index)
    grids = Counter((v.ds_H, v.ds_W) for v in view_index)
    return {
        "n_views": len(view_index),
        "n_scans": n_scan,
        "n_config_groups": len(groups),
        "n_flaw_views": sum(1 for v in view_index if v.flaw),
        "n_clean_views": sum(1 for v in view_index if not v.flaw),
        "defect_type_views": dict(types),
        "material_views": dict(mats),
        "sensor_views": dict(sens),
        "grid_views_after_ds": {f"{h}x{w}": n for (h, w), n in grids.items()},
    }
