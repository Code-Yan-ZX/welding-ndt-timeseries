"""统一 batch / collate: 变长样本 pad + valid mask。

- batch 内要求 shape_kind 一致 (1d 或 2d);
- 1d: (C, T) → (B, C_max, L_max); 2d: (H, W) → (B, H_max, W_max);
- valid_mask 记录真实位置 (1) / padding (0);
- 长度差异较大的批量建议按 (channels, length) bucket 分组, 减少 padding 浪费
  (第一版提供 bucket_indices 工具函数, 训练时可选使用)。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from general_ndt.datasets.schema import GeneralNDTBatch, GeneralNDTSample


def collate_general_ndt(samples: Sequence[GeneralNDTSample]) -> GeneralNDTBatch:
    if not samples:
        raise ValueError("空 batch")
    kind = samples[0].shape_kind
    if any(s.shape_kind != kind for s in samples):
        raise ValueError(f"batch 内 shape_kind 必须一致, 得到 {set(s.shape_kind for s in samples)}")

    if kind == "1d":
        c_max = max(s.signal.shape[0] for s in samples)
        l_max = max(s.signal.shape[1] for s in samples)
        padded = np.zeros((len(samples), c_max, l_max), dtype=np.float32)
        valid = np.zeros((len(samples), l_max), dtype=np.int64)
        for b, s in enumerate(samples):
            c, l = s.signal.shape
            padded[b, :c, :l] = s.signal
            valid[b, :l] = 1
    else:  # 2d
        h_max = max(s.signal.shape[0] for s in samples)
        w_max = max(s.signal.shape[1] for s in samples)
        padded = np.zeros((len(samples), h_max, w_max), dtype=np.float32)
        valid = np.zeros((len(samples), h_max, w_max), dtype=np.int64)
        for b, s in enumerate(samples):
            h, w = s.signal.shape
            padded[b, :h, :w] = s.signal
            valid[b, :h, :w] = 1

    return GeneralNDTBatch(
        padded_signal=padded,
        valid_mask=valid,
        shape_kind=kind,
        sample_ids=[s.sample_id for s in samples],
        specimen_ids=[s.specimen_id for s in samples],
        labels=[s.label for s in samples],
        modalities=[s.modality for s in samples],
        metadata=[s.metadata for s in samples],
    )


def bucket_indices(
    samples: Sequence[GeneralNDTSample], max_batch: int = 64
) -> list[list[int]]:
    """按 (channels, length) 分桶, 返回桶内样本索引列表 (减少 padding 浪费)。"""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        key = (s.signal.shape[0], s.signal.shape[1])
        buckets[key].append(i)
    out = []
    for key in sorted(buckets):
        idxs = buckets[key]
        for k in range(0, len(idxs), max_batch):
            out.append(idxs[k : k + max_batch])
    return out
