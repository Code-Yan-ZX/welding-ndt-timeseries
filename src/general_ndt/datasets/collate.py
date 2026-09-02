"""统一 batch / collate: 变长样本 pad + valid mask。

- batch 内要求 shape_kind 一致 (1d 或 2d);
- 1d: (C, T) → (B, C_max, L_max);  valid_mask (B, L_max) 记录时间有效 (1=真实, 0=padding);
- 2d: (H, W)/(C, H, W) → (B, C_max, H_max, W_max) (单通道归一化为 C=1);
      valid_mask (B, H_max, W_max) 记录空间有效区 (1=有效, 0=padding/空洞)。
- batch.shapes 记录每样本真实形状 → token 级 valid mask 用它区分 channel padding 与
  time/spatial padding (见 ssl/token_masks.py)。
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
        shapes = []
        for b, s in enumerate(samples):
            c, l = s.signal.shape
            padded[b, :c, :l] = s.signal
            if s.valid_mask is None:
                valid[b, :l] = 1
            else:
                vm = np.asarray(s.valid_mask, dtype=np.int64)
                # 允许 (T,) 或 (C, T); 投影到时间维: 任一通道有效则时间有效
                if vm.ndim == 1:
                    valid[b, : min(l, vm.shape[0])] = vm[: min(l, vm.shape[0])]
                else:
                    valid[b, : min(l, vm.shape[1])] = (
                        vm[:, : min(l, vm.shape[1])].any(axis=0)
                    )
            shapes.append((c, l))
    else:  # 2d — 归一化为 (C, H, W)
        def _to3(s: GeneralNDTSample) -> np.ndarray:
            sig = np.asarray(s.signal)
            return sig[None, ...] if sig.ndim == 2 else sig

        c_max = max(_to3(s).shape[0] for s in samples)
        h_max = max(_to3(s).shape[1] for s in samples)
        w_max = max(_to3(s).shape[2] for s in samples)
        padded = np.zeros((len(samples), c_max, h_max, w_max), dtype=np.float32)
        valid = np.zeros((len(samples), h_max, w_max), dtype=np.int64)
        shapes = []
        for b, s in enumerate(samples):
            sig = _to3(s)
            c, h, w = sig.shape
            padded[b, :c, :h, :w] = sig
            if s.valid_mask is None:
                valid[b, :h, :w] = 1
            else:
                vm = np.asarray(s.valid_mask)
                if vm.ndim == 2:
                    valid[b, : min(h, vm.shape[0]), : min(w, vm.shape[1])] = (
                        vm[: min(h, vm.shape[0]), : min(w, vm.shape[1])]
                    )
                else:
                    valid[b, : min(h, vm.shape[1]), : min(w, vm.shape[2])] = (
                        vm[:, : min(h, vm.shape[1]), : min(w, vm.shape[2])].any(axis=0)
                    )
            shapes.append((c, h, w))

    return GeneralNDTBatch(
        padded_signal=padded,
        valid_mask=valid,
        shape_kind=kind,
        sample_ids=[s.sample_id for s in samples],
        specimen_ids=[s.specimen_id for s in samples],
        labels=[s.label for s in samples],
        modalities=[s.modality for s in samples],
        metadata=[s.metadata for s in samples],
        shapes=shapes,
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
