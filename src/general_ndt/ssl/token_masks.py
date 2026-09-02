"""token 级 valid mask: 把 batch 的样本级 valid mask 转成 patch-token 级。

背景 (Phase 2A): collate 的 valid_mask 是采样点级 (1d: (B, L_max); 2d: (B, H_max, W_max)),
而 Transformer / 掩码控制器需要 patch-token 级 (B, N) mask。

规则:
- 1d (C_max, n_col): token (c, t) 有效 ⟺ c < C_i 且 时间 patch [t·pl, (t+1)·pl) 全部有效
  (被 padding 覆盖的 patch 无效; 通道 padding 无效)。
- 2d (C_max, n_h, n_w): token (c, r, p) 有效 ⟺ c < C_i 且 空间 patch [r·P:(r+1)·P, p·P:(p+1)·P)
  全部有效 (native_grid_2d 空洞所在 patch 无效)。
- 判定直接用 batch.valid_mask (已合并样本长度 padding 与 native 空洞) + batch.shapes
  (每样本真实通道数 C_i)。
"""
from __future__ import annotations

import numpy as np

from general_ndt.datasets.schema import GeneralNDTBatch


def _view(batch) -> "GeneralNDTBatch":
    """接受 GeneralNDTBatch 或等价 dict (padded_signal/valid_mask/shape_kind/shapes)。"""
    if isinstance(batch, GeneralNDTBatch):
        return batch
    if isinstance(batch, dict):
        return GeneralNDTBatch(
            padded_signal=np.asarray(batch["padded_signal"]),
            valid_mask=np.asarray(batch["valid_mask"]),
            shape_kind=batch["shape_kind"],
            sample_ids=batch.get("sample_ids", []),
            specimen_ids=batch.get("specimen_ids", []),
            labels=batch.get("labels", []),
            modalities=batch.get("modalities", []),
            metadata=batch.get("metadata", []),
            shapes=batch.get("shapes", []),
        )
    raise TypeError(f"batch 必须是 GeneralNDTBatch 或 dict, 得到 {type(batch)}")


def token_valid_mask(
    batch, patch_len: int, patch2d: int | None = None
) -> np.ndarray:
    """batch → (B, N) bool token-level valid mask (N = 网格 token 总数)。

    - 1d: N = C_max * (L_max // patch_len)
    - 2d: N = C_max * (H_max // P) * (W_max // P), P = patch2d
    """
    batch = _view(batch)
    B = batch.batch_size
    vm = np.asarray(batch.valid_mask)  # (B, L_max) 或 (B, H_max, W_max)
    if batch.shape_kind == "1d":
        pl = patch_len
        l_max = vm.shape[1]
        n_col = l_max // pl
        c_max = batch.padded_signal.shape[1]
        out = np.zeros((B, c_max, n_col), dtype=bool)
        for b, (c_i, l_i) in enumerate(batch.shapes):
            vm_b = vm[b]
            for t in range(min(n_col, l_i // pl)):
                if bool(vm_b[t * pl : (t + 1) * pl].all()):
                    out[b, :c_i, t] = True
        return out.reshape(B, -1)
    else:  # 2d
        P = patch2d or patch_len
        h_max, w_max = vm.shape[1], vm.shape[2]
        n_h, n_w = h_max // P, w_max // P
        c_max = batch.padded_signal.shape[1]
        out = np.zeros((B, c_max, n_h, n_w), dtype=bool)
        for b, (c_i, h_i, w_i) in enumerate(batch.shapes):
            vm_b = vm[b]
            for r in range(min(n_h, h_i // P)):
                for p in range(min(n_w, w_i // P)):
                    if bool(vm_b[r * P : (r + 1) * P, p * P : (p + 1) * P].all()):
                        out[b, :c_i, r, p] = True
        return out.reshape(B, -1)


def patchify_target(
    batch, patch_len: int, patch2d: int | None = None
) -> np.ndarray:
    """把 batch 的原始信号切成 patch 真值 → (B, N, patch_dim)。

    - 1d: token (c, t) → 原始向量 signal[c, t·pl:(t+1)·pl]  (patch_dim = patch_len)
    - 2d: token (c, r, p) → signal[c, r·P:(r+1)·P, p·P:(p+1)·P].flatten()
          (patch_dim = P*P)
    供 MAE 重建损失 (pred 与 target 逐 token 对齐)。padded 位置为 0 (valid mask 负责排除)。
    """
    batch = _view(batch)
    sig = np.asarray(batch.padded_signal, dtype=np.float32)
    B = batch.batch_size
    if batch.shape_kind == "1d":
        pl = patch_len
        l_max = sig.shape[2]
        n_col = l_max // pl
        c_max = sig.shape[1]
        out = np.zeros((B, c_max * n_col, pl), dtype=np.float32)
        for b in range(B):
            for c in range(c_max):
                for t in range(n_col):
                    out[b, c * n_col + t, :] = sig[b, c, t * pl : (t + 1) * pl]
        return out
    else:  # 2d
        P = patch2d or patch_len
        h_max, w_max = sig.shape[2], sig.shape[3]
        n_h, n_w = h_max // P, w_max // P
        c_max = sig.shape[1]
        pdim = P * P
        out = np.zeros((B, c_max * n_h * n_w, pdim), dtype=np.float32)
        for b in range(B):
            for c in range(c_max):
                for r in range(n_h):
                    for p in range(n_w):
                        out[b, (c * n_h + r) * n_w + p, :] = sig[
                            b, c, r * P : (r + 1) * P, p * P : (p + 1) * P
                        ].reshape(-1)
        return out
