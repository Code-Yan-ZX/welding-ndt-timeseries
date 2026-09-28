"""缺陷感知掩码 (F4 vs F5 唯一差异): region-biased masked token 采样。

语义对齐机制 (F 系列):
- 缺陷区域元数据由各 dataset loader 注入 sample.metadata["defect_regions"]
  (归一化坐标; SWRD: bbox {x0,y0,x1,y1,cls}; EddyCus native_grid_2d: 缺陷几何;
  PENELOPE 默认 none — 保持与 E1 的 3000 步同分布, 见 configs F 系列)。
- 采样: k = round(n_valid·ratio) 个 masked token 中, k_r = round(k·region_frac)
  个从 (region ∩ valid) 采样, 其余从 (valid \ region) 采样。
- region token 不足 → 全部从 region∩valid 取, 缺额落回 random (不虚增比例)。
- regions 为空/None → 退化为纯 random (与 ssl.masking.sample_mask("random")
  同 seed 结果一致, 保证 F4/F5 随机部分同种子可比)。

纯 numpy, 无 torch 依赖, 便于单测与审计。
"""
from __future__ import annotations

import numpy as np


def regions_to_token_mask(
    grid: tuple[int, ...],
    valid: np.ndarray,
    regions: list[dict] | None,
    spatial_kind: str,
) -> np.ndarray:
    """归一化 region 坐标 → token 布尔图 (与 valid 同形状)。

    - 1d grid (C, n_col): region 用 {"t0","t1"} ⊂ [0,1] 时间区间;
      token 列 t 覆盖 [t/n_col, (t+1)/n_col), 与任一 region 相交即 True, 对 C 维复制。
    - 2d grid (C, n_h, n_w): region 用 {"x0","y0","x1","y1"} ⊂ [0,1]
      (x=列/W, y=行/H); token (r,p) 覆盖 [r/n_h,(r+1)/n_h) × [p/n_w,(p+1)/n_w),
      对 C 维复制。
    """
    regions = regions or []
    region_mask = np.zeros(valid.shape, dtype=bool)
    if not regions:
        return region_mask
    if spatial_kind == "1d":
        n_col = grid[-1]
        for reg in regions:
            t0, t1 = float(reg.get("t0", 0.0)), float(reg.get("t1", 0.0))
            p = max(0, int(np.floor(t0 * n_col)))
            q = min(n_col, int(np.ceil(t1 * n_col)))
            if q > p:
                region_mask[..., p:q] = True
    elif spatial_kind == "2d":
        n_h, n_w = grid[-2], grid[-1]
        for reg in regions:
            x0 = float(reg.get("x0", 0.0)); y0 = float(reg.get("y0", 0.0))
            x1 = float(reg.get("x1", 0.0)); y1 = float(reg.get("y1", 0.0))
            r0 = max(0, int(np.floor(y0 * n_h))); r1 = min(n_h, int(np.ceil(y1 * n_h)))
            c0 = max(0, int(np.floor(x0 * n_w))); c1 = min(n_w, int(np.ceil(x1 * n_w)))
            if r1 > r0 and c1 > c0:
                region_mask[..., r0:r1, c0:c1] = True
    else:
        raise ValueError(f"spatial_kind 必须是 1d/2d, 得到 {spatial_kind}")
    return region_mask


def sample_region_bias_mask(
    shape: tuple[int, ...],
    ratio: float,
    valid: np.ndarray,
    regions: list[dict] | None,
    region_frac: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """region-biased 掩码采样 (只在 valid token 内; 见模块 docstring)。

    - regions 空/None → 纯 random (与 sample_mask("random") 同 seed 一致)。
    """
    shape = tuple(int(s) for s in shape)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid {valid.shape} 与网格 {shape} 不一致")
    rng = np.random.default_rng(seed)
    n_valid = int(valid.sum())
    if n_valid == 0 or not regions:
        # 退化: 纯 random (复用 masking.sample_mask 保证逐位一致)
        from general_ndt.ssl.masking import sample_mask
        return sample_mask("random", shape, ratio, valid=valid, seed=seed)

    k = max(1, int(round(n_valid * ratio)))
    cands = np.argwhere(valid)
    if k >= len(cands):
        return valid.copy()

    spatial_kind = "1d" if len(shape) == 2 else "2d"
    region_mask = regions_to_token_mask(shape, valid, regions, spatial_kind)
    region_cands = np.argwhere(valid & region_mask)
    k_r = min(int(round(k * region_frac)), len(region_cands))

    masked = np.zeros(shape, dtype=bool)
    if k_r > 0:
        idx = rng.choice(len(region_cands), size=k_r, replace=False)
        masked[tuple(region_cands[idx].T)] = True
    # 其余从 region 外采样 (语义对齐只通过前 k_r 个体现); region 外不足时
    # 缺额回填 region 剩余 token (不虚增 region 占比, 总数守恒 = k)
    rest_cands = np.argwhere(valid & ~region_mask & ~masked)
    k_rest = min(k - k_r, len(rest_cands))
    if k_rest > 0:
        idx = rng.choice(len(rest_cands), size=k_rest, replace=False)
        masked[tuple(rest_cands[idx].T)] = True
    deficit = k - k_r - k_rest
    if deficit > 0:
        fill_cands = np.argwhere(valid & ~masked)
        if len(fill_cands) > 0:
            idx = rng.choice(len(fill_cands), size=min(deficit, len(fill_cands)),
                             replace=False)
            masked[tuple(fill_cands[idx].T)] = True
    return masked
