"""物理感知掩码控制器 (方法规格 §3.3; Phase 2A 修复)。

在 token 网格上工作:
- 1d 形态:  (C, n_col)         行=通道(传感器/波束), 列=时间 token
- 2d 形态:  (C, n_h, n_w)      通道 × 空间 patch 网格 (native_grid_2d / 时频)
- 时频视图: (1, n_f, n_tf)     行=频率 token, 列=时间 token

掩码模式:
- random          随机 token
- time_segment    连续时间段 (整列块; 3d 作用于最后维 n_w)
- sensor_channel  整行掩码 (1d: 传感器/波束通道; 3d: 通道维 C)
- freq_band       连续行块 (时频视图/多频通道维)
- spatial_region  连续矩形区域 (2d: 空间扫描区域; 3d: 对全部通道复制同一空间区域)

Phase 2A 修复: **mask 只作用于 valid token**。
- sample_mask 可接收 valid (bool 网格, 1=有效), 只从有效 token 中采样;
  结构化模式先生成再与 valid 求交, 不足额用随机 valid token 补齐。
- MaskController.__call__(shape, valid=None, seed=None)。

纯 numpy 实现, 无 torch 依赖, 便于单测与审计。
"""
from __future__ import annotations

import numpy as np

MASK_MODES = ("random", "time_segment", "sensor_channel", "freq_band", "spatial_region")


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def mask_random(shape: tuple[int, int], ratio: float, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    n = shape[0] * shape[1]
    k = int(round(n * ratio))
    flat = np.zeros(n, dtype=bool)
    idx = rng.choice(n, size=min(k, n), replace=False)
    flat[idx] = True
    return flat.reshape(shape)


def mask_time_segment(
    shape: tuple[int, int], ratio: float, seed: int | None = None
) -> np.ndarray:
    """连续时间段: 掩掉整个时间段 (所有行对应列)。"""
    rng = _rng(seed)
    n_col = shape[1]
    block = max(1, int(round(n_col * ratio)))
    start = rng.integers(0, n_col)
    cols = [(start + i) % n_col for i in range(block)]
    m = np.zeros(shape, dtype=bool)
    m[:, cols] = True
    return m


def mask_sensor_channel(
    shape: tuple[int, int], ratio: float, seed: int | None = None
) -> np.ndarray:
    """整行掩码: 1D 形态下掩掉整个传感器/波束通道。"""
    rng = _rng(seed)
    n_row = shape[0]
    k = max(1, int(round(n_row * ratio)))
    rows = rng.choice(n_row, size=min(k, n_row), replace=False)
    m = np.zeros(shape, dtype=bool)
    m[rows, :] = True
    return m


def mask_freq_band(
    shape: tuple[int, int], ratio: float, seed: int | None = None
) -> np.ndarray:
    """连续行块: 时频视图下掩掉连续频率带。"""
    rng = _rng(seed)
    n_row = shape[0]
    band = max(1, int(round(n_row * ratio)))
    start = rng.integers(0, n_row)
    rows = [(start + i) % n_row for i in range(band)]
    m = np.zeros(shape, dtype=bool)
    m[rows, :] = True
    return m


def mask_spatial_region(
    shape: tuple[int, int], ratio: float, seed: int | None = None
) -> np.ndarray:
    """连续矩形区域: 2D 栅格上掩掉局部空间扫描区域。"""
    rng = _rng(seed)
    n_row, n_col = shape
    dh = max(1, int(round(n_row * np.sqrt(ratio))))
    dw = max(1, int(round(n_col * np.sqrt(ratio))))
    r0 = rng.integers(0, n_row)
    c0 = rng.integers(0, n_col)
    m = np.zeros(shape, dtype=bool)
    for dr in range(dh):
        for dc in range(dw):
            m[(r0 + dr) % n_row, (c0 + dc) % n_col] = True
    return m


def _structured_mask(
    mode: str, shape: tuple[int, ...], ratio: float, rng: np.random.Generator
) -> np.ndarray:
    """结构化掩码 (2d 或 3d 网格), 不感知 valid (由 sample_mask 求交)。"""
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    m = np.zeros(shape, dtype=bool)
    if nd == 2:
        n_row, n_col = shape
        if mode == "time_segment":
            block = max(1, int(round(n_col * ratio)))
            start = rng.integers(0, n_col)
            m[:, [(start + i) % n_col for i in range(block)]] = True
        elif mode in ("sensor_channel", "freq_band"):
            n = n_row
            if mode == "sensor_channel":
                k = max(1, int(round(n * ratio)))
                rows = rng.choice(n, size=min(k, n), replace=False)
            else:
                band = max(1, int(round(n * ratio)))
                start = rng.integers(0, n)
                rows = [(start + i) % n for i in range(band)]
            m[rows, :] = True
        elif mode == "spatial_region":
            dh = max(1, int(round(n_row * np.sqrt(ratio))))
            dw = max(1, int(round(n_col * np.sqrt(ratio))))
            r0 = rng.integers(0, n_row)
            c0 = rng.integers(0, n_col)
            for dr in range(dh):
                for dc in range(dw):
                    m[(r0 + dr) % n_row, (c0 + dc) % n_col] = True
    elif nd == 3:
        C, n_h, n_w = shape
        if mode == "time_segment":
            block = max(1, int(round(n_w * ratio)))
            start = rng.integers(0, n_w)
            m[:, :, [(start + i) % n_w for i in range(block)]] = True
        elif mode in ("sensor_channel", "freq_band"):
            n = C
            if mode == "sensor_channel":
                k = max(1, int(round(n * ratio)))
                ch = rng.choice(n, size=min(k, n), replace=False)
            else:
                band = max(1, int(round(n * ratio)))
                start = rng.integers(0, n)
                ch = [(start + i) % n for i in range(band)]
            m[ch, :, :] = True
        elif mode == "spatial_region":
            dh = max(1, int(round(n_h * np.sqrt(ratio))))
            dw = max(1, int(round(n_w * np.sqrt(ratio))))
            r0 = rng.integers(0, n_h)
            c0 = rng.integers(0, n_w)
            for dr in range(dh):
                for dc in range(dw):
                    m[:, (r0 + dr) % n_h, (c0 + dc) % n_w] = True
    else:
        raise ValueError(f"mask 网格仅支持 2/3 维, 得到 {shape}")
    return m


def sample_mask(
    mode: str,
    shape: tuple[int, ...],
    ratio: float,
    valid: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """从 valid token 中采样掩码 (2d/3d 网格)。

    - valid=None → 全部 token 有效 (等价旧行为);
    - random: 只在有效 token 中精确采样, 比例 = 有效数×ratio;
    - 结构化模式: 结构 & valid 求交 (比例近似)。不做逐 token 补齐 —— 补齐会破坏
      spatial_region 的跨通道复制语义 (native_grid_2d 空洞为通道共享, 求交后复制保持)。
    """
    shape = tuple(int(s) for s in shape)
    if mode not in MASK_MODES:
        raise ValueError(f"未知掩码模式: {mode} (可选 {MASK_MODES})")
    rng = _rng(seed)
    valid = (
        np.ones(shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid.shape != shape:
        raise ValueError(f"valid {valid.shape} 与网格 {shape} 不一致")
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros(shape, dtype=bool)
    if mode == "random":
        target = max(1, int(round(n_valid * ratio)))
        cands = np.argwhere(valid)
        if target >= len(cands):
            return valid.copy()
        idx = rng.choice(len(cands), size=target, replace=False)
        m = np.zeros(shape, dtype=bool)
        m[tuple(cands[idx].T)] = True
        return m
    return _structured_mask(mode, shape, ratio, rng) & valid


class MaskController:
    """按样本采样物理掩码。mode 可为单个字符串或按概率分布的 dict。"""

    def __init__(self, mode: str | dict[str, float] = "random", ratio: float = 0.3):
        if isinstance(mode, str):
            if mode not in MASK_MODES:
                raise ValueError(f"未知掩码模式: {mode} (可选 {MASK_MODES})")
            self.mode_probs = {mode: 1.0}
        else:
            total = sum(mode.values())
            if total <= 0:
                raise ValueError("掩码概率和必须 > 0")
            self.mode_probs = {k: v / total for k, v in mode.items()}
        self.ratio = ratio

    def __call__(
        self,
        shape: tuple[int, ...],
        valid: np.ndarray | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        modes = list(self.mode_probs)
        probs = list(self.mode_probs.values())
        rng = np.random.default_rng(seed)
        mode = rng.choice(modes, p=probs)
        # 确定性: mask 子种子 = seed 派生 (模式选择与 mask 采样解耦)
        mask_seed = None if seed is None else seed * 1000 + modes.index(mode) + 1
        return sample_mask(mode, shape, self.ratio, valid=valid, seed=mask_seed)
