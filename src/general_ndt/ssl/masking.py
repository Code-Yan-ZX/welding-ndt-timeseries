"""物理感知掩码控制器 (方法规格 §3.3)。

在 token 网格 (n_row, n_col) 上工作:
- 1d 形态:  (C, n_col)    行=通道(传感器/波束), 列=时间 token
- 2d 形态:  (n_h, n_w)    行/列 = patch 网格
- 时频视图: (n_f, n_tf)   行=频率 token, 列=时间 token

掩码模式:
- random          随机 token
- time_segment    连续时间段 (整列块)
- sensor_channel  整行掩码 (1D: 传感器/波束通道)
- freq_band       连续行块 (时频视图: 频率带)
- spatial_region  连续矩形区域 (2D: 空间扫描区域)

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

    def __call__(self, shape: tuple[int, int], seed: int | None = None) -> np.ndarray:
        modes = list(self.mode_probs)
        probs = list(self.mode_probs.values())
        mode = np.random.default_rng(seed).choice(modes, p=probs)
        fn = {
            "random": mask_random,
            "time_segment": mask_time_segment,
            "sensor_channel": mask_sensor_channel,
            "freq_band": mask_freq_band,
            "spatial_region": mask_spatial_region,
        }[mode]
        return fn(shape, self.ratio, seed=seed)
