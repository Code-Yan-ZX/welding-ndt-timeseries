"""PAUT B-scan 物理增强 (训练时用)。

作用于原始 (归一化前) 的逐位置 B-scan (n_beams, seq_len)。所有变换都是
尺度无关的 (乘性 / 结构性 / 相对), 与幅度单位无关, 因此对原始 int16 量化的
rectified envelope 与任意归一化都安全:

  beam_dropout   : 随机置零部分波束 (模拟阵元缺失 / 死元素)
  time_shift     : 沿时间(深度)轴循环平移 (模拟闸门 / 同步错位)
  amp_jitter     : 乘性增益抖动 (模拟通道增益漂移)
  gaussian_noise : 加性高斯噪声, 标准差为信号 std 的相对比例 (模拟电子噪声)

设计为可单独开关, 便于消融: 传入的 aug dict 的 key 即启用的增强, value 为
参数 (或 True/None 用默认量级)。
"""
from __future__ import annotations

import numpy as np

# 经验量级 (相对 / 无量纲)
DEFAULTS = {
    "beam_dropout": {"prob": 0.5, "max_drop_frac": 0.15},
    "time_shift": {"max_shift": 8},
    "amp_jitter": {"max_ratio": 0.15},
    "gaussian_noise": {"rel_std": 0.03},
}

ALL_AUGS = list(DEFAULTS.keys())


def augment_bscan(x: np.ndarray, aug: dict) -> np.ndarray:
    """x: (C, T) float32 原始信号 (归一化前)。aug: {name: params|True|None}。

    用全局 np.random (与 PAUTSeriesDataset 一致; set_seed 已设种子)。
    """
    x = np.array(x, dtype=np.float32, copy=True)
    C, T = x.shape
    for name, params in aug.items():
        if params is True or params is None:
            params = DEFAULTS.get(name, {})
        if name == "beam_dropout":
            if np.random.random() < params.get("prob", 0.5):
                max_drop = int(params.get("max_drop_frac", 0.15) * C)
                if max_drop >= 1:
                    n_drop = int(np.random.randint(0, max_drop + 1))
                    if n_drop > 0:
                        idx = np.random.choice(C, n_drop, replace=False)
                        x[idx] = 0.0
        elif name == "time_shift":
            ms = int(params.get("max_shift", 8))
            s = int(np.random.randint(-ms, ms + 1))
            if s != 0:
                x = np.roll(x, s, axis=-1)
        elif name == "amp_jitter":
            mr = float(params.get("max_ratio", 0.15))
            x = x * (1.0 + float(np.random.uniform(-mr, mr)))
        elif name == "gaussian_noise":
            rs = float(params.get("rel_std", 0.03))
            sd = float(x.std()) + 1e-8
            x = x + np.random.normal(0.0, rs * sd, size=x.shape).astype(np.float32)
    return x
