"""通用 NDT 样本结构 (GeneralNDTSample) 与统一 batch (GeneralNDTBatch)。

对齐方法规格 (docs/general_ndt_foundation/phase1_method_spec.md) 的输入结构:
signal / mask / modality / sample_id / specimen_id / sensor_id / sampling_rate /
spatial_coordinates / label / metadata。

样本表示规则:
- signal 为 numpy 数组, shape 两种:
    * 1D 形态: (C, T)  channel×time   (AE 波形 / 导波 A 扫 / 多波束 PAUT / 多频 ECT 曲线)
    * 2D 形态: (H, W)  time×space 栅格 (B-scan / C-scan), 或时频谱图
  shape_kind 显式声明 ("1d" / "2d"), 由 adapter 按形态分派 stem。
- 本层为纯 numpy/dataclass, 不依赖 torch; collate 才进入 batch。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# 模态枚举 (与 manifest schema / configs/general_ndt_datasets.yaml 对齐)
MODALITIES = ("ultrasonic", "guided_wave", "eddy_current", "acoustic_emission", "vibration")

# 标签类型
LABEL_TYPES = ("binary", "multiclass", "regression", "none")


@dataclass
class GeneralNDTSample:
    """一个通用 NDT 样本 (最小物理可划分单元的观测)。"""

    sample_id: str
    signal: np.ndarray                  # (C, T) 1D 或 (H, W) 2D
    shape_kind: str = "1d"              # "1d" | "2d"
    modality: str = "ultrasonic"        # MODALITIES 之一
    specimen_id: Optional[str] = None   # 最小物理独立单元 (coupon/配置组/试件)
    sensor_id: Optional[str] = None     # 传感器/波束/通道标识
    sampling_rate: Optional[float] = None  # Hz (或名义空间分辨率)
    spatial_coordinates: Optional[np.ndarray] = None  # (N_coord,) 或 None
    label: Optional[Any] = None         # 0/1、类别码、回归值、或 None
    label_type: str = "none"            # LABEL_TYPES 之一
    defect_type: Optional[str] = None   # 自由字符串缺陷类型
    split_group: Optional[str] = None   # 预声明的严格划分组 (来自 manifest)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape_kind not in ("1d", "2d"):
            raise ValueError(f"shape_kind 必须是 1d/2d, 得到 {self.shape_kind}")
        if self.modality not in MODALITIES:
            raise ValueError(f"未知 modality: {self.modality} (可选 {MODALITIES})")
        if self.label_type not in LABEL_TYPES:
            raise ValueError(f"未知 label_type: {self.label_type} (可选 {LABEL_TYPES})")
        self.signal = np.asarray(self.signal)
        if self.signal.ndim != 2:
            raise ValueError(f"signal 必须是 2D 数组 (1D 形态 (C,T) 或 2D 形态 (H,W)), 得到 {self.signal.shape}")

    @property
    def channels(self) -> int:
        return self.signal.shape[0]

    @property
    def length(self) -> int:
        return self.signal.shape[1]


@dataclass
class GeneralNDTBatch:
    """统一 batch: 对变长样本做 pad + valid mask。

    - padded_signal: (B, C_max, L_max)  1D 形态; 或 (B, H_max, W_max) 2D 形态
    - valid_mask: (B, L_max) 或 (B, H_max, W_max), 1=真实, 0=padding
    - batch_shape_kind: 由 collate 保证 batch 内形态一致
    - sample_ids / specimen_ids / labels: 按样本对齐的元数据
    """

    padded_signal: np.ndarray
    valid_mask: np.ndarray
    shape_kind: str
    sample_ids: list
    specimen_ids: list
    labels: list
    modalities: list
    metadata: list = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.padded_signal.shape[0]
