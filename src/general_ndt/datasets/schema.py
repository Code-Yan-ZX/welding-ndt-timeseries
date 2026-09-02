"""通用 NDT 样本结构 (GeneralNDTSample) 与统一 batch (GeneralNDTBatch)。

对齐方法规格 (docs/general_ndt_foundation/phase1_method_spec.md) 的输入结构:
signal / mask / modality / sample_id / specimen_id / sensor_id / sampling_rate /
spatial_coordinates / label / metadata。

样本表示规则:
- signal 为 numpy 数组, shape 两种:
    * 1D 形态: (C, T)  channel×time   (AE 波形 / 导波 A 扫 / 多波束 PAUT / 多频 ECT 曲线)
    * 2D 形态: (H, W)  单通道 time×space 栅格 (B-scan / C-scan / 时频谱图)
              或 (C, H, W) 多通道 2D 栅格 (如 EddyCus native_grid_2d: 4 频×I/Q 通道 × H×W)
  shape_kind 显式声明 ("1d" / "2d"), 由 adapter 按形态分派 stem。
- valid_mask: 可选样本级有效区 mask (与 signal 空间维一致):
    * 1D 形态: (T,) 或 (C, T) —— None 表示全部有效
    * 2D 形态: (H, W) 或 (C, H, W) —— 供 native_grid_2d 的空洞/scatter 缺失位 (0=无效)
  collate 会把样本级 valid_mask 合并进 batch.valid_mask。
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
    signal: np.ndarray                  # (C, T) 1D 或 (H, W)/(C, H, W) 2D
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
    valid_mask: Optional[np.ndarray] = None  # 样本级有效区 mask (见模块 docstring)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shape_kind not in ("1d", "2d"):
            raise ValueError(f"shape_kind 必须是 1d/2d, 得到 {self.shape_kind}")
        if self.modality not in MODALITIES:
            raise ValueError(f"未知 modality: {self.modality} (可选 {MODALITIES})")
        if self.label_type not in LABEL_TYPES:
            raise ValueError(f"未知 label_type: {self.label_type} (可选 {LABEL_TYPES})")
        self.signal = np.asarray(self.signal)
        if self.shape_kind == "1d" and self.signal.ndim != 2:
            raise ValueError(f"1D 形态 signal 必须是 2D (C,T), 得到 {self.signal.shape}")
        if self.shape_kind == "2d" and self.signal.ndim not in (2, 3):
            raise ValueError(f"2D 形态 signal 必须是 (H,W) 或 (C,H,W), 得到 {self.signal.shape}")
        if self.valid_mask is not None:
            self.valid_mask = np.asarray(self.valid_mask)
            # 允许与 signal 同形状, 或空间维形状 (通道共享):
            #   1d: (T,) 或 (C, T); 2d: (H, W) 或 (C, H, W)
            sig = self.signal
            ok = self.valid_mask.shape == sig.shape or self.valid_mask.shape == sig.shape[-2:]
            if not ok:
                raise ValueError(
                    f"valid_mask 形状 {self.valid_mask.shape} 应等于 signal {sig.shape} "
                    f"或其空间维 {sig.shape[-2:]}")

    @property
    def channels(self) -> int:
        # 1D: (C, T); 2D: (H, W) -> 1 通道, (C, H, W) -> C 通道
        if self.shape_kind == "1d":
            return self.signal.shape[0]
        return 1 if self.signal.ndim == 2 else self.signal.shape[0]

    @property
    def length(self) -> int:
        if self.shape_kind == "1d":
            return self.signal.shape[1]
        return self.signal.shape[-1]


@dataclass
class GeneralNDTBatch:
    """统一 batch: 对变长样本做 pad + valid mask。

    - padded_signal: (B, C_max, L_max)  1D 形态; 或 (B, C_max, H_max, W_max) 2D 形态
      (2D 单通道样本归一化为 C=1)
    - valid_mask: (B, L_max) 或 (B, H_max, W_max), 1=真实/有效区, 0=padding/空洞
    - shapes: 每样本真实形状 [(C,L), ...] 或 [(C,H,W), ...] —— 用于 token 级 valid mask
      (区分 channel padding 与 time/spatial padding)
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
    shapes: list = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.padded_signal.shape[0]
