"""general_ndt.ssl — 物理感知掩码 + 自监督目标 (重建 / 时频一致性 / 跨传感器不变性)。"""
from general_ndt.ssl.masking import MaskController
from general_ndt.ssl.objectives import (
    cross_sensor_invariance_loss,
    masked_recon_loss,
    tf_consistency_loss,
)

__all__ = [
    "MaskController",
    "masked_recon_loss",
    "tf_consistency_loss",
    "cross_sensor_invariance_loss",
]
