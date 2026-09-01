"""自监督目标: masked reconstruction / raw↔时频一致性 / 跨传感器不变性。

方法规格 §3.5:
    L = λ₁·L_recon + λ₂·L_tf [+ λ₃·L_inv]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MAE 式重建损失: 只算 masked ∩ valid 的 L2。

    pred/target: (B, N, d_patch) 逐 token 预测/真值 (pad 位置 target 建议为 0)
    mask:        (B, N) bool — 被掩码的 token
    valid:       (B, N) bool — 非 padding 的 token
    """
    m = mask.bool()
    if valid is not None:
        m = m & valid.bool()
    diff = (pred - target) ** 2
    denom = m.sum().clamp(min=1.0)
    return (diff * m.unsqueeze(-1)).sum() / denom


def tf_consistency_loss(
    z_raw: torch.Tensor, z_tf: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """原始视图 ↔ 时频视图 InfoNCE (同一样本两视图为正对)。

    z_raw/z_tf: (B, d) 已 L2 归一化的池化表征。
    """
    z_raw = F.normalize(z_raw, dim=-1)
    z_tf = F.normalize(z_tf, dim=-1)
    logits = z_raw @ z_tf.T / temperature          # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits, labels)
    # 对称项
    loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
    return loss


def cross_sensor_invariance_loss(
    pairs: list[tuple[torch.Tensor, torch.Tensor]], eps: float = 1e-6
) -> torch.Tensor:
    """跨传感器不变性 (条件启用): 同一物理位置 l 的传感器 a,b 的表征应一致。

    pairs: [(z_a, z_b), ...]  z: (d,) 池化表征。
    """
    if not pairs:
        return torch.tensor(0.0, requires_grad=False)
    diffs = torch.stack([(a - b) ** 2 for a, b in pairs]).sum(-1)
    return diffs.mean()
