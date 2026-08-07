"""DANN (Domain-Adversarial Neural Network) for PAUT cross-coupon transfer.

在 SSF (谱-空-频) 编码器之上加一个试件域判别器, 经梯度反转层 (GRL) 连接:
编码器同时学「缺陷判别」与「抗域判别」, 从而抽出跨试件域不变的特征, 期望提升
LOOCV 泛化。推理时只取标签头, 域头丢弃。

  encode(x) -> z (B, 3*d_model)        复用 SSF 三分支融合特征
  label_logits(x) -> (B, n_classes)     缺陷标签头 (与 SSF 相同)
  forward(x, lam) -> (label_logits, domain_logits)   训练用, GRL 反转域梯度
"""
from __future__ import annotations

import torch
import torch.nn as nn

from wndt.models.ssf import SSFClassifier


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_)


def grl_lambda_schedule(step: int, total: int, max_lambda: float = 1.0) -> float:
    """GRL 权重随训练从 0 平滑升到 max_lambda (DANN 原文 2/(1+e^{-10p})-1)。"""
    p = step / max(1, total)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0))


class DANNSSFClassifier(SSFClassifier):
    """SSF + 域对抗。子类化 SSFClassifier 复用三分支与标签头, 加 GRL + 域头。"""

    def __init__(self, *, n_beams: int = 49, seq_len: int = 512, d_model: int = 128,
                 dropout: float = 0.3, n_classes: int = 2, n_domains: int = 5,
                 max_lambda: float = 1.0, in_channels: int = 1):
        super().__init__(n_beams=n_beams, seq_len=seq_len, d_model=d_model,
                         dropout=dropout, n_classes=n_classes, in_channels=in_channels)
        self.n_domains = n_domains
        self.max_lambda = max_lambda
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Dropout(dropout),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_domains),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回三分支融合特征 z (B, 3*d_model)。"""
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)                                  # (B, 1, Bm, L)
        z_sp = self.spatial(x)
        sp_t = self._mag_fft(x[:, 0], dim=-1).unsqueeze(1)
        z_spec = self.spectral(sp_t)
        sp_b = self._mag_fft(x[:, 0], dim=-2).unsqueeze(1)
        z_freq = self.frequency(sp_b)
        return torch.cat([z_sp, z_spec, z_freq], dim=1)

    def label_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def forward(self, x: torch.Tensor, lambda_: float = 1.0):
        z = self.encode(x)
        label_logits = self.head(z)
        domain_logits = self.domain_head(self.grl(z, lambda_))
        return label_logits, domain_logits
