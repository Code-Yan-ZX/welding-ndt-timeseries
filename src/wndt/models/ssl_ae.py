"""PAUT 自监督掩码自编码器 (Masked Autoencoder) + 下游分类/异常检测 (P1)。

在全部无标注 B-scan (49, 512) 上预训练: 随机掩码部分波束 (模拟缺失孔径), 编码可见
部分, 解码重建被掩码波束。编码器学到焊缝超声专属表征, 下游:
  - 分类: 冻结 SSL 编码器 + 可训练分类头, 做 LOOCV (与 from-scratch encoder/SSF 对照)
  - 异常检测: 重建误差作异常分, McKnight 式 Weibull 拟合 clean 类 (无标注 baseline)

  MAEEncoder : (B,1,49,512) -> z (d_model)            可复用于下游
  MAEDecoder  : z -> (B,1,49,512) 重建
  MaskedAE    : 编码器+解码器, 掩码波束重建 (MAE 风格)
  SSLClassifier : MAEEncoder (冻结可选) + 分类头
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEEncoder(nn.Module):
    """Conv 编码器: (B,1,49,512) -> z (d_model)。"""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 7), padding=(1, 3)), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 7), padding=(1, 3)), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 7), padding=(1, 3)), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d((2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.proj(self.pool(self.conv(x)))


class MAEDecoder(nn.Module):
    """z (d_model) -> 重建 (B,1,49,512)。"""

    def __init__(self, d_model: int = 128, mid_h: int = 7, mid_w: int = 64):
        super().__init__()
        self.mid_h, self.mid_w = mid_h, mid_w
        self.fc = nn.Linear(d_model, 128 * mid_h * mid_w)
        self.refine = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.size(0), 128, self.mid_h, self.mid_w)
        h = F.interpolate(h, size=(49, 512), mode="bilinear", align_corners=False)
        return self.refine(h)


class MaskedAE(nn.Module):
    """掩码波束重建自编码器。mask_ratio 比例的波束被置零, 重建被掩码波束。"""

    def __init__(self, d_model: int = 128, mask_ratio: float = 0.3, dropout: float = 0.2,
                 noise_std: float = 0.02):
        super().__init__()
        self.encoder = MAEEncoder(d_model, dropout)
        self.decoder = MAEDecoder(d_model)
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def mask_beams(self, x: torch.Tensor):
        """x: (B,1,49,512)。返回 (masked_x, mask), mask=0 处为被掩码波束。"""
        B, C, H, W = x.shape
        n_mask = max(1, int(H * self.mask_ratio))
        # 每个样本独立随机掩码 n_mask 个波束
        mask = torch.ones(B, 1, H, 1, device=x.device)
        rand = torch.rand(B, H, device=x.device)
        idx = rand.topk(n_mask, dim=1).indices  # (B, n_mask)
        for b in range(B):
            mask[b, 0, idx[b], 0] = 0.0
        x_masked = x * mask
        if self.noise_std > 0 and self.training:
            x_masked = x_masked + torch.randn_like(x_masked) * self.noise_std * x_masked.std()
        return x_masked, mask

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x_masked, mask = self.mask_beams(x)
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, x, mask

    def recon_loss(self, recon, target, mask):
        """Huber (smooth_l1) 重建损失 -- 对重尾回波值稳健。掩码波束 + 全图。"""
        inv = 1.0 - mask  # (B,1,H,1) 被掩码处
        # 掩码波束上的 Huber
        diff = (recon - target) * inv
        masked = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="sum") / inv.sum().clamp(min=1)
        full = F.smooth_l1_loss(recon, target)
        return masked + 0.5 * full


class SSLClassifier(nn.Module):
    """SSL 编码器 + 分类头。freeze_encoder=True 时只训头。"""

    def __init__(self, encoder: MAEEncoder, d_model: int = 128, n_classes: int = 2,
                 freeze_encoder: bool = True, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)
