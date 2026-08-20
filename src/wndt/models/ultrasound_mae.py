"""M0-2B 统一超声 MAE（共享输入 patch embedding + 共享 Transformer encoder）。

三个外部超声数据集（PENELOPE PAUT / ML-NDT / NDT_ML_Flaw）都被编码为
**单通道 2D 超声张量**，用一个共享 encoder 做掩码重建（MAE）预训练：

- 共享 ``Conv2d`` patch embedding（patch size 默认 16×16）；
- 2D sin-cos 位置编码，支持不同 batch 不同空间尺寸（token 数可变）；
- Transformer encoder（默认 ``d_model=128`` / ``depth=4`` / ``n_heads=4`` /
  ``mlp_ratio=4``）；
- 共享**线性** patch reconstruction head（逐 token 重建被掩码 patch 的像素）；
- mask ratio 默认 0.5；masked-patch SmoothL1 损失；
- encoder 输出 mean pooling，供下游冻结-encoder 二分类头使用。

约定：**不要求同一 batch 内混合不同形状**——不同数据集可在不同 batch 中，
各自空间尺寸不同（如 PENELOPE (512,64)、ML-NDT 帧 (256,256)、
NDT_ML_Flaw 窗口 (480,256)）。

- ``UltrasoundMAE``        : 完整 MAE（patch embed + PE + encoder + recon head）
- ``build_2d_sincos_pe``   : 形状可变的 2D sin-cos 位置编码（供测试/复用）
- ``UltrasoundMAE.encode`` : (B,1,H,W) -> (B,L,D)  token
- ``UltrasoundMAE.encode_pooled`` : (B,1,H,W) -> (B,D)  mean pooling（下游输入）
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pe(h: int, w: int, d_model: int, device: torch.device | str) -> torch.Tensor:
    """构建形状可变的 2D sin-cos 位置编码，返回 ``(h*w, d_model)``。

    编码维度分两半：前半编码**行**位置，后半编码**列**位置。网格尺寸
    (h, w) 由当前 batch 的 patch 网格决定，因此同一模型可处理可变 token 数。
    """
    assert d_model % 2 == 0, "d_model must be even for 2D sin-cos PE"
    d_half = d_model // 2
    pe = torch.zeros(h, w, d_model, device=device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_half, 2, device=device).float() / d_half))
    pos_h = torch.arange(h, device=device).float()          # (h,)
    pos_w = torch.arange(w, device=device).float()          # (w,)
    pe_h = pos_h[:, None] * inv_freq[None, :]               # (h, d_half/2)
    pe_w = pos_w[:, None] * inv_freq[None, :]               # (w, d_half/2)
    # 行编码 -> 前半维度
    pe[:, :, 0:d_half:2] = torch.sin(pe_h)[:, None, :]
    pe[:, :, 1:d_half:2] = torch.cos(pe_h)[:, None, :]
    # 列编码 -> 后半维度
    pe[:, :, d_half::2] = torch.sin(pe_w)[None, :, :]
    pe[:, :, d_half + 1::2] = torch.cos(pe_w)[None, :, :]
    return pe.reshape(-1, d_model)


class PatchEmbed(nn.Module):
    """共享 2D 卷积 patch embedding：``(B, C, H, W) -> (B, L, D)``。

    ``L = (H/ph)*(W/pw)`` 随 batch 形状变化（支持可变 token 数）。
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: tuple[int, int]):
        super().__init__()
        ph, pw = patch_size
        self.patch_size = (ph, pw)
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=(ph, pw), stride=(ph, pw))
        self.norm = nn.LayerNorm(d_model)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W) -> (B, L, C*ph*pw)``（像素目标，供重建）。"""
        b, c, h, w = x.shape
        ph, pw = self.patch_size
        assert h % ph == 0 and w % pw == 0, f"input {h}x{w} not divisible by patch {ph}x{pw}"
        x = x.reshape(b, c, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // ph) * (w // pw), ph * pw * c)
        return x

    def unpatchify(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(B, L, C*ph*pw) -> (B, C, H, W)``（重建像素还原，诊断用）。"""
        b = tokens.shape[0]
        ph, pw = self.patch_size
        l = (h // ph) * (w // pw)
        x = tokens[:, :l].reshape(b, h // ph, w // pw, ph, pw, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, -1, h, w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x).flatten(2).transpose(1, 2)      # (B, L, D)
        return self.norm(z)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer 块：注意力 + MLP，残差连接。"""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class UltrasoundMAE(nn.Module):
    """统一超声掩码自编码器（共享 encoder，M0-2B 预训练骨干）。

    输入为单通道 2D 超声张量 ``(B, 1, H, W)``；``H``/``W`` 需能被 patch size
    整除。mask ratio 下随机掩码 patch token，编码可见 token，线性头重建被
    掩码 patch 的像素，SmoothL1 损失只在被掩码 patch 上计算。
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: tuple[int, int] = (16, 16),
        mask_ratio: float = 0.5,
        in_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = tuple(patch_size)
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(in_channels, d_model, self.patch_size)
        self.encoder = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        # 共享线性 patch reconstruction head（逐 token 预测被掩码 patch 的像素）
        self.recon_head = nn.Linear(d_model, in_channels * self.patch_size[0] * self.patch_size[1])

    # ------------------------------------------------------------------
    # 结构信息
    # ------------------------------------------------------------------
    @property
    def grid(self) -> tuple[int, int]:
        """当前最后 forward 的 patch 网格（供 smoke/审计核对）。"""
        return self._grid

    @property
    def n_patch_pixels(self) -> int:
        return self.in_channels * self.patch_size[0] * self.patch_size[1]

    def arch_signature(self) -> dict:
        """E1/E2/E3 结构一致性审计用（相同配置应产生相同 signature）。"""
        return {"d_model": self.d_model, "depth": self.depth,
                "n_heads": self.n_heads, "mlp_ratio": self.mlp_ratio,
                "patch_size": list(self.patch_size), "in_channels": self.in_channels}

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, H, W) -> (B, L, D)``（patch embed + PE + Transformer）。"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        b, _, h, w = x.shape
        ph, pw = self.patch_size
        hg, wg = h // ph, w // pw
        self._grid = (hg, wg)
        x_embed = self.patch_embed(x)                       # (B, L, D)
        pe = build_2d_sincos_pe(hg, wg, self.d_model, x.device)
        x_embed = x_embed + pe[None, :, :]
        return self.encoder(x_embed)

    def encode_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, H, W) -> (B, D)`` mean pooling（下游冻结-encoder 二分类头输入）。"""
        return self.encode(x).mean(dim=1)

    @staticmethod
    def random_masking(seq: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """对 ``(B, L, D)`` 的 token 序列做每样本独立随机掩码。

        返回 ``(B, L)`` bool mask，``True`` = 被掩码。每样本恰好掩码
        ``max(1, round(L * mask_ratio))`` 个 token。
        """
        b, l, _ = seq.shape
        n_mask = max(1, int(round(l * mask_ratio)))
        noise = torch.rand(b, l, device=seq.device)
        ids = noise.argsort(dim=1)                          # 升序 -> 前 n_mask 最小
        mask = torch.zeros(b, l, dtype=torch.bool, device=seq.device)
        mask.scatter_(1, ids[:, :n_mask], True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """MAE 前向。``x: (B, 1, H, W)``。

        返回 dict：``loss``（masked-patch SmoothL1）、``recon``、``target``、
        ``mask``、``z``（编码 token）、``pe``。
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        b, _, h, w = x.shape
        ph, pw = self.patch_size
        hg, wg = h // ph, w // pw
        self._grid = (hg, wg)

        target = self.patch_embed.patchify(x)               # (B, L, P*P)
        x_embed = self.patch_embed(x)                       # (B, L, D)
        pe = build_2d_sincos_pe(hg, wg, self.d_model, x.device)
        x_embed = x_embed + pe[None, :, :]

        mask = self.random_masking(x_embed, mask_ratio)     # (B, L) bool
        x_masked = x_embed.clone()
        x_masked[mask] = 0.0                                # 置零被掩码 token

        z = self.encoder(x_masked)                          # (B, L, D)
        recon = self.recon_head(z)                          # (B, L, P*P)

        loss = F.smooth_l1_loss(recon[mask], target[mask], reduction="mean")
        return {"loss": loss, "recon": recon, "target": target,
                "mask": mask, "z": z, "pe": pe}
