"""M0-2A 数据集专属 stem / tokenizer（统一输出 patch/token embedding）。

三个外部超声数据集的原生 tensor 形状各异，**不强制插值成同一二维图片**；
每个数据集一个 stem，把各自原生形状编成 ``(B, L, D)`` 的 patch/token
embedding，供后续共享 encoder 使用（L 为每数据集自身的 token 数）。

- ``PenelopeStem``      : B-scan ``(B, 49, 512)`` → 2D 卷积 patch → ``(B, 7*16, D)``
- ``MLNDTFrameStem``    : B-scan 帧 ``(B, 256, 256)`` → 2D 卷积 patch → ``(B, 16*16, D)``
  （``MLNDTVolumeStem`` 对 ``(B, 100, 256, 256)`` 体积做 3D→2D 分帧 patch）
- ``NDTMLFlawStripStem``: 条带 ``(B, 480, 7168)`` → 先 AdaptivePool 到内部网格
  → 2D patch → ``(B, L, D)``（条带极宽，不做全局池化以保留扫描方向结构）

输出约定：``(B, L, D)`` 序列 token，可供 ``NDTEncoder`` / ITFormer 风格桥接。
smoke 阶段只验证形状，不承载正式训练。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvPatchStem(nn.Module):
    """2D 卷积 patch embedding：输入 (B, C, H, W) → (B, L, D)。

    ``kernel``/``stride`` 决定 token 数 L = H//stride[0] * W//stride[1]。
    """

    def __init__(self, in_channels: int, out_dim: int, kernel: tuple[int, int],
                 stride: tuple[int, int]):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=kernel, stride=stride)
        self.embed = nn.LayerNorm(out_dim)
        self._L = None

    @property
    def n_tokens(self) -> int:
        assert self._L is not None, "run forward once to infer token count"
        return self._L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.proj(x)                       # (B, D, h', w')
        self._L = h.shape[2] * h.shape[3]
        h = h.flatten(2).transpose(1, 2)       # (B, L, D)
        return self.embed(h)


class PenelopeStem(nn.Module):
    """PENELOPE B-scan (B, 49, 512) → patch(7, 32) → (B, 49, D)。"""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.stem = _ConvPatchStem(1, out_dim, kernel=(7, 32), stride=(7, 32))
        self.n_tokens = 112                     # 7 * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 49, 512) -> (B, 1, 49, 512)
        return self.stem(x.unsqueeze(1))


class MLNDTFrameStem(nn.Module):
    """ML-NDT 单帧 B-scan (B, 256, 256) → patch(16,16) → (B, 256, D)。"""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.stem = _ConvPatchStem(1, out_dim, kernel=(16, 16), stride=(16, 16))
        self.n_tokens = 256                     # 16 * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x.unsqueeze(1))


class MLNDTVolumeStem(nn.Module):
    """ML-NDT 体积 (B, 100, 256, 256) → 每帧独立 patch → (B, 100*16*16, D)。

    把体积视为帧序列，逐帧过 ``MLNDTFrameStem`` 再串接，保留帧时序 token。
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.frame_stem = MLNDTFrameStem(out_dim=out_dim)
        self.n_tokens = 100 * 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, H, W)
        b, f = x.shape[:2]
        xr = x.reshape(b * f, 1, x.shape[2], x.shape[3])
        toks = self.frame_stem.stem(xr)         # (B*F, L, D)
        return toks.reshape(b, f * self.frame_stem.n_tokens, self.out_dim)


class NDTMLFlawStripStem(nn.Module):
    """NDT_ML_Flaw 条带 (B, 480, 7168) → AdaptivePool → 2D patch → (B, L, D)。

    条带是 (深度, 扫描) 的窄高宽结构：先 AdaptiveAvgPool 到 (60, 1792)，
    再 2D patch (20, 64) → 3*28 = 84 tokens。池化仅在该数据集内部完成，
    不把三个数据集对齐成同一尺寸。
    """

    def __init__(self, out_dim: int = 128, grid: tuple[int, int] = (60, 1792),
                 patch: tuple[int, int] = (20, 64)):
        super().__init__()
        self.out_dim = out_dim
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.patch = patch
        self.stem = _ConvPatchStem(1, out_dim, kernel=patch, stride=patch)
        self.n_tokens = (grid[0] // patch[0]) * (grid[1] // patch[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 480, 7168) -> (B, 1, 480, 7168) -> pool -> (B, 1, H', W')
        return self.stem(self.pool(x.unsqueeze(1)))


class EddyCusStem(nn.Module):
    """EddyCus I/Q stem：原生 (N, 2) 1D I/Q -> 2D 栅格 (2, H, W) -> token。

    H/W 从 spatial_data（track_number / sample_number）重建，按真实网格保留；
    I/Q 双通道（real/imaginary），不强制 49×512。栅格尺寸随扫描变化
    （101×451 / 51×451 / 202×1067 ...），stem 用 AdaptiveAvgPool 出固定 token 数，
    对任意 H×W 稳健。
    """

    def __init__(self, out_dim: int = 128, grid: tuple[int, int] = (101, 451),
                 tokens: tuple[int, int] = (4, 8)):
        super().__init__()
        self.grid = grid
        self.tokens = tokens
        self.proj = nn.Conv2d(2, out_dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, grid: tuple[int, int] | None = None) -> torch.Tensor:
        # x: (B, N, 2) 或 (B, 2, H, W)
        if x.dim() == 3 and x.shape[-1] == 2:
            h, w = grid or self.grid
            n = x.shape[1]
            # 1D -> 栅格：填满 h*w，尾部不足补零（栅格近似规则，±1 点/track）
            if n < h * w:
                x = torch.nn.functional.pad(x, (0, 0, 0, h * w - n))
            x = x[:, : h * w].reshape(x.shape[0], h, w, 2).permute(0, 3, 1, 2).contiguous()
        h = self.proj(x)                                  # (B, D, H, W)
        h = torch.nn.functional.adaptive_avg_pool2d(h, self.tokens)  # (B, D, 4, 8)
        h = h.flatten(2).transpose(1, 2)                  # (B, 32, D)
        return self.norm(h)


DATASET_STEMS = {
    "penelope_paut": PenelopeStem,
    "ml_ndt": MLNDTFrameStem,      # 默认按帧；体积用 MLNDTVolumeStem
    "ml_ndt_volume": MLNDTVolumeStem,
    "ndt_ml_flaw": NDTMLFlawStripStem,
    "eddycus": EddyCusStem,
}


def build_dataset_stem(dataset_name: str, out_dim: int = 128) -> nn.Module:
    """按数据集名构建对应 stem（未知数据集报错而非静默退化为普通 stem）。"""
    if dataset_name not in DATASET_STEMS:
        raise KeyError(
            f"no dataset stem for {dataset_name!r}; available: {sorted(DATASET_STEMS)}")
    return DATASET_STEMS[dataset_name](out_dim=out_dim)
