"""GDXray Welds 数据集 (WRT-SAM 复现)。

目录约定 (见 data/raw/gdxray/README.md):
  W0001/W0001_xxxx.png  原图 (灰度 png)
  W0002/W0002_xxxx.png  理想分割掩码 (1-bit 二值)
  W0003/RRT01|02/*.tif  无标注数字化底片 (GDXray-58 零样本测试)

预处理: 论文 §4.3 "保持高度、沿宽度裁剪到 640px" — 实现两种模式:
  mode="strip": 保持高度, 居中裁宽 640, 再整体缩放到 inp_size (论文直译)
  mode="full":  整图等比缩放进 inp_size (SAM 标准做法, 作为对照)
归一化同 baseline: (x-0.5)/0.5。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[2]
GDXRAY = REPO / "data" / "raw" / "gdxray"

# W0001 来源映射 (W0001 readme): 用于构造 GDXray-58 剔除清单
W0001_SOURCES = [
    ("RRT01", "bam5.tif"), ("RRT01", "RRT-12R.tif"), ("RRT01", "RRT-22R.tif"),
    ("RRT01", "RRT-28R.tif"), ("RRT01", "RRT-39R.tif"), ("RRT01", "RRT-40R.tif"),
    ("RRT01", "RRT-13R.tif"), ("RRT01", "RRT-31R.tif"),
    ("RRT02", "RRT-106R.tif"), ("RRT02", "RRT-107R.tif"),
]


def gdxray58_files() -> list[Path]:
    """W0003 全部 68 张中剔除 W0001 来源 10 张后的 58 张."""
    excluded = {GDXRAY / "W0003" / sub / name for sub, name in W0001_SOURCES}
    files = sorted((GDXRAY / "W0003").glob("*/*.tif"))
    keep = [f for f in files if f not in excluded]
    assert len(keep) == 58, f"expected 58, got {len(keep)}"
    return keep


def _to3(x: torch.Tensor) -> torch.Tensor:
    """灰度图复制为 3 通道 (SAM 输入约定, 同 baseline 数据管线)."""
    return x.repeat(3, 1, 1)


def _load_pair(idx: int, mode: str, inp_size: int, crop: str = "center", rng=None):
    """crop: center (固定, 供验证/复现) | random (训练增强, 每次随机宽度条带)."""
    img = Image.open(GDXRAY / f"W0001/W0001_{idx:04d}.png").convert("L")
    gt = Image.open(GDXRAY / f"W0002/W0002_{idx:04d}.png").convert("L")
    if mode == "strip":
        w0, h0 = img.size
        target_w = 640
        if w0 > target_w:
            if crop == "random":
                left = int(rng.integers(0, w0 - target_w + 1))
            else:
                left = (w0 - target_w) // 2
            box = (left, 0, left + target_w, h0)
            img = img.crop(box)
            gt = gt.crop(box)
    img = img.resize((inp_size, inp_size), Image.BILINEAR)
    gt = gt.resize((inp_size, inp_size), Image.NEAREST)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
    y = torch.from_numpy((np.asarray(gt, np.float32) / 255.0 > 0.5).astype(np.float32))[None]
    x = _to3(x)
    x = (x - 0.5) / 0.5
    return x, y


class GDXray10(Dataset):
    """GDXray-10: idx 1..10, 按 8:2 划分 (val=最后 2 张, 同 baseline CAMO 式划分).

    strips_per_image > 1 且 mode=strip 时, 训练集做随机条带增强:
    数据集长度 = 8 * strips_per_image, 每个样本随机取一条 640px 宽度条带.
    """

    def __init__(self, split: str = "train", inp_size: int = 1024, mode: str = "strip",
                 strips_per_image: int = 1, seed: int = 0):
        idxs = list(range(1, 11))
        if split == "train":
            self.idxs = idxs[:8]
        elif split == "val":
            self.idxs = idxs[8:]
        else:  # all
            self.idxs = idxs
        self.split, self.inp_size, self.mode = split, inp_size, mode
        self.strips_per_image = strips_per_image if (split == "train" and mode == "strip") else 1
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idxs) * self.strips_per_image

    def __getitem__(self, i):
        idx = self.idxs[i % len(self.idxs)]
        crop = "random" if self.split == "train" else "center"
        x, y = _load_pair(idx, self.mode, self.inp_size, crop=crop, rng=self.rng)
        return {"inp": x, "gt": y}


class GDXray58(Dataset):
    """GDXray-58: 无官方掩码, 仅用于零样本推理可视化 (见报告对 Table 6 的讨论)."""

    def __init__(self, inp_size: int = 1024, mode: str = "strip"):
        self.files = gdxray58_files()
        self.inp_size, self.mode = inp_size, mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = Image.open(f).convert("L")
        w0, h0 = img.size
        if self.mode == "strip" and w0 > 640:
            left = (w0 - 640) // 2
            img = img.crop((left, 0, left + 640, h0))
        img = img.resize((self.inp_size, self.inp_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None]
        x = _to3(x)
        x = (x - 0.5) / 0.5
        return {"inp": x, "path": str(f)}
