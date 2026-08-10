#!/usr/bin/env python
"""渲染 PAUT B-scan 为图像供多模态 LLM (P2) 使用。

每位置渲染两种图:
  - 灰度 B-scan: (49, 512) 原始包络, 百分位归一化 -> 灰度图 (512x512)
  - VLT 式频谱伪彩色: 沿时间轴 rfft 的 log 幅度 -> turbo colormap -> RGB (512x512)
百分位 (2-98) 归一化避免重尾回波过曝。输出 data/processed/paut/images/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO = Path(__file__).resolve().parents[1]
processed = REPO / "data/processed/paut"
OUT = processed / "images"
SIZE = 512


def norm_pct(x: np.ndarray, p_lo=2, p_hi=98) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def render_grayscale(bscan: np.ndarray) -> Image.Image:
    """bscan (49,512) -> 灰度 PIL (512x512)。"""
    n = norm_pct(bscan)
    img = Image.fromarray((n * 255).astype(np.uint8), mode="L").convert("RGB")
    return img.resize((SIZE, SIZE), Image.BILINEAR)


def render_spectral(bscan: np.ndarray) -> Image.Image:
    """bscan (49,512) -> 沿时间 rfft log 幅度 -> turbo 伪彩色 (512x512)。"""
    spec = np.fft.rfft(bscan, axis=-1, norm="ortho")
    mag = np.log1p(np.abs(spec)).astype(np.float32)      # (49, 257)
    n = norm_pct(mag)
    rgb = (cm.turbo(n)[:, :, :3] * 255).astype(np.uint8)  # (49, 257, 3)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((SIZE, SIZE), Image.BILINEAR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ascans = np.load(processed / "ascans.npy", mmap_mode="r")
    coupons = np.load(processed / "meta_coupon.npy")
    labels = np.load(processed / "meta_label.npy")
    N = len(coupons)
    print(f"渲染 {N} 位置 × 2 图 (灰度+频谱) -> {OUT}")
    manifest = []
    for i in range(N):
        b = np.array(ascans[i], dtype=np.float32)
        render_grayscale(b).save(OUT / f"{i:05d}_bscan.png")
        render_spectral(b).save(OUT / f"{i:05d}_spec.png")
        manifest.append({"idx": i, "coupon": str(coupons[i]), "label": int(labels[i])})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N}")
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh)
    print(f"完成: {N*2} 张图 -> {OUT}")


if __name__ == "__main__":
    main()
