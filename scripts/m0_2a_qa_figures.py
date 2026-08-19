#!/usr/bin/env python
"""M0-2A 数据 QA 检查图：对 ML-NDT 与 NDT_ML_Flaw 随机读取 8–16 个样本，
输出 shape / 频谱 / 标签检查图。仅作数据 QA，不做训练。

输出到 experiments/results/m0_2a/qa/：
- ml_ndt_qa.png       : 3×2 子图（B-scan + 频谱 + 标签）对 2 个 volume
- ndt_ml_flaw_qa.png  : 3×3 子图对 3 个条带（真实批）

运行（需原始数据已下载）：
    python scripts/m0_2a_qa_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wndt.data.adapters.ml_ndt import MLNDTAdapter          # noqa: E402
from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter  # noqa: E402
from wndt.data.adapters.unified import read_random           # noqa: E402

OUT = REPO / "experiments" / "results" / "m0_2a" / "qa"


def _spectrum(x: np.ndarray) -> np.ndarray:
    # 对深度轴（axis=0）做 FFT，返回幅度谱（dB）
    spec = np.abs(np.fft.rfft(x, axis=0))
    return spec


def qa_ml_ndt(seed: int = 0) -> None:
    ad = MLNDTAdapter()
    samples = read_random(ad, 2, seed=seed)   # 2 个 volume
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, inst in enumerate(samples):
        vol = inst.tensors["volume"]          # (100, 256, 256)
        # 取一个中央帧作为 B-scan 展示
        frame = vol[vol.shape[0] // 2]
        ax = axes[i, 0]
        im = ax.imshow(frame, aspect="auto", cmap="gray",
                       vmin=0, vmax=np.percentile(frame, 99.5))
        ax.set_title(f"{inst.record_id} | label={inst.metadata.get('label_status')}\nframe {vol.shape[0]//2} of {vol.shape[0]}")
        ax.set_xlabel("x (voxel)"); ax.set_ylabel("y (voxel)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        # 频谱：帧内沿 y 轴平均后对 x 轴 FFT
        spec = _spectrum(frame.mean(axis=0))
        ax = axes[i, 1]
        ax.plot(10 * np.log10(spec + 1e-6))
        ax.set_title("spectrum (avg over y, rFFT)")
        ax.set_xlabel("freq bin"); ax.set_ylabel("dB")
        # 标签检查：flaw 帧范围 vs 全帧标签
        ax = axes[i, 2]
        labels = inst.metadata.get("frame_labels", [])
        if labels:
            ax.plot(labels, "o-", ms=3)
            ax.set_title("frame labels [flaw 0/1]")
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.text(0.5, 0.5, f"label_status={inst.label_status}\n"
                               f"n_frames={vol.shape[0]}", ha="center")
            ax.set_title("label check")
            ax.axis("off")
    fig.suptitle("ML-NDT QA: shape / spectrum / label", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "ml_ndt_qa.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"saved {p}")


def qa_ndt_ml_flaw(seed: int = 0) -> None:
    ad = NDTMLFlawAdapter()
    # 只取真实批（.xz）避免仿真实例掩盖真实信号；每批抽 1 条
    real_ids = [r.acquisition_id for r in ad.records()
                if r.data_origin == "measured"]
    seen = []
    for bid in dict.fromkeys(real_ids):
        rows = [i for i, r in enumerate(ad.records())
                if r.acquisition_id == bid]
        if len(seen) >= 3:
            break
        seen.append(rows[len(rows) // 2])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, idx in enumerate(seen[:3]):
        r = ad.records()[idx]
        strip = ad.read_strip(idx)            # (480, 7168)
        ax = axes[i, 0]
        # 条带极宽：展示降采样后的全貌 + 中央 512 列细节
        ds = strip[:, ::8]
        im = ax.imshow(ds, aspect="auto", cmap="gray",
                       vmin=0, vmax=np.percentile(strip, 99.5))
        ax.set_title(f"{r.record_id}\nlabel={r.label_status} "
                     f"defect={r.defect_instance_id}")
        ax.set_xlabel("scan (voxel, /8)"); ax.set_ylabel("depth (voxel)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        # 频谱：对深度轴 FFT（取扫描子段）
        patch = strip[:, 3000:3300]
        spec = _spectrum(patch)
        ax = axes[i, 1]
        ax.plot(10 * np.log10(spec.mean(axis=1) + 1e-6))
        ax.set_title("spectrum (rFFT over depth, patch)")
        ax.set_xlabel("freq bin"); ax.set_ylabel("dB")
        # 标签检查：batch 内缺陷率 + 该条带元数据
        ax = axes[i, 2]
        row = r.extra.get("row", {})
        ax.text(0.05, 0.85, f"label_status = {r.label_status}\n"
                             f"data_origin = {r.data_origin}\n"
                             f"augmentation = {row.get('augmentation')}\n"
                             f"depth = {row.get('depth')}\n"
                             f"position = {row.get('position')}\n"
                             f"size_mm = {row.get('size_mm')}\n"
                             f"defect_type = {row.get('defect_type')}",
                fontsize=9, transform=ax.transAxes)
        ax.set_title("label check")
        ax.axis("off")
    fig.suptitle("NDT_ML_Flaw QA: shape / spectrum / label (real batches)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "ndt_ml_flaw_qa.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"saved {p}")


def main() -> None:
    qa_ml_ndt()
    qa_ndt_ml_flaw()
    print("QA figures done.")


if __name__ == "__main__":
    main()
