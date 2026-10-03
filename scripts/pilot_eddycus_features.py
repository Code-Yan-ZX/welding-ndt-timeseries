#!/usr/bin/env python3
"""Pilot-B3 前置: EddyCus 特征缓存（ndt_public_benchmark_pilot）。

对 695 个有信号扫描各生成两类表示并落盘（不覆盖原始数据）：

1. grid_2d: native_grid_2d scatter 重建 (8, H, W)（4 频率 × I/Q），空洞以逐通道
   中位数填充，**等比缩放进 (96, 384) 盒内 + 中位数 padding**（保持纵横比，
   501×560 大扫描不强行变形）→ float32 (695, 8, 96, 384)。
2. classical: 扫描级手工特征（每频率 I/Q/magnitude 的 mean/std/min/max/ptp/median
   + log-magnitude mean/std + 2D FFT 幅值 mean/std/max），供 LR/RF/SVM。

输出：
- data/processed/eddycus_pilot/grid_96x384.npy   (695, 8, 96, 384) float32
- data/processed/eddycus_pilot/features_classical.csv
- data/processed/eddycus_pilot/scan_order.json   扫描顺序 = scan_table 顺序
- experiments/results/ndt_pilot/eddycus_feature_build.json（runtime + 映射记录）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.eddycus import _load_native_grid, BlockingError  # noqa: E402

CACHE = REPO / "data" / "processed" / "eddycus_pilot"
OUT_RESULTS = REPO / "experiments" / "results" / "ndt_pilot"
BOX_H, BOX_W = 96, 384


def fit_box_resize(grid: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    """(8,H,W) → 等比缩放进 (BOX_H, BOX_W) 盒内 + 中位数 padding（不变形）。"""
    _, h, w = grid.shape
    scale = min(BOX_H / h, BOX_W / w, 1.0)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    x = F.interpolate(grid.unsqueeze(0), size=(nh, nw), mode="area").squeeze(0)
    out = fill[:, None, None].repeat(1, BOX_H, BOX_W).clone()
    out[:, (BOX_H - nh) // 2:(BOX_H - nh) // 2 + nh,
        (BOX_W - nw) // 2:(BOX_W - nw) // 2 + nw] = x
    return out


def classical_features(grids: np.ndarray) -> dict[str, float]:
    """grids: (8, H, W) 4 频率 × (I,Q)（padding 后；幅值/相位由 I/Q 导出）。"""
    feats: dict[str, float] = {}
    for f in range(4):
        i_ch, q_ch = grids[2 * f], grids[2 * f + 1]
        mag = np.hypot(i_ch, q_ch)
        logmag = np.log1p(np.clip(mag, 0, None))
        for name, arr in (("I", i_ch), ("Q", q_ch), ("mag", mag)):
            feats[f"f{f + 1}_{name}_mean"] = float(arr.mean())
            feats[f"f{f + 1}_{name}_std"] = float(arr.std())
            feats[f"f{f + 1}_{name}_min"] = float(arr.min())
            feats[f"f{f + 1}_{name}_max"] = float(arr.max())
            feats[f"f{f + 1}_{name}_ptp"] = float(np.ptp(arr))
            feats[f"f{f + 1}_{name}_median"] = float(np.median(arr))
        feats[f"f{f + 1}_logmag_mean"] = float(logmag.mean())
        feats[f"f{f + 1}_logmag_std"] = float(logmag.std())
        fft = np.abs(np.fft.fft2(logmag))
        feats[f"f{f + 1}_fft_mean"] = float(fft.mean())
        feats[f"f{f + 1}_fft_std"] = float(fft.std())
        feats[f"f{f + 1}_fft_max"] = float(fft.max())
    return feats


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = pd.read_csv(REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv")
    sig = table[table.has_signal].reset_index(drop=True)

    grids_out = np.zeros((len(sig), 8, BOX_H, BOX_W), dtype=np.float32)
    feat_rows, blocked = [], []
    for k, row in sig.iterrows():
        path = REPO / "data" / "raw" / "EddyCus-HDF5" / "output" / row.file
        try:
            g, valid, info = _load_native_grid(path)  # (8,H,W) float32
        except BlockingError as e:
            blocked.append({"scan_id": row.scan_id, "reason": str(e)})
            continue
        # 空洞以逐通道中位数填充（避免"扫描边缘"伪特征）
        fill = torch.from_numpy(
            np.array([np.median(c[valid]) if valid.any() else 0.0 for c in g], dtype=np.float32))
        g2 = g.copy()
        g2[:, ~valid] = fill.numpy()[:, None]
        grids_out[k] = fit_box_resize(torch.from_numpy(g2), fill).numpy()
        feat_rows.append({"scan_id": row.scan_id, **classical_features(g)})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(sig)} elapsed {time.time() - t0:.0f}s", flush=True)

    np.save(CACHE / "grid_96x384.npy", grids_out)
    pd.DataFrame(feat_rows).to_csv(CACHE / "features_classical.csv", index=False)
    (CACHE / "scan_order.json").write_text(json.dumps(
        {"scan_ids": sig.scan_id.tolist(), "box": [BOX_H, BOX_W],
         "channels": "4 freqs x (I,Q), holes=per-channel median, aspect-preserving resize+pad"},
        indent=1))
    meta = {
        "runtime_s": round(time.time() - t0, 1),
        "n_scans": len(sig), "n_built": len(feat_rows), "n_blocked": len(blocked),
        "blocked": blocked, "box": [BOX_H, BOX_W],
        "input_mapping_record": "raw I/Q (float64→float32) → (8,H,W) native grid → "
                                "holes filled per-channel median → aspect-preserving "
                                "area-resize into (96,384) + median pad; 幅值未做全局归一化",
    }
    (OUT_RESULTS / "eddycus_feature_build.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "blocked"}, indent=1))


if __name__ == "__main__":
    main()
