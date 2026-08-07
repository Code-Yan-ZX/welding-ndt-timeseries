#!/usr/bin/env python
"""预处理 PAUT 多视角数据 (P0-4)。

每个试件有两个 .nde (90族 / 270族), 每个含 G0(71°, 49波束, 3500采样) 与
G1(47°, 22波束, 4500采样)。本脚本提取 4 个视角:

  V0: 90族 / G0  (71°, 49bm)   -- 即现有单视角基线所用
  V1: 270族/ G0  (71°, 49bm)   -- 同角度、反向侧 (skew 270 vs 90)
  V2: 90族 / G1  (47°, 22bm)   -- 不同折射角
  V3: 270族/ G1  (47°, 22bm)

每个视角: 时间轴 max-pool 下采样到 512; G1 的 22 波束线性插值到 49 以对齐空间
维度。4 视图堆叠为 (N, 4, 49, 512) -> ascans_mv.npy。

注意: 个别试件 (PP6) 的 90/270 扫描位置数不一致 (601 vs 596), 按每个试件取
两族最小位置数截断对齐 (假设两族扫描起点/分辨率一致, position i 对应同一 x)。
相应地构建对齐的 meta_label_mv / meta_coupon_mv / meta_pos_mv / splits_mv.npz。

Usage: python scripts/paut_preprocess_multiview.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from paut_preprocess import (  # noqa: E402
    ROOT_DEFAULT, OUT_DEFAULT, TRAIN_COUPONS, VAL_COUPONS, TEST_COUPONS,
    read_group0, downsample_max,
)

TARGET_LEN = 512
TARGET_BEAMS = 49
VIEWS = ["90/G0(71°)", "270/G0(71°)", "90/G1(47°)", "270/G1(47°)"]


def find_family_file(ndt_dir: Path, family: str) -> Path:
    cands = sorted(ndt_dir.glob("*.nde"))
    for p in cands:
        n = p.name
        if family == "270":
            if "270" in n:
                return p
        else:
            if "270" not in n and ("PAUT_90" in n or n.endswith("_90.nde")):
                return p
    raise FileNotFoundError(f"no {family}-family .nde under {ndt_dir}")


def resample_beams(x: np.ndarray, target: int = TARGET_BEAMS) -> np.ndarray:
    """沿倒数第二轴 (beam) 线性插值到 target。x: (..., B, T)。"""
    B = x.shape[-2]
    if B == target:
        return x
    idx = np.linspace(0, B - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, B - 1)
    frac = (idx - lo).astype(np.float32)
    return x[..., lo, :] * (1 - frac)[None, :, None] + x[..., hi, :] * frac[None, :, None]


def extract_family_views(nde: Path) -> np.ndarray:
    """读 G0 与 G1, 下采样+波束对齐, 返回 (n_pos, 2, 49, 512) [G0, G1]。"""
    g0, _, _ = read_group0(nde, 0)
    g1, _, _ = read_group0(nde, 1)
    g0 = downsample_max(g0.astype(np.float32), TARGET_LEN)
    g1 = downsample_max(g1.astype(np.float32), TARGET_LEN)
    g1 = resample_beams(g1, TARGET_BEAMS)
    return np.stack([g0, g1], axis=1)


def main() -> None:
    out = OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)
    # 既有 meta (position 级, 基于 90 族) 用于对齐标签
    labels_all = np.load(out / "meta_label.npy")
    coupons_all = np.load(out / "meta_coupon.npy")
    pos_all = np.load(out / "meta_pos.npy")

    mv_chunks, lab_chunks, coup_chunks, pos_chunks = [], [], [], []
    cursor = 0
    print(f"=== PAUT 多视角预处理 | 4 视图 × {TARGET_BEAMS}bm × {TARGET_LEN} ===")
    for c in TRAIN_COUPONS + VAL_COUPONS + TEST_COUPONS:
        ndt = ROOT_DEFAULT / c / "2. ndt_data"
        f90 = find_family_file(ndt, "90")
        f270 = find_family_file(ndt, "270")
        v90 = extract_family_views(f90)     # (n90, 2, 49, 512)
        v270 = extract_family_views(f270)   # (n270, 2, 49, 512)
        n90, n270 = v90.shape[0], v270.shape[0]
        min_n = min(n90, n270)
        if n90 != n270:
            print(f"  [警告] {c}: 90族={n90} vs 270族={n270} 位置数不一致, 截断到 {min_n}")
        v90, v270 = v90[:min_n], v270[:min_n]
        mv = np.concatenate([v90, v270], axis=1)          # (min_n, 4, 49, 512)
        # 对齐 meta: 取该 coupon 前 min_n 个位置 (既有 meta 中该 coupon 段)
        seg = np.nonzero(coupons_all == c)[0]
        assert len(seg) == n90, f"{c}: 既有 meta 段长 {len(seg)} != 90族 n_pos {n90}"
        seg = seg[:min_n]
        mv_chunks.append(mv.astype(np.float32))
        lab_chunks.append(labels_all[seg])
        coup_chunks.append(coupons_all[seg])
        pos_chunks.append(pos_all[seg])
        print(f"  {c}: 90={f90.name} 270={f270.name} | n_pos={min_n} (90:{n90},270:{n270})")

    ascans_mv = np.concatenate(mv_chunks, axis=0)
    labels_mv = np.concatenate(lab_chunks)
    coupons_mv = np.concatenate(coup_chunks)
    pos_mv = np.concatenate(pos_chunks)
    np.save(out / "ascans_mv.npy", ascans_mv)
    np.save(out / "meta_label_mv.npy", labels_mv)
    np.save(out / "meta_coupon_mv.npy", coupons_mv)
    np.save(out / "meta_pos_mv.npy", pos_mv)

    # splits_mv: 与单视角同策略 (train=PP3/4/5, val=PP6, test=PP7) 供单点参考
    def idx_of(cs):
        return np.nonzero(np.isin(coupons_mv, cs))[0].astype(np.int64)
    tr, va, te = idx_of(TRAIN_COUPONS), idx_of(VAL_COUPONS), idx_of(TEST_COUPONS)
    np.savez(out / "splits_mv.npz", train=tr, val=va, test=te)

    # 多视角归一化统计 (全量参考; LOOCV 按折在 train 上重算)
    flat = ascans_mv.reshape(-1, ascans_mv.shape[-1])
    ts_mean = flat.mean(axis=0).astype(np.float32)
    ts_std = (flat.std(axis=0) + 1e-8).astype(np.float32)
    stats = {"per_timestep": {"mean": ts_mean.tolist(), "std": ts_std.tolist()},
             "global": {"mean": float(flat.mean()), "std": float(flat.std() + 1e-8)},
             "n_views": 4, "target_len": TARGET_LEN, "n_beams": TARGET_BEAMS,
             "views": VIEWS, "n_samples": int(len(ascans_mv)),
             "defect_rate": float(labels_mv.mean())}
    with open(out / "norm_stats_mv.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nascans_mv: {ascans_mv.shape} | N={len(ascans_mv)} (单视角 N=3000, "
          f"PP6 截断 601->596) | defect_rate={labels_mv.mean():.4f}")
    print(f"saved -> ascans_mv.npy / meta_*_mv.npy / splits_mv.npz / norm_stats_mv.json")


if __name__ == "__main__":
    main()
