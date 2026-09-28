#!/usr/bin/env python3
"""S0: external_weld_ut (Strathclyde FMC/PAUT) 只读审计 — F 系列前置。

目的: 落定 FMC .mat 的变量名/形状/dtype/**轴序假设**与 PAUT.zip 的 B-scan 格式,
输出 data/manifests/external_weld_ut/audit.json, 供 preprocess 脚本按可审计参数运行。

轴序判定方法 (只读, 无监督假设):
- 时间轴特征: 波形平滑 + 能量集中于前段 (超声回波); 非时间轴 (发射/接收元素) 上
  逐元素近似独立噪声。对候选轴计算相邻点差分的自相关/平滑度区分。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path("data/raw/external_weld_ut")
OUT = Path("data/manifests/external_weld_ut/audit.json")

FMC_FILES = {
    "A": ROOT / "A" / "Lack_of_fusion_FMC_DORT_2016.mat",
    "B": ROOT / "B" / "FMC_2012_04_26_at_16_16.mat",
    "C": ROOT / "C" / "FMC_RR3_2_25MHz_3mmsdh.mat",
}
PAUT_ZIP = ROOT / "D" / "PAUT.zip"


def axis_scores(arr: np.ndarray, axis: int, probe_idx: dict) -> float:
    """平滑度得分: 相邻差分 RMS / 信号 RMS, 越小越平滑 (时间轴应显著小于 1)。
    probe_idx 固定其它轴的探针位置。"""
    move = [slice(None)] * arr.ndim
    for ax, ix in probe_idx.items():
        if ax != axis:
            move[ax] = ix
    prof = arr[tuple(move)].astype(np.float64)
    rms = float(np.sqrt(np.mean(prof**2)))
    if rms < 1e-12:
        return 1e9
    diff_rms = float(np.sqrt(np.mean(np.diff(prof) ** 2)))
    return diff_rms / rms


def audit_fmc(tag: str, path: Path) -> dict:
    info = sio.whosmat(path)
    entries = [{"name": n, "shape": list(s), "dtype": d} for n, s, d in info]
    var, (d0, d1, d2), _ = info[0]
    raw = sio.loadmat(path)[var]
    shape = raw.shape
    print(f"[{tag}] {var} {shape} {raw.dtype} min={raw.min()} max={raw.max()}")
    # 三个候选轴各测平滑度 (探针位置固定中段)
    probe = {1: shape[1] // 2, 2: shape[2] // 2}
    scores = {ax: axis_scores(raw, ax, probe) for ax in range(3)}
    time_axis = min(scores, key=scores.get)
    print(f"[{tag}] 平滑度得分 (越小=时间轴): "
          + ", ".join(f"ax{ax}={s:.3f}" for ax, s in scores.items())
          + f" → time_axis={time_axis}")
    # 能量集中度 (时间轴前 1/4)
    move = [slice(None)] * 3
    move[time_axis] = slice(None)
    for ax, ix in probe.items():
        if ax != time_axis:
            move[ax] = ix
    prof = raw[tuple(move)].astype(np.float64) ** 2
    q = len(prof) // 4
    e_front = float(prof[:q].sum() / max(prof.sum(), 1e-12))
    print(f"[{tag}] 时间轴前1/4能量占比: {e_front:.3f}")
    out = {
        "tag": tag, "file": str(path), "variable": var,
        "shape": list(shape), "dtype": str(raw.dtype),
        "min": int(raw.min()) if raw.dtype.kind in "iu" else float(raw.min()),
        "max": int(raw.max()) if raw.dtype.kind in "iu" else float(raw.max()),
        "smoothness_scores": {f"ax{a}": round(s, 4) for a, s in scores.items()},
        "time_axis": int(time_axis),
        "energy_front_quarter": round(e_front, 4),
        "whosmat": entries,
    }
    del raw
    return out


def audit_paut_zip(path: Path) -> dict:
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.endswith(".txt")]
    sizes = [z.getinfo(n).file_size for n in names]
    with z.open(names[0]) as f:
        arr = np.loadtxt(f)
    prefixes = sorted({Path(n).stem[0] for n in names})
    print(f"[D] {len(names)} 个 txt B-scan, 前缀组 {prefixes}, "
          f"单文件 shape {arr.shape}, min={arr.min():.0f} max={arr.max():.0f}")
    return {
        "tag": "D", "file": str(path), "n_txt": len(names),
        "prefix_groups": prefixes,
        "example_shape": list(arr.shape),
        "example_min": float(arr.min()), "example_max": float(arr.max()),
        "total_mb": round(sum(sizes) / 1e6, 1),
    }


def main() -> None:
    results = {"fmc": [], "paut": None}
    for tag in ("B", "C", "A"):  # 小 → 大, 逐个加载释放
        results["fmc"].append(audit_fmc(tag, FMC_FILES[tag]))
    results["paut"] = audit_paut_zip(PAUT_ZIP)
    results["decision"] = {
        "axis_order": {
            "A": "(T, Ne, Nr) — FMC_new (10000,128,128) int32, time_axis=ax0",
            "B": "(T, Ne, Nr) — FMC_new (10000,45,45) int32, time_axis=ax0",
            "C": "(Ne, Nr, T) — fmc (128,128,976) float64, time_axis=ax2",
        },
        "representation": "FMC: 每发射元素一个 view, 接收维均匀降采样 16, 时间 max-pool → 512; "
                          "PAUT(D): 每 txt 一个 view, 位置维均匀降采样 16, 时间 401→零填充 512+valid_mask",
        "n_views_expected": {"A": 128, "B": 45, "C": 128, "D": 389, "total": 690},
        "dtype_note": "A/B 为 int32 容器存 int16 值域; C 为 float64 — 统一转 float32 输出",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"audit → {OUT}")


if __name__ == "__main__":
    main()
