#!/usr/bin/env python3
"""S1: external_weld_ut (Strathclyde FMC/PAUT) 预处理 → data/processed/external_weld_ut/。

输入 (data/raw/external_weld_ut/, 先校验 checksums.txt):
  A/B  FMC_new (T, Ne, Nr) int32   — 每发射元素一个 view
  C    fmc     (Ne, Nr, T) float64 — 每发射元素一个 view
  D    PAUT.zip 389 个 txt (762 位置 × 401 时间), 振幅 (非负, 已包络化)

表示 (F 系列预训练语料, 1D 形态 (C=16, T=512), 与 PENELOPE 同通路走 Stem1D):
  FMC: view e → (Nr, T) → 接收维 np.linspace 固定索引降采样 16 → 时间 max-|·| 池化到 512
       (原始 RF 双极性, max|·| 保峰且保留极性; PENELOPE 为已包络数据用 max-pool, 此处等价推广)
  PAUT: 每 txt 一个 view → 位置维固定索引降采样 16 → 时间 401 零填充到 512,
       valid_mask (T,) = [1]*401 + [0]*111 (collate/token_valid_mask 自动排除 padding patch)

输出:
  signals.npy        (690, 16, 512) float32
  manifest.parquet   sample_id / group_id / source_file / view_index / n_rx_raw / n_time_raw /
                     time_pool / rx_keep / padded_time / modality / label(None) / split_group
  meta_summary.json  逐组统计 + 参数 + 审计引用 (audit.json 的轴序结论)
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path("data/raw/external_weld_ut")
OUT = Path("data/processed/external_weld_ut")
AUDIT = Path("data/manifests/external_weld_ut/audit.json")

RX_KEEP = 16
TIME_KEEP = 512
FMC_FILES = {
    "A": (ROOT / "A" / "Lack_of_fusion_FMC_DORT_2016.mat", "FMC_new", "T_first"),
    "B": (ROOT / "B" / "FMC_2012_04_26_at_16_16.mat", "FMC_new", "T_first"),
    "C": (ROOT / "C" / "FMC_RR3_2_25MHz_3mmsdh.mat", "fmc", "T_last"),
}


def verify_checksums() -> None:
    ok = True
    for line in (ROOT / "checksums.txt").read_text().strip().splitlines():
        tag, name, size, md5, sha256 = line.split("|")
        p = ROOT / tag / name
        if int(size) != p.stat().st_size:
            ok = False
            print(f"[checksum] {tag} 尺寸不符")
            continue
        h = hashlib.md5()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            ok = False
            print(f"[checksum] {tag} md5 不符")
    if not ok:
        raise SystemExit("checksum 校验失败, 中止预处理")


def downsample_maxabs(x: np.ndarray, target: int) -> np.ndarray:
    """最后一维 max-|·| 池化到 target bins (双极性 RF 保峰保极性;
    非负包络输入时与 max-pool 等价)。"""
    L = x.shape[-1]
    if L == target:
        return x
    if L < target:
        idx = np.linspace(0, L - 1, target)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, L - 1)
        frac = (idx - lo).astype(np.float32)
        return x[..., lo] * (1 - frac) + x[..., hi] * frac
    parts = np.array_split(x, target, axis=-1)             # 末段 window 可不等长 (同 PENELOPE)
    out = np.empty(x.shape[:-1] + (target,), dtype=np.float32)
    for j, p in enumerate(parts):
        pick = np.abs(p).argmax(axis=-1)
        out[..., j] = np.take_along_axis(p, pick[..., None], axis=-1)[..., 0]
    return out


def rx_indices(n_rx: int) -> np.ndarray:
    return np.linspace(0, n_rx - 1, RX_KEEP).round().astype(int)


def process_fmc(tag: str, rows: list[dict], mm: np.ndarray) -> None:
    path, var, order = FMC_FILES[tag]
    print(f"[{tag}] loadmat {path.name} ...")
    raw = sio.loadmat(path)[var]
    if order == "T_first":                                  # (T, Ne, Nr) → view (Nr, T)
        n_views, n_rx, n_time = raw.shape[1], raw.shape[2], raw.shape[0]
        views = (raw[:, e, :] for e in range(n_views))      # (T, Nr) 生成器, 不复制
    else:                                                   # (Ne, Nr, T) → view (Nr, T)
        n_views, n_rx, n_time = raw.shape[0], raw.shape[1], raw.shape[2]
        views = (raw[e, :, :] for e in range(n_views))
    ridx = rx_indices(n_rx)
    print(f"[{tag}] {var} views={n_views} rx={n_rx} time={n_time} → rx_keep={RX_KEEP}, pool→{TIME_KEEP}")
    for e, v in enumerate(views):
        sig = np.asarray(v, dtype=np.float32)[ridx, :]      # (RX_KEEP, n_time)
        sig = downsample_maxabs(sig, TIME_KEEP).astype(np.float32)
        mm[len(rows)] = sig
        rows.append({
            "sample_id": f"external_weld_ut:{tag}{e:03d}",
            "group_id": tag, "source_file": str(path), "view_index": e,
            "n_rx_raw": int(n_rx), "n_time_raw": int(n_time),
            "rx_keep": RX_KEEP, "time_pool": TIME_KEEP, "padded_time": False,
            "modality": "ultrasonic", "label": None, "label_type": "none",
            "split_group": f"specimen:{tag}",
        })
    del raw


def process_paut(rows: list[dict], mm: np.ndarray) -> None:
    path = ROOT / "D" / "PAUT.zip"
    z = zipfile.ZipFile(path)
    names = sorted(n for n in z.namelist() if n.endswith(".txt"))
    print(f"[D] {len(names)} txt B-scans")
    for i, name in enumerate(names):
        with z.open(name) as f:
            arr = np.loadtxt(f)                             # (762 位置, 401 时间)
        pidx = rx_indices(arr.shape[0])
        sig = arr[pidx, :].astype(np.float32)               # (RX_KEEP, 401)
        n_time = sig.shape[1]
        pad = np.zeros((RX_KEEP, TIME_KEEP), dtype=np.float32)
        pad[:, :n_time] = sig
        mm[len(rows)] = pad
        rows.append({
            "sample_id": f"external_weld_ut:D{name.split('/')[-1].replace('.txt', '')}",
            "group_id": "D", "source_file": f"{path}!{name}", "view_index": i,
            "n_rx_raw": int(arr.shape[0]), "n_time_raw": int(n_time),
            "rx_keep": RX_KEEP, "time_pool": TIME_KEEP, "padded_time": True,
            "modality": "ultrasonic", "label": None, "label_type": "none",
            "split_group": "specimen:D",
        })


def main() -> None:
    verify_checksums()
    audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else None
    if audit is None:
        raise SystemExit("缺 data/manifests/external_weld_ut/audit.json — 先跑 audit_external_weld_ut.py")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    mm = np.lib.format.open_memmap(
        OUT / "signals.npy", mode="w+", dtype=np.float32,
        shape=(690, RX_KEEP, TIME_KEEP))
    for tag in ("A", "B", "C"):
        process_fmc(tag, rows, mm)
    process_paut(rows, mm)
    mm.flush()
    del mm

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "manifest.parquet", index=False)
    n_by_group = df.groupby("group_id").size().to_dict()
    # 逐组幅值统计 (探测组间量级差, 供训练 z-score 归一化必要性确认)
    sig = np.load(OUT / "signals.npy", mmap_mode="r")
    stats = {
        g: {
            "n": int(n_by_group[g]),
            "rms": round(float(np.sqrt(np.mean(np.asarray(sig[df.index[df.group_id == g]].astype(np.float64)) ** 2))), 2),
        }
        for g in sorted(n_by_group)
    }
    meta = {
        "source": "external_weld_ut (Strathclyde, CC BY 4.0; M0-3 audit)",
        "audit": "data/manifests/external_weld_ut/audit.json",
        "axis_order_decision": audit["decision"]["axis_order"],
        "params": {
            "rx_keep": RX_KEEP, "time_keep": TIME_KEEP,
            "rx_method": "np.linspace 固定索引 (确定性, 不随机)",
            "time_method": "max-|·| 池化 (双极性 RF 保峰保极性; D 已包络, 等价 max-pool)",
            "padded_time": "D: 401 → 512 零填充, valid_time=401",
        },
        "n_samples": int(len(df)), "n_by_group": {k: int(v) for k, v in n_by_group.items()},
        "group_stats": stats,
        "labels": "无逐位置缺陷标签 (仅整试件已知缺陷) → 仅 SSL 预训练, 禁止评测 (tier B)",
        "split_groups": sorted(df.split_group.unique()),
    }
    (OUT / "meta_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"完成: {len(df)} 样本 → {OUT} (signals.npy / manifest.parquet / meta_summary.json)")
    print(json.dumps(meta["n_by_group"], ensure_ascii=False))


if __name__ == "__main__":
    main()
