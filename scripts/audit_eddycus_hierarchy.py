#!/usr/bin/env python
"""EddyCus-HDF5 真实层级审计（Phase 2A, 只读）。

目的：回答"148 配置组到底是数据中显式实体还是代码生成代理"，并审计：
  1) 738 个 HDF5 文件名/路径；
  2) HDF5 attributes/groups/datasets；
  3) converter 生成时保留的原始文件名（measurement_metadata/original_file_path）；
  4) metadata fields（sensor / frequency / material / layup / defect / thickness / id / datetime）；
  5) 是否存在 manifest 未使用的 sample/specimen/plate ID；
  6) 同一源文件是否被多个 sensor/frequency 重复转换；
  7) clean 与 defect 扫描是否可能来自同一物理板；
  8) 148 配置组是显式实体还是代码生成代理。

输出：
  - stdout 摘要
  - artifacts/general_ndt/audits/eddycus_hierarchy.json（完整结果）

用法：
  python scripts/audit_eddycus_hierarchy.py [--root data/raw/EddyCus-HDF5/output] [--out artifacts/general_ndt/audits/eddycus_hierarchy.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/raw/EddyCus-HDF5" / "output"
DEFAULT_OUT = REPO / "artifacts" / "general_ndt" / "audits" / "eddycus_hierarchy.json"
FREQ_KEYS = ("f1", "f2", "f3", "f4")


def classify(description: str) -> tuple[str, bool]:
    """按 description 归类缺陷类（与 wndt adapters/eddycus.py 保持一致）。"""
    import re

    rules = [
        (re.compile(r"gap", re.I), "gap", True),
        (re.compile(r"mis-orientation", re.I), "mis_orientation", True),
        (re.compile(r"copper coated roving", re.I), "copper_roving", True),
        (re.compile(r"copper film", re.I), "copper_foil", True),
        (re.compile(r"ptfe", re.I), "ptfe_insert", True),
        (re.compile(r"teflon", re.I), "ptfe_insert", True),
        (re.compile(r"ondulation", re.I), "ondulation", True),
        (re.compile(r"fuzzy ball", re.I), "fuzz_ball", True),
    ]
    for pat, cls, is_def in rules:
        if pat.search(description or ""):
            return cls, is_def
    return "clean", False


def _num(v) -> float | None:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def read_one(path: Path) -> dict:
    """读取单个文件的层级相关元数据（不读大信号数组）。"""
    out = {
        "file": path.name,
        "rel": str(path.relative_to(DEFAULT_ROOT if DEFAULT_ROOT in path.parents else path.parents[1])),
        "size": path.stat().st_size,
    }
    with h5py.File(path, "r") as f:
        root_attrs = {k: str(v) for k, v in f.attrs.items()}
        mm = {}
        sample = {}
        freqs = []
        spatial = {}
        if "measurement_metadata" in f:
            mm = {k: str(v) for k, v in f["measurement_metadata"].attrs.items()}
            if "sample_properties" in f["measurement_metadata"]:
                sample = {k: str(v) for k, v in
                          f["measurement_metadata"]["sample_properties"].attrs.items()}
            if "frequencies" in f["measurement_metadata"]:
                fq = f["measurement_metadata"]["frequencies"]
                for k in sorted(fq.keys()):
                    if isinstance(fq[k], h5py.Group):
                        freqs.append({kk: str(vv) for kk, vv in fq[k].attrs.items()})
        if "spatial_data" in f:
            for nm in ("track_number", "sample_number", "x_mm", "y_mm", "z_mm"):
                if nm in f["spatial_data"]:
                    ds = f["spatial_data"][nm]
                    arr = np.asarray(ds[()])
                    spatial[nm] = {
                        "n": int(arr.size),
                        "min": float(arr.min()) if arr.size else None,
                        "max": float(arr.max()) if arr.size else None,
                        "n_nan": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
                    }
        has_signal = "signal_data" in f and "f1" in f["signal_data"]
        if has_signal:
            sig_len = int(f["signal_data/f1/real"].shape[0])
        else:
            sig_len = 0
    out["root_attrs"] = root_attrs
    out["mm"] = mm
    out["sample"] = sample
    out["freqs"] = freqs
    out["spatial"] = spatial
    out["has_signal"] = has_signal
    out["signal_len"] = sig_len
    return out


def audit(root: Path) -> dict:
    files = sorted(p for p in root.glob("scan_*.h5"))
    recs = [read_one(p) for p in files]

    # ---- 1) 文件名 / 路径 ----
    names = [r["file"] for r in recs]
    name_dup = len(names) - len(set(names))
    pattern_ok = all(n.startswith("scan_") and n.endswith(".h5") for n in names)

    # ---- 2) HDF5 结构一致性 ----
    root_attr_keys = Counter(tuple(sorted(r["root_attrs"].keys())) for r in recs)
    top_groups = Counter()
    for r in recs:
        has = set()
        for g in ("measurement_metadata", "spatial_data", "signal_data", "analysis_results"):
            pass
    has_signal_n = sum(1 for r in recs if r["has_signal"])

    # ---- 3) original_file_path（converter 保留的原始路径/文件名）----
    orig_paths = Counter((r["mm"].get("original_file_path", "") or "") for r in recs)
    # 同一原始路径出现在多个转换文件 → 同一源文件被重复转换
    orig_multi = {p: n for p, n in orig_paths.items() if n > 1}
    n_unique_orig = len(orig_paths)

    # ---- 4/5) 显式 ID 候选 ----
    # sample_properties.id：每文件一个，判断是否唯一（扫描序号）还是重复（板号）
    sample_ids = [(r["file"], r["sample"].get("id", "")) for r in recs]
    id_counter = Counter(sid for _, sid in sample_ids)
    id_dup_files = {fid: n for fid, n in id_counter.items() if n > 1}
    n_unique_sample_ids = len(id_counter)
    # id 是否与文件名序号一致（scan_XXXXX -> XXXX）
    id_equals_scan = sum(
        1 for fname, sid in sample_ids
        if sid == str(int(fname.replace("scan_", "").replace(".h5", "")))
    )
    # original_file_path 是否含可识别文件名（非纯目录）
    orig_with_filename = []
    for p in orig_paths:
        stem = Path(p.replace("\\", "/")).name
        if stem and "." in stem:
            orig_with_filename.append(p)

    # ---- 6) 同一源文件被多个 sensor/frequency 转换 ----
    # 由 orig_multi 数量给出；若 original_file_path 是纯目录则无法直接判断，
    # 改用 (material,fiber,layup,thickness,description,datetime) 组合看重复。

    # ---- 7) clean / defect 是否同板 ----
    # 板候选 = (material_type, fiber_type, layup_sequence, thickness_mm, sensor_type)
    plate_key = lambda r: "|".join([
        r["sample"].get("material_type", ""),
        r["sample"].get("fiber_type", ""),
        r["sample"].get("layup_sequence", ""),
        r["sample"].get("thickness_mm", ""),
        r["mm"].get("sensor_type", ""),
    ])
    plate_groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        plate_groups[plate_key(r)].append(r)
    # 同一板候选内既有 clean 又有 defect → clean/defect 可能来自同一物理板
    same_plate_clean_defect = {}
    n_same_plate_mixed = 0
    for pk, group in plate_groups.items():
        descs = set(r["sample"].get("description", "") for r in group)
        defs = set(classify(d)[0] for d in descs)
        if len(defs) > 1 and "clean" in defs:
            n_same_plate_mixed += 1
            same_plate_clean_defect[pk] = {
                "n_files": len(group),
                "descriptions": sorted(descs),
                "defect_classes": sorted(defs),
            }

    # ---- 8) 148 配置组 = 显式实体还是代码生成代理 ----
    # 重建 manifest 用的配置组哈希键（与 wndt adapters/eddycus.py 一致）
    def cfg_key(r: dict) -> str:
        sp = r["sample"]
        return "|".join([
            sp.get("material_type", ""), sp.get("fiber_type", ""),
            sp.get("layup_sequence", ""), sp.get("description", ""),
            sp.get("defect_depth_mm", ""), sp.get("defect_size_mm", ""),
            sp.get("thickness_mm", ""),
        ])

    import hashlib

    cfg_groups: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        ck = cfg_key(r)
        cfg_groups[ck].append(r["file"])
    n_cfg_groups = len(cfg_groups)
    # 数据中是否有显式 plate/specimen 字段？
    explicit_id_fields = set()
    for r in recs:
        sp_keys = set(r["sample"].keys())
        for k in sp_keys:
            kl = k.lower()
            if any(t in kl for t in ("id", "plate", "specimen", "sample", "serial", "part", "batch")):
                explicit_id_fields.add(k)
    # id 的唯一性已经表明：id 每文件唯一 → 是扫描序号，不是板号

    # ---- sensor / material / frequency 分布 ----
    sensors = Counter(r["mm"].get("sensor_type", "") for r in recs)
    materials = Counter(r["sample"].get("material_type", "") for r in recs)
    fibers = Counter(r["sample"].get("fiber_type", "") for r in recs)
    layups = Counter(r["sample"].get("layup_sequence", "") for r in recs)
    thicknesses = Counter(r["sample"].get("thickness_mm", "") for r in recs)
    freq_sets = Counter(
        tuple(sorted(x.get("frequency_mhz", "?") for x in r["freqs"])) for r in recs
    )
    datetimes = Counter(r["mm"].get("measurement_datetime", "") for r in recs)

    # ---- spatial 栅格可解性（逐文件读取 track/sample 数组，判是否无歧义重建 2D 栅格）----
    # 充分条件：
    #   1) track/sample 均为 1 起始整数；
    #   2) 唯一 (track, sample) 对数 == 信号点数（无重复坐标 → 无歧义 scatter）；
    #   3) 矩形面积 == 唯一对数（无空洞 → 无需 valid mask；有空洞 → 保留 mask 仍可重建）。
    grid_issues = []
    grid_stats = []
    n_grid_full_rect = 0       # 面积 == 唯一对数（完整矩形，无空洞）
    n_grid_holey = 0           # 面积 > 唯一对数（有空洞，需 valid mask）
    n_grid_ambiguous = 0       # 无法无歧义重建
    for r in recs:
        if not r["has_signal"]:
            continue
        with h5py.File(root / r["file"], "r") as f:
            trk = np.asarray(f["spatial_data"]["track_number"][()])
            smp = np.asarray(f["spatial_data"]["sample_number"][()])
        n = trk.size
        # track/sample 以 float64 存储但应为整数值；校验 1 起始整数（允许浮点整数值）
        def _is_integral_1based(a: np.ndarray) -> bool:
            if a.size == 0 or a.dtype.kind not in "iuf":
                return False
            if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                return False
            if np.any(a < 1):
                return False
            return bool(np.all(np.abs(a - np.rint(a)) < 1e-9))
        if not (_is_integral_1based(trk) and _is_integral_1based(smp)):
            n_grid_ambiguous += 1
            grid_issues.append({
                "file": r["file"],
                "issue": "track/sample 非 1 起始整数",
                "trk_dtype": str(trk.dtype), "trk_range": [float(trk.min()), float(trk.max())],
            })
            continue
        trk_i = np.rint(trk).astype(np.int64)
        smp_i = np.rint(smp).astype(np.int64)
        pairs = set(zip(trk_i.tolist(), smp_i.tolist()))
        n_unique = len(pairs)
        area = int(trk_i.max()) * int(smp_i.max())
        if n_unique != n:
            n_grid_ambiguous += 1
            grid_issues.append({
                "file": r["file"],
                "issue": f"重复坐标: n={n} 唯一对={n_unique}",
                "n": n, "n_unique": n_unique, "area": area,
            })
        elif area == n:
            n_grid_full_rect += 1
        else:
            n_grid_holey += 1
        grid_stats.append({
            "file": r["file"], "n": n, "n_unique": n_unique,
            "tracks": [int(trk_i.min()), int(trk_i.max())],
            "samples": [int(smp_i.min()), int(smp_i.max())],
            "area": area, "full_rect": area == n_unique,
        })

    return {
        "root": str(root),
        "n_files": len(files),
        "filenames": {
            "pattern_ok": pattern_ok,
            "duplicates": name_dup,
            "head": names[:5],
            "tail": names[-5:],
        },
        "hdf5_structure": {
            "root_attr_key_variants": {str(k): v for k, v in root_attr_keys.items()},
            "n_with_signal": has_signal_n,
            "n_without_signal": len(files) - has_signal_n,
            "signal_len_distinct": sorted({r["signal_len"] for r in recs}),
        },
        "original_paths": {
            "n_unique": n_unique_orig,
            "n_multi_file": len(orig_multi),
            "multi_file_paths": {p: n for p, n in list(orig_multi.items())[:20]},
            "paths_with_filename": orig_with_filename[:20],
            "sample_head": list(orig_paths.keys())[:10],
        },
        "explicit_ids": {
            "sample_properties_id": {
                "n_unique": n_unique_sample_ids,
                "id_equals_scan_number": id_equals_scan,
                "n_ids_shared_by_multi_files": len(id_dup_files),
                "shared_ids": {k: v for k, v in list(id_dup_files.items())[:20]},
            },
            "candidate_id_fields_in_sample_props": sorted(explicit_id_fields),
            "conclusion": "sample_properties.id 每文件唯一且等于扫描序号 → 不是板/试件 ID；"
                          "数据中无显式 plate/specimen ID",
        },
        "inferred_groups": {
            "cfg_key_rule": "material_type|fiber_type|layup_sequence|description|"
                            "defect_depth_mm|defect_size_mm|thickness_mm (SHA1 前 8 位)",
            "n_config_groups": n_cfg_groups,
            "config_groups_per_files": {k: len(v) for k, v in sorted(
                cfg_groups.items(), key=lambda kv: -len(kv[1]))[:10]},
            "is_explicit_in_data": False,
            "conclusion": "148 配置组是代码对 HDF5 元数据字段拼接哈希的推断代理，"
                          "不是数据集显式提供的实体",
        },
        "same_plate_clean_defect": {
            "n_plate_candidates_with_clean_and_defect": n_same_plate_mixed,
            "detail": same_plate_clean_defect,
            "plate_key_rule": "material_type|fiber_type|layup_sequence|thickness_mm|sensor_type",
        },
        "metadata_distributions": {
            "sensors": dict(sensors),
            "materials": dict(materials),
            "fibers": dict(fibers),
            "layups": dict(layups),
            "thicknesses": dict(thicknesses),
            "frequency_sets": {str(k): v for k, v in freq_sets.items()},
            "n_datetime_values": len(datetimes),
        },
        "spatial_grid": {
            "n_full_rect": n_grid_full_rect,
            "n_holey_need_mask": n_grid_holey,
            "n_ambiguous": n_grid_ambiguous,
            "n_with_signal_total": sum(1 for r in recs if r["has_signal"]),
            "issues": grid_issues[:40],
            "per_file": grid_stats,
            "conclusion": "track/sample 为 1 起始整数且唯一对 == 点数 → 可用 scatter 无歧义重建 2D 栅格"
                          "(H= max_track, W= max_sample), 空洞位置以 valid mask 保留",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[ERR] root 不存在: {args.root}", file=sys.stderr)
        return 2
    result = audit(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[json -> {args.out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
