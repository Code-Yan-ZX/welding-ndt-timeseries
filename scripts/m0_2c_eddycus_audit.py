#!/usr/bin/env python
"""M0-2C EddyCus-HDF5 真实数据审计（只读，不训练）。

对 Zenodo 19251759 (EddyCus-HDF5) 解压目录做通用 HDF5 结构探测与全量聚合统计，
供 docs/M0_2C_eddycus_data_audit.md 使用。不修改任何数据文件。

用法:
  python scripts/m0_2c_eddycus_audit.py --probe <file.h5>   # 递归打印单文件结构
  python scripts/m0_2c_eddycus_audit.py --full [--root <dir>] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/raw/EddyCus-HDF5"


def _ds_stats(ds: h5py.Dataset):
    """小数值数据集的统计摘要。大数组只取抽样。"""
    try:
        if ds.size == 0:
            return {"size": 0, "empty": True}
        # 抽样：每维取前 256 个，最多 1<<20 元素
        max_elems = 1 << 20
        if ds.size <= max_elems:
            arr = ds[...]
        else:
            sl = tuple(slice(0, min(s, 256)) for s in ds.shape)
            arr = ds[sl]
        arr = np.asarray(arr)
        # 结构化 dtype（如 [('real','<f8'),('imaginary','<f8>')]）→ 按 I/Q 处理
        if arr.dtype.names:
            fields = {n: np.asarray(arr[n]) for n in arr.dtype.names}
            return {
                "dtype": str(arr.dtype), "shape": list(arr.shape),
                "structured_fields": {
                    n: {"min": float(v.min()), "max": float(v.max()),
                        "mean": float(v.mean()), "std": float(v.std()),
                        "n_nan": int(np.isnan(v).sum())}
                    for n, v in fields.items()
                },
                "sampled": arr.size < ds.size,
            }
        if arr.dtype.kind in "fc":
            if arr.dtype.kind == "c":
                re = arr.real
                im = arr.imag
                return {
                    "dtype": str(arr.dtype), "shape": list(arr.shape),
                    "complex": True,
                    "real": {"min": float(re.min()), "max": float(re.max()),
                             "mean": float(re.mean()), "std": float(re.std())},
                    "imag": {"min": float(im.min()), "max": float(im.max()),
                             "mean": float(im.mean()), "std": float(im.std())},
                    "n_nan": int(np.isnan(arr).sum()), "n_inf": int(np.isinf(arr).sum()),
                    "sampled": arr.size < ds.size,
                }
            return {
                "dtype": str(arr.dtype), "shape": list(arr.shape),
                "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean()), "std": float(arr.std()),
                "n_nan": int(np.isnan(arr).sum()), "n_inf": int(np.isinf(arr).sum()),
                "sampled": arr.size < ds.size,
            }
        # 整数 / 其它
        uniq = None
        if arr.size <= 1 << 16:
            try:
                uniq = sorted(int(x) for x in np.unique(arr))
            except Exception:
                uniq = None
        return {
            "dtype": str(arr.dtype), "shape": list(arr.shape),
            "min": int(arr.min()) if arr.size else None,
            "max": int(arr.max()) if arr.size else None,
            "n_nan": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
            "uniq_n": len(uniq) if uniq is not None else None,
            "uniq_head": uniq[:8] if uniq else None,
        }
    except Exception as e:  # 复合 dtype / 无法读取
        return {"error": str(e), "dtype": str(ds.dtype), "shape": list(ds.shape)}


def probe(path: Path):
    """递归打印一个文件的完整结构。"""
    out = {"file": str(path), "groups": [], "datasets": [], "attrs": {}}
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Group):
                out["groups"].append({
                    "name": name,
                    "attrs": {k: str(v) for k, v in obj.attrs.items()},
                    "n_children": len(obj),
                })
            else:
                out["datasets"].append({
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "attrs": {k: str(v) for k, v in obj.attrs.items()},
                    "stats": _ds_stats(obj),
                })
        f.visititems(walk)
        out["attrs"] = {k: str(v) for k, v in f.attrs.items()}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def full(root: Path, out_path: Path | None):
    """全量聚合：所有 .h5/.hdf5 文件的通用结构 + 统计。"""
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".h5", ".hdf5"))
    agg = {
        "root": str(root),
        "n_files": len(files),
        "files": [],
        "dataset_paths": {},   # name -> {shape_set, dtypes, ...}
    }
    def _group_attrs(g: h5py.Group) -> dict:
        return {k: str(v) for k, v in g.attrs.items()}

    for p in files:
        rec = {"file": p.name, "rel": str(p.relative_to(root)), "size": p.stat().st_size,
               "n_groups": 0, "datasets": [], "attrs": {}, "top_groups": [],
               "meta": {}, "sample": {}, "freqs": []}
        try:
            with h5py.File(p, "r") as f:
                rec["attrs"] = {k: str(v) for k, v in f.attrs.items()}
                # EddyCus 专属：measurement_metadata / sample_properties / frequencies
                if "measurement_metadata" in f:
                    mm = f["measurement_metadata"]
                    rec["meta"] = _group_attrs(mm)
                    if "sample_properties" in mm:
                        rec["sample"] = _group_attrs(mm["sample_properties"])
                    if "frequencies" in mm:
                        fq = mm["frequencies"]
                        for k in sorted(fq.keys()):
                            a = _group_attrs(fq[k])
                            a["_key"] = k
                            rec["freqs"].append(a)
                def walk(name, obj):
                    if isinstance(obj, h5py.Group):
                        rec["n_groups"] += 1
                        if name.count("/") == 0:
                            rec["top_groups"].append(name)
                    else:
                        ds = {
                            "name": name,
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                            "stats": _ds_stats(obj),
                            "attrs": {k: str(v) for k, v in obj.attrs.items()},
                        }
                        rec["datasets"].append(ds)
                        key = name
                        entry = agg["dataset_paths"].setdefault(
                            key, {"shapes": [], "dtypes": [], "files": 0, "size_sample": []})
                        entry["shapes"].append(list(obj.shape))
                        entry["dtypes"].append(str(obj.dtype))
                        entry["files"] += 1
                        if len(entry["size_sample"]) < 3:
                            entry["size_sample"].append({
                                "file": p.name, "size": p.stat().st_size})
                f.visititems(walk)
        except Exception as e:
            rec["error"] = str(e)
        agg["files"].append(rec)

    # 汇总 shape 集合（每个 dataset path 出现的不同 shape）
    for key, entry in agg["dataset_paths"].items():
        entry["shape_set"] = sorted({tuple(s) for s in entry["shapes"]})
        entry["dtype_set"] = sorted(set(entry["dtypes"]))

    # EddyCus 聚合：specimen / sensor / material / defect / freq / 网格
    samples = {}      # sample_properties.id -> {meta, n_files, files[]}
    sensors = {}      # sensor_type -> n_files
    materials = {}    # (material_type, fiber_type, layup_sequence) -> n_files
    defects = {}      # (description, defect_depth_mm, defect_size_mm) -> n_files
    freq_sets = {}    # sorted tuple(freq_mhz) -> n_files
    grid_info = {}    # (n_tracks, n_samples_max) -> n_files
    signal_len = {}   # len per freq -> n_files
    for r in agg["files"]:
        if "error" in r:
            continue
        sp = r.get("sample") or {}
        sid = sp.get("id", "?")
        s = samples.setdefault(sid, {"n_files": 0, "files": []})
        s["n_files"] += 1
        s["files"].append(r["file"])
        if len(s["files"]) == 1:
            s["meta"] = sp
            s["sensor"] = r.get("meta", {}).get("sensor_type", "?")
            s["datetime"] = r.get("meta", {}).get("measurement_datetime", "?")
        st = r.get("meta", {}).get("sensor_type", "?")
        sensors[st] = sensors.get(st, 0) + 1
        mat = (sp.get("material_type", "?"), sp.get("fiber_type", "?"),
               sp.get("layup_sequence", "?"))
        materials[mat] = materials.get(mat, 0) + 1
        dk = (sp.get("description", "?"), sp.get("defect_depth_mm", "?"),
              sp.get("defect_size_mm", "?"))
        defects[dk] = defects.get(dk, 0) + 1
        freqs = tuple(sorted(x.get("frequency_mhz", "?") for x in r.get("freqs", [])))
        freq_sets[freqs] = freq_sets.get(freqs, 0) + 1
        # 网格：track_number 最大值 × sample_number 最大值；以及各频信号长度
        trk = [d for d in r["datasets"] if d["name"] == "spatial_data/track_number"]
        smp = [d for d in r["datasets"] if d["name"] == "spatial_data/sample_number"]
        if trk and smp:
            gk = (int(trk[0]["stats"]["max"]), int(smp[0]["stats"]["max"]))
            grid_info[gk] = grid_info.get(gk, 0) + 1
        for d in r["datasets"]:
            if d["name"].startswith("signal_data/") and d["name"].endswith("/real"):
                key = tuple(d["shape"])
                signal_len[key] = signal_len.get(key, 0) + 1

    # 输出：默认只打印紧凑摘要，完整 JSON 写文件
    summary = {
        "root": str(root),
        "n_files": agg["n_files"],
        "total_bytes": sum(r.get("size", 0) for r in agg["files"]),
        "dataset_paths": {
            k: {"shape_set": v["shape_set"], "dtype_set": v["dtype_set"], "files": v["files"]}
            for k, v in agg["dataset_paths"].items()
        },
        "errors": [r.get("error") for r in agg["files"] if "error" in r],
        "eddycus": {
            "n_unique_samples": len(samples),
            "samples": {k: {"n_files": v["n_files"], "material_type": v.get("meta", {}).get("material_type"),
                            "fiber_type": v.get("meta", {}).get("fiber_type"),
                            "layup": v.get("meta", {}).get("layup_sequence"),
                            "description": v.get("meta", {}).get("description"),
                            "defect_depth_mm": v.get("meta", {}).get("defect_depth_mm"),
                            "defect_size_mm": v.get("meta", {}).get("defect_size_mm"),
                            "sensor": v.get("sensor"), "datetime": v.get("datetime"),
                            "thickness_mm": v.get("meta", {}).get("thickness_mm")}
                        for k, v in samples.items()},
            "n_sensors": len(sensors),
            "sensors": sensors,
            "n_materials": len(materials),
            "materials": {f"{a}|{b}|{c}": n for (a, b, c), n in materials.items()},
            "n_defect_groups": len(defects),
            "defects": {f"{a}|depth={b}|size={c}": n for (a, b, c), n in defects.items()},
            "freq_sets": {f"({', '.join(x)})": n for x, n in freq_sets.items()},
            "grids": {f"tracks={a} x samples={b}": n for (a, b), n in grid_info.items()},
            "signal_len_per_freq": {str(k): n for k, n in signal_len.items()},
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if out_path:
        out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[full JSON -> {out_path}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=None, help="单文件结构探测")
    ap.add_argument("--full", action="store_true", help="全量聚合统计")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=REPO / "experiments/results/m0_2c_eddycus_audit.json")
    args = ap.parse_args()

    if args.probe:
        probe(args.probe)
    elif args.full:
        if not args.root.exists():
            print(f"[ERR] root 不存在: {args.root}", file=sys.stderr)
            sys.exit(2)
        full(args.root, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
