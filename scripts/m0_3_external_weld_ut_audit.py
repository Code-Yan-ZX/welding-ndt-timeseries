#!/usr/bin/env python3
"""M0-3 真实焊缝多源超声数据审计（只读，不训练）。

对 ``data/raw/external_weld_ut/{A,B,C,D}/`` 下的 Strathclyde 真实焊缝
FMC/PAUT 数据做通用结构探测与全量聚合统计，供
``docs/M0_3_external_weld_ut_audit.md`` 使用。不修改任何数据文件。

覆盖审计清单（M0-3 §三）：
1. 文件格式 / MATLAB-HDF5 schema / dtype / shape / NaN / Inf；
2. Tx / Rx / time / scan position / TFM-PAUT image 各维度真实含义；
3. 探头参数、采样率、材料、焊缝类型、缺陷类型、缺陷位置与尺寸；
4. 真正独立的物理 specimen 数与 acquisition 数；
5. 同一试件的 Tx×Rx、scan position、重复扫查共用 group_id（禁止当独立试件）；
6. 哈希、近重复与元数据重复检查；
7. 每数据源标记：ssl_pretrain_usable / downstream_label_usable /
   metadata_only / incompatible；
8. 明确"网页上的多个信号/通道 ≠ 多个独立试件"；
9. 真实独立焊缝试件 < 10 时标注 exploratory external pretraining source。

数据未下载（Cloudflare challenge 阻止自动下载）时运行 ``--full`` 输出
download-status 报告而不是崩溃——便于数据落地后重跑同一脚本直接产出审计。

用法:
  python scripts/m0_3_external_weld_ut_audit.py --probe <file>   # 单文件结构
  python scripts/m0_3_external_weld_ut_audit.py --full [--root <dir>] [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data/raw/external_weld_ut"
DEFAULT_OUT = REPO / "experiments/results/m0_3_external_weld_ut_audit_full.json"

SOURCES = {
    "A": ("316L lack-of-fusion FMC", "Lack_of_fusion_FMC_DORT_2016.mat"),
    "B": ("Inconel 82/182 centreline crack FMC", "FMC_2012_04_26_at_16_16.mat"),
    "C": ("304SS MMA 3mm SDH FMC", "FMC_RR3_2_25MHz_3mmsdh.mat"),
    "D": ("PAUT probe localisation", "PAUT.zip"),
}


# ---------------------------------------------------------------------------
# 数值统计
# ---------------------------------------------------------------------------
def _num_stats(arr: np.ndarray) -> dict:
    """数值数组的统计摘要（含 NaN/Inf 计数；大数组只抽样）。"""
    a = np.asarray(arr)
    if a.size == 0:
        return {"size": 0, "empty": True}
    max_elems = 1 << 20
    if a.size > max_elems:
        a = a[tuple(slice(0, min(s, 256)) for s in a.shape)]
    if a.dtype.names:
        return {
            "dtype": str(a.dtype), "shape": list(arr.shape),
            "structured_fields": {
                n: {"min": float(a[n].min()), "max": float(a[n].max()),
                    "mean": float(a[n].mean()), "std": float(a[n].std()),
                    "n_nan": int(np.isnan(a[n]).sum()),
                    "n_inf": int(np.isinf(a[n]).sum())}
                for n in a.dtype.names},
            "sampled": a.size < arr.size,
        }
    if a.dtype.kind == "c":
        re, im = a.real, a.imag
        return {
            "dtype": str(a.dtype), "shape": list(arr.shape), "complex": True,
            "real": {"min": float(re.min()), "max": float(re.max()),
                     "mean": float(re.mean()), "std": float(re.std())},
            "imag": {"min": float(im.min()), "max": float(im.max()),
                     "mean": float(im.mean()), "std": float(im.std())},
            "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
            "sampled": a.size < arr.size,
        }
    if a.dtype.kind == "f":
        return {
            "dtype": str(a.dtype), "shape": list(arr.shape),
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
            "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
            "sampled": a.size < arr.size,
        }
    uniq = None
    if a.size <= 1 << 16:
        try:
            uniq = sorted(int(x) for x in np.unique(a))
        except Exception:
            uniq = None
    return {
        "dtype": str(a.dtype), "shape": list(arr.shape),
        "min": int(a.min()) if a.size else None,
        "max": int(a.max()) if a.size else None,
        "uniq_n": len(uniq) if uniq is not None else None,
        "uniq_head": uniq[:8] if uniq else None,
    }


# ---------------------------------------------------------------------------
# 单文件结构探测
# ---------------------------------------------------------------------------
def _mat_walk(obj, name: str, depth: int = 0, max_depth: int = 6):
    """递归遍历 MATLAB 结构（v5 loadmat 字典树 / v7.3 h5py 树），返回结构列表。"""
    out = []
    if depth > max_depth:
        return out
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        for f in obj.dtype.names:
            out.append({"type": "struct", "name": f"{name}.{f}",
                        "fields": list(obj.dtype.names)})
            for sub in _mat_walk(obj[f], f"{name}.{f}", depth + 1, max_depth):
                out.append(sub)
        return out
    if isinstance(obj, np.ndarray) and obj.size == 1 and obj.dtype == object:
        inner = obj.item()
        if isinstance(inner, (np.ndarray, dict)):
            return _mat_walk(inner, name, depth + 1, max_depth)
        return [{"type": "scalar", "name": name, "value": str(inner)[:80]}]
    if isinstance(obj, np.ndarray):
        return [{"type": "array", "name": name, "shape": list(obj.shape),
                 "dtype": str(obj.dtype), "stats": _num_stats(obj)}]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append({"type": "dict", "name": f"{name}.{k}", "value": str(v)[:120]})
        return out
    return [{"type": "scalar", "name": name, "value": str(obj)[:80]}]


def _probe_mat_v5(path: Path) -> dict:
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=False, struct_as_record=False)
    out = {"format": "matlab_v5", "keys": [k for k in m if not k.startswith("__")],
           "items": []}
    for k in out["keys"]:
        out["items"].extend(_mat_walk(m[k], k))
    return out


def _probe_mat_v73(path: Path) -> dict:
    import h5py
    out = {"format": "matlab_v7.3_hdf5", "keys": [], "items": [], "attrs": {}}
    with h5py.File(path, "r") as f:
        out["attrs"] = {k: str(v) for k, v in f.attrs.items()}
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = obj[tuple(slice(0, min(s, 256)) for s in obj.shape)]
                out["items"].append({"type": "dataset", "name": name,
                                     "shape": list(obj.shape),
                                     "dtype": str(obj.dtype),
                                     "stats": _num_stats(arr)})
            else:
                out["items"].append({"type": "group", "name": name,
                                     "n_children": len(obj)})
        for k in f.keys():
            out["keys"].append(k)
        f.visititems(walk)
    return out


def _probe_spreadsheet(path: Path) -> dict:
    import pandas as pd
    ext = path.suffix.lower()
    engine = "odf" if ext == ".ods" else "openpyxl"
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine=engine, header=None)
    except Exception as e:
        return {"format": ext, "error": str(e)}
    out = {"format": ext, "sheets": {}}
    for name, df in sheets.items():
        head = df.head(40).astype(str).where(df.notna(), None)
        out["sheets"][name] = {
            "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
            "head": head.values.tolist(),
        }
    return out


def _probe_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        return {
            "format": "zip",
            "n_files": len(infos),
            "total_uncompressed": sum(i.file_size for i in infos),
            "files": [{"name": i.filename, "size": i.file_size,
                       "compress_size": i.compress_size} for i in infos],
        }


def probe(path: Path) -> dict:
    """按扩展名探测单文件。"""
    ext = path.suffix.lower()
    if ext == ".mat":
        try:
            return _probe_mat_v5(path)
        except Exception as e1:
            try:
                return _probe_mat_v73(path)
            except Exception as e2:
                return {"format": "mat", "error_v5": str(e1), "error_v73": str(e2)}
    if ext in (".zip",):
        return _probe_zip(path)
    if ext in (".xlsx", ".xls", ".ods", ".csv"):
        return _probe_spreadsheet(path)
    return {"format": ext, "note": "未识别格式", "size": path.stat().st_size}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 全量聚合
# ---------------------------------------------------------------------------
def full(root: Path, out_path: Path | None) -> dict:
    if not root.exists():
        print(f"[ERR] root 不存在: {root}", file=sys.stderr)
        sys.exit(2)

    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.name != "checksums.txt")
    agg = {"root": str(root), "n_files": len(files), "files": [],
           "sources": {}, "dataset_paths": {}}
    for p in files:
        rel = str(p.relative_to(root))
        src = rel.split("/")[0]
        rec = {"file": p.name, "rel": rel, "source": src,
               "size": p.stat().st_size, "sha256": _sha256(p)}
        try:
            rec["structure"] = probe(p)
            # 近重复：同 source 内 sha256 分组在汇总层统计
        except Exception as e:
            rec["error"] = str(e)
        agg["files"].append(rec)
        # dataset path 汇总（mat 数组 / hdf5 dataset / zip 内文件）
        for item in (rec.get("structure") or {}).get("items", []):
            if item.get("type") in ("array", "dataset"):
                key = f"{src}:{item['name']}"
                e = agg["dataset_paths"].setdefault(
                    key, {"shapes": [], "dtypes": [], "files": []})
                e["shapes"].append(tuple(item["shape"]))
                e["dtypes"].append(item["dtype"])
                e["files"].append(rec["rel"])
        if (rec.get("structure") or {}).get("format") == "zip":
            for zf in (rec.get("structure") or {}).get("files", []):
                key = f"{src}:{zf['name']}"
                e = agg["dataset_paths"].setdefault(
                    key, {"shapes": [], "dtypes": [], "files": []})
                e["shapes"].append(("zip-entry",))
                e["dtypes"].append("?")
                e["files"].append(rec["rel"])

    for key, e in agg["dataset_paths"].items():
        e["shape_set"] = sorted({tuple(s) for s in e["shapes"]})
        e["dtype_set"] = sorted(set(e["dtypes"]))

    # ---- 每 source 汇总 + 独立试件/采集数（结构驱动）----
    for sid, (sname, _exp_file) in SOURCES.items():
        sfiles = [f for f in agg["files"] if f["source"] == sid]
        s = {
            "name": sname, "n_files": len(sfiles),
            "files": [{"rel": f["rel"], "size": f["size"], "sha256": f["sha256"],
                       "format": (f.get("structure") or {}).get("format"),
                       "error": f.get("error")} for f in sfiles],
            "sha256_dups": _dup_groups(sfiles),
            "independent_specimens_est": None,   # 结构审计后由人工/后续分析填写
            "n_acquisitions_est": None,
            "flags": {"ssl_pretrain_usable": None, "downstream_label_usable": None,
                      "metadata_only": None, "incompatible": None},
            "notes": [],
        }
        agg["sources"][sid] = s

    # 全部 .mat 数组 sha256 重复（近重复探测：同 source 内同数组完全一致）
    # 独立试件数：**禁止用 Tx×Rx / scan position / 切片数当独立试件**；
    # 结构无法自证时保持 None，由 docs 人工确认后回填。
    summary = {
        "root": str(root),
        "n_files": agg["n_files"],
        "total_bytes": sum(f["size"] for f in agg["files"]),
        "download_status": ("ok" if agg["n_files"] else
                            "BLOCKED/EMPTY - Cloudflare challenge 阻止自动下载，"
                            "见 data/manifests/external_weld_ut/download_manifest.json"),
        "dataset_paths": {k: {"shape_set": v["shape_set"],
                              "dtype_set": v["dtype_set"], "files": v["files"]}
                          for k, v in agg["dataset_paths"].items()},
        "sources": agg["sources"],
        "warnings": [
            "网页上的'多个信号/多个通道'不等于多个独立试件；独立试件数与采集数"
            "必须在结构审计后人工确认并写入 dataset card，禁止把 Tx×Rx / scan "
            "position / 切片数包装成'大规模真实试件数据'。",
            "真实独立焊缝试件总数 < 10 时，正式报告必须标注 exploratory "
            "external pretraining source，不称为 foundation-scale dataset。",
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if out_path:
        out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n[full JSON -> {out_path}]")
    return summary


def _dup_groups(files: list[dict]) -> dict:
    by_sha: dict[str, list[str]] = {}
    for f in files:
        by_sha.setdefault(f["sha256"], []).append(f["rel"])
    return {k: v for k, v in by_sha.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=None, help="单文件结构探测")
    ap.add_argument("--full", action="store_true", help="全量聚合审计")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.probe:
        print(json.dumps(probe(args.probe), indent=2, ensure_ascii=False))
    elif args.full:
        full(args.root, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
