#!/usr/bin/env python3
"""Pilot-B1: EddyCus-HDF5 数据结构审计（ndt_public_benchmark_pilot）。

复用 docs/general_ndt_foundation/phase2_eddycus_admission.md 与
scripts/audit_eddycus_hierarchy.py 的既有结论，落盘一份结构化审计表 + 汇总统计：

- experiments/results/ndt_pilot/eddycus_scan_table.csv   每扫描一行（分组/划分的 ground truth）
- experiments/results/ndt_pilot/eddycus_audit.json       汇总统计（B1 问题清单逐项回答）

要点（沿用既有审计，禁止推翻）：
- 738 文件 = 738 次扫描，sample_properties.id 是扫描序号不是试件号；
- 无显式试件 ID；specimen 代理 = (material,fiber,layup,description,defect_depth,
  defect_size,thickness) 配置组哈希（148 组，含标签）；
- sensor_type 归一化前有格式化污染（真实 7 个传感器）；
- 标签粒度 = scan 级（attrs description），无 defect mask / bbox / 坐标。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.eddycus import normalize_sensor  # noqa: E402
from wndt.data.adapters.eddycus import EddyCusAdapter, classify  # noqa: E402

OUT_DIR = REPO / "experiments" / "results" / "ndt_pilot"
H5_ROOT = REPO / "data" / "raw" / "EddyCus-HDF5" / "output"


def build_scan_table() -> pd.DataFrame:
    ad = EddyCusAdapter()
    recs = ad.records()
    rows = []
    for i, r in enumerate(recs):
        p = Path(r.tensor_path)
        import h5py

        with h5py.File(p, "r") as f:
            has_signal = "signal_data" in f and "f1" in f["signal_data"]
            grid_h = grid_w = n_points = None
            n_freq_present = 0
            freqs_mhz = []
            if "measurement_metadata" in f and "frequencies" in f["measurement_metadata"]:
                fq = f["measurement_metadata"]["frequencies"]
                for k in sorted(fq.keys()):
                    if isinstance(fq[k], h5py.Group):
                        v = fq[k].attrs.get("frequency_mhz")
                        if v:
                            freqs_mhz.append(float(v))
            if has_signal:
                sp = f["spatial_data"]
                trk = np.asarray(sp["track_number"][()])
                smp = np.asarray(sp["sample_number"][()])
                grid_h, grid_w = int(trk.max()), int(smp.max())
                n_points = int(trk.size)
                n_freq_present = sum(
                    1 for fk in ("f1", "f2", "f3", "f4") if f"signal_data/{fk}" in f["signal_data"]
                )
            mm = dict(f["measurement_metadata"].attrs) if "measurement_metadata" in f else {}
        d = r.domain
        desc = d.get("description", "")
        dtype_, is_def = classify(desc)
        sensor_raw = r.geometry.get("sensor_type", "")
        rows.append(
            {
                "scan_id": r.record_id.replace("eddycus:scan_", "scan_"),
                "file": p.name,
                "has_signal": has_signal,
                "sensor_raw": sensor_raw,
                "sensor": normalize_sensor(sensor_raw),
                "material": d.get("material_type", ""),
                "fiber": d.get("fiber_type", ""),
                "layup": d.get("layup_sequence", ""),
                "thickness_mm": d.get("thickness_mm", ""),
                "description": desc,
                "defect_type": dtype_,
                "defect_present": is_def,
                "defect_depth_mm": d.get("defect_depth_mm", ""),
                "defect_size_mm": d.get("defect_size_mm", ""),
                "specimen_cfg": r.specimen_id,       # 推断配置组（含标签）
                "defect_group": r.defect_instance_id or f"clean:{r.specimen_id}",
                "frequencies_mhz": ";".join(f"{x:g}" for x in freqs_mhz),
                "n_freq_present": n_freq_present,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "n_points": n_points,
                "datetime": str(mm.get("measurement_datetime", "")),
                "sensor_orientation_degree": d.get("sensor_orientation_degree", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    sig = df[df.has_signal]
    return {
        "n_files": len(df),
        "n_with_signal": int(df.has_signal.sum()),
        "n_metadata_only": int((~df.has_signal).sum()),
        "n_sensors_normalized": int(df.sensor.nunique()),
        "sensor_counts": df.sensor.value_counts().to_dict(),
        "n_materials": int(df.material.nunique()),
        "material_counts": df.material.value_counts().to_dict(),
        "n_specimen_cfg_groups": int(df.specimen_cfg.nunique()),
        "n_defect_groups": int(df[df.defect_present].defect_group.nunique()),
        "n_clean_cfg_groups": int(df[~df.defect_present].defect_group.nunique()),
        "defect_type_counts": df.defect_type.value_counts().to_dict(),
        "defect_present_counts": {str(k): int(v) for k, v in df.defect_present.value_counts().items()},
        "defect_present_with_signal": {str(k): int(v) for k, v in sig.defect_present.value_counts().items()},
        "frequency_combo_counts": {k: int(v) for k, v in df.frequencies_mhz.value_counts().items()},
        "grid_size_counts": {
            f"{h}x{w}": int(c) for (h, w), c in Counter(zip(sig.grid_h, sig.grid_w)).most_common(12)
        },
        "scans_per_specimen_cfg_hist": {str(k): int(v) for k, v in
                                        df.groupby("specimen_cfg").size().value_counts().sort_index().items()},
        # 同一物理 defect 是否被多 sensor / 多频率重复扫描：
        "defect_groups_multi_sensor": int(sum(
            1 for _, g in df[df.defect_present].groupby("defect_group") if g.sensor.nunique() > 1)),
        "specimen_cfg_multi_sensor": int(sum(
            1 for _, g in df.groupby("specimen_cfg") if g.sensor.nunique() > 1)),
        "specimen_cfg_multi_datetime": int(sum(
            1 for _, g in df.groupby("specimen_cfg") if g.datetime.nunique() > 1)),
        # label 粒度
        "label_granularity": "scan-level (HDF5 attrs description); 无 defect mask/bbox/坐标",
        "grid_size_uniform": bool(sig[["grid_h", "grid_w"]].drop_duplicates().shape[0] == 1),
        # 有信号文件中 defect/clean 每传感器分布（sensor-held-out 可行性）
        "per_sensor_defect_clean_with_signal": {
            s: {"defect": int((g.defect_present).sum()), "clean": int((~g.defect_present).sum())}
            for s, g in sig.groupby("sensor")
        },
        "per_material_defect_clean_with_signal": {
            m: {"defect": int((g.defect_present).sum()), "clean": int((~g.defect_present).sum())}
            for m, g in sig.groupby("material")
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_scan_table()
    csv_path = OUT_DIR / "eddycus_scan_table.csv"
    df.to_csv(csv_path, index=False)
    stats = summarize(df)
    stats["scan_table"] = str(csv_path.relative_to(REPO))
    (OUT_DIR / "eddycus_audit.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
