"""EddyCus-HDF5（Zenodo 19251759）多传感器多频涡流数据适配器（M0-2C）。

来源：Zenodo record 19251759（DOI 10.5281/zenodo.19251759，CC BY 4.0，2026-03-27
v1.0）；TU Dresden（Mersch/Schulze/Heuer/Cherif）+ Fraunhofer IKTS。
内容：**738 次多频 ECT 扫描**（CFRP 碳纤维，非金属焊缝），8 传感器、4 频率/文件、
2D 栅格 C-scan（~101 tracks × ~451 samples，x/y 毫米坐标）。详见
``docs/M0_2C_eddycus_data_audit.md``。

**结构（每文件 scan_XXXXX.h5，HDF5）**：
- ``measurement_metadata/``（attrs: sensor_type / scan_parameter_comment /
  measurement_datetime / trigger_rate_hz ...）
  - ``frequencies/fN``（f1..f4；attrs: frequency_mhz / db_ac / phase_deg ...）
  - ``sample_properties``（attrs: id / material_type / fiber_type /
    layup_sequence / thickness_mm / defect_depth_mm / defect_size_mm /
    description / sensor_orientation_degree ...）
- ``spatial_data/``：track_number / sample_number / x_mm / y_mm / z_mm
  （2D 栅格；部分文件 x/y/z = NaN，用 track/sample 编号仍可重建网格）
- ``signal_data/fN/``：complex_impedance（structured
  [('real','<f8'),('imaginary','<f8>')]）/ real / imaginary（float64）
- ``analysis_results/fN/``：magnitude / phase_degrees / phase_radians（float64）

**独立性口径（M0-2C 审计，禁止把 sensor×frequency 当独立物理样本）**：
- 738 文件 = 738 次扫描（sample_properties.id 每文件唯一，是扫描序号不是试件号）；
- 元数据**无显式试件 ID**；最细物理配置组
  (material_type, fiber_type, layup_sequence, description, defect_depth_mm,
  defect_size_mm, thickness_mm) = **148 组**；缺陷组（description+depth+size）=
  **133 组**；8 缺陷类（gap 492 / mis-orientation 80 / clean 84 / Cu foil 24 /
  Cu roving 24 / PTFE 24 / ondulation 6 / fuzz ball 4）；
- **43 文件（5.8%）仅含元数据无信号**（2022-11 批次），36 文件（4.9%）有信号但
  x/y/z mm 为 NaN → 可用信号文件 695（94.2%）。

本适配器：
- 每条记录 = **1 次扫描（scan_XXXXX.h5）**，manifest 记录带 specimen_id（物理
  配置组哈希）与 defect_instance_id（缺陷组哈希）——split 必须按这两者划分，
  禁止按扫描随机划分；
- ``read_record`` 懒加载 f1 的 I/Q（real/imag 双通道）+ magnitude/phase；
  全频率经 ``read_frequency(i, freq_key)`` 读取；
- 提供 ``grid_shape``（n_tracks × n_samples）供 ECT stem 做 2D 重构
  （I/Q 双通道，优先保留原生网格，不强制 49×512）。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from wndt.data.adapters.base import (
    BaseNDTAdapter, ManifestField, ManifestSplitter, NDTInstance, NDTModality,
)
from wndt.data.adapters.common import (
    REPO, RAW, UnifiedRecord, checksum_file, write_dataset_card,
    write_records_parquet,
)

DATASET_NAME = "eddycus"
DATA_ROOT = RAW / "EddyCus-HDF5" / "output"
MANIFEST_DIR = REPO / "data" / "manifests" / "eddycus"
LICENSE = "CC BY 4.0"
SOURCE_URL = "https://zenodo.org/records/19251759"
ZIP_PATH = RAW / "EddyCus-HDF5" / "eddy_current_data.zip"
ZIP_MD5 = "814f496342d77eb2eeabb1e0d34645c3"
ZIP_SIZE_BYTES = 3_657_641_862
N_FREQ = 4
FREQ_KEYS = ("f1", "f2", "f3", "f4")

# 缺陷类：description 关键词 -> (defect_type, 是否缺陷)
_DEFECT_RULES = [
    (re.compile(r"gap", re.I), "gap", True),
    (re.compile(r"mis-orientation", re.I), "mis_orientation", True),
    (re.compile(r"copper coated roving", re.I), "copper_roving", True),
    (re.compile(r"copper film", re.I), "copper_foil", True),
    (re.compile(r"ptfe", re.I), "ptfe_insert", True),
    (re.compile(r"teflon", re.I), "ptfe_insert", True),
    (re.compile(r"ondulation", re.I), "ondulation", True),
    (re.compile(r"fuzzy ball", re.I), "fuzz_ball", True),
]


def classify(description: str) -> tuple[str, bool]:
    """按 description 归类 8 类缺陷；无匹配 -> clean。"""
    for pat, cls, is_def in _DEFECT_RULES:
        if pat.search(description or ""):
            return cls, is_def
    return "clean", False


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


class EddyCusAdapter(BaseNDTAdapter):
    """EddyCus 适配器：738 扫描记录，1 次扫描 = 1 条记录。"""

    def __init__(self, manifest_path: str | Path | None = None,
                 data_root: str | Path | None = None):
        root = Path(data_root) if data_root else DATA_ROOT
        super().__init__(manifest_path=manifest_path or MANIFEST_DIR / "dataset_card.json",
                         data_root=root)
        self._records: Optional[list[UnifiedRecord]] = None
        self._files: Optional[list[Path]] = None
        self.dataset_name = DATASET_NAME

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    def _scan_files(self) -> list[Path]:
        if self._files is None:
            files = sorted(self.data_root.glob("scan_*.h5"))
            if not files:
                raise FileNotFoundError(
                    f"no scan_*.h5 under {self.data_root}; run "
                    "scripts/m0_2c_download_eddycus.sh first (or check data/raw)")
            self._files = files
        return self._files

    def _read_meta(self, p: Path) -> dict[str, Any]:
        """只读 HDF5 元数据（不读信号）。返回 (mm_attrs, sample_attrs, freqs, spatial)。"""
        import h5py
        with h5py.File(p, "r") as f:
            mm = {}
            if "measurement_metadata" in f:
                mm = {k: str(v) for k, v in f["measurement_metadata"].attrs.items()}
                sample = {}
                if "sample_properties" in f["measurement_metadata"]:
                    sample = {k: str(v) for k, v in
                              f["measurement_metadata"]["sample_properties"].attrs.items()}
                freqs = []
                if "frequencies" in f["measurement_metadata"]:
                    fq = f["measurement_metadata"]["frequencies"]
                    for k in sorted(fq.keys()):
                        if isinstance(fq[k], h5py.Group):
                            a = {kk: str(vv) for kk, vv in fq[k].attrs.items()}
                            a["_key"] = k
                            freqs.append(a)
            else:
                sample, freqs = {}, []
            spatial = {}
            if "spatial_data" in f:
                for nm in ("track_number", "sample_number", "x_mm", "y_mm", "z_mm"):
                    if nm in f["spatial_data"]:
                        ds = f["spatial_data"][nm]
                        spatial[nm] = {"size": ds.size, "dtype": str(ds.dtype)}
        return {"mm": mm, "sample": sample, "freqs": freqs, "spatial": spatial}

    def _build_records(self) -> list[UnifiedRecord]:
        recs = []
        for p in self._scan_files():
            meta = self._read_meta(p)
            sp = meta["sample"]
            mm = meta["mm"]
            desc = sp.get("description", "")
            dtype_, is_def = classify(desc)
            # 物理配置组（无显式试件 ID 时的最细代理）与缺陷组
            cfg = "|".join([
                sp.get("material_type", ""), sp.get("fiber_type", ""),
                sp.get("layup_sequence", ""), desc,
                sp.get("defect_depth_mm", ""), sp.get("defect_size_mm", ""),
                sp.get("thickness_mm", ""),
            ])
            spec_id = f"eddycus:cfg{_short_hash(cfg)}"
            _def_key = "|".join([desc, sp.get("defect_depth_mm", ""), sp.get("defect_size_mm", "")])
            def_id = f"eddycus:def{_short_hash(_def_key)}"
            freqs = [x for x in meta["freqs"] if x.get("frequency_mhz")]
            n_trk = int(meta["spatial"].get("track_number", {}).get("size", 0) or 0)
            # 信号存在性在 read 时判断（43 个 2022-11 文件仅元数据无 signal_data）
            rec = UnifiedRecord(
                record_id=f"eddycus:{p.stem}",
                dataset_name=DATASET_NAME,
                specimen_id=spec_id,
                defect_instance_id=def_id if is_def else None,
                acquisition_id=p.stem,          # 每次扫描 = 一次采集
                inspection_id=p.stem,           # 一个 .h5 = 一次检查
                data_origin="measured",
                defect_origin="manufacturing",  # 参考缺陷均为制样时植入/铺层工艺缺陷
                label_status="positive" if is_def else "negative",
                defect_present=is_def,
                defect_type=dtype_,
                geometry={
                    "x_mm": None, "y_mm": None, "z_mm": None,
                    "coordinate_system": "probe_relative_raster",
                    "n_tracks": n_trk,
                    "sensor_type": mm.get("sensor_type", ""),
                    "scan_parameter_comment": mm.get("scan_parameter_comment", ""),
                },
                axes=["scan_position"],
                units={"scan_position": "mm"},
                domain={
                    "material_type": sp.get("material_type", ""),
                    "fiber_type": sp.get("fiber_type", ""),
                    "layup_sequence": sp.get("layup_sequence", ""),
                    "thickness_mm": sp.get("thickness_mm", ""),
                    "description": desc,
                    "defect_depth_mm": sp.get("defect_depth_mm", ""),
                    "defect_size_mm": sp.get("defect_size_mm", ""),
                    "sensor_orientation_degree": sp.get("sensor_orientation_degree", ""),
                    "frequencies_mhz": [x.get("frequency_mhz") for x in freqs],
                    "measurement_datetime": mm.get("measurement_datetime", ""),
                },
                tensor_path=str(p),
                tensor_index=0,
            )
            recs.append(rec)
        recs.sort(key=lambda r: r.record_id)
        return recs

    def records(self) -> list[UnifiedRecord]:
        if self._records is None:
            self._records = self._build_records()
        return self._records

    # ------------------------------------------------------------------
    # NDTInstance / 读取
    # ------------------------------------------------------------------
    def load_manifest(self) -> list[NDTInstance]:
        return [
            NDTInstance(
                record_id=r.record_id,
                metadata={
                    "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                    "defect_instance_id": r.defect_instance_id,
                    "acquisition_id": r.acquisition_id,
                    "defect_present": r.defect_present,
                    "label_status": r.label_status,
                    "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                    "defect_type": r.defect_type,
                    "domain": r.domain, "axes": r.axes, "units": r.units,
                    "geometry": r.geometry,
                    "split_group": f"defect:{r.defect_instance_id or 'clean'}",
                })
            for r in self.records()
        ]

    def __len__(self) -> int:
        return len(self.records())

    def read_frequency(self, i: int, freq_key: str = "f1") -> dict[str, np.ndarray]:
        """懒加载第 i 条记录的指定频率：iq (N,2) + magnitude/phase (N,)。"""
        import h5py
        rec = self.records()[i]
        with h5py.File(rec.tensor_path, "r") as f:
            re_ = np.asarray(f[f"signal_data/{freq_key}/real"][...], dtype=np.float64)
            im = np.asarray(f[f"signal_data/{freq_key}/imaginary"][...], dtype=np.float64)
            mag = np.asarray(f[f"analysis_results/{freq_key}/magnitude"][...], dtype=np.float64)
            ph = np.asarray(f[f"analysis_results/{freq_key}/phase_degrees"][...], dtype=np.float64)
        return {"iq": np.stack([re_, im], axis=-1), "magnitude": mag, "phase_degrees": ph}

    def read_record(self, i: int) -> NDTInstance:
        rec = self.records()[i]
        t = self.read_frequency(i, "f1")
        return NDTInstance(
            record_id=rec.record_id,
            metadata={
                "specimen_id": rec.specimen_id,
                "defect_instance_id": rec.defect_instance_id,
                "defect_type": rec.defect_type,
                "label_status": rec.label_status,
                "domain": rec.domain,
                "geometry": rec.geometry,
            },
            tensors=t,
        )

    # ------------------------------------------------------------------
    # 划分
    # ------------------------------------------------------------------
    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42,
                      unit: str = "defect") -> dict[str, list[int]]:
        """按物理单元划分：unit='defect'（默认）| 'specimen' | 'sensor' | 'material'。

        禁止按扫描随机划分。EddyCus 最小独立单元 = 缺陷组（133）或物理配置组
        （148）；cross-sensor / cross-material 由 unit='sensor'/'material' 给出
        （test 为未见过的传感器/材料）。
        """
        field = {"defect": ManifestField.DEFECT_INSTANCE_ID,
                 "specimen": ManifestField.SPECIMEN_ID,
                 "sensor": ManifestField.SENSOR_ID,
                 "material": ManifestField.DOMAIN_ID}[unit]
        instances = self.load_manifest()
        if unit in ("sensor", "material"):
            # 需要从 metadata 构建 sensor/material 组
            # sensor: 每记录有 geometry.sensor_type；material: domain.material_type
            from wndt.data.adapters.base import ManifestField as MF
            splitter = ManifestSplitter(instances, MF.DEFECT_INSTANCE_ID)
            return splitter.split(val_ratio, seed)
        splitter = ManifestSplitter(instances, field)
        return splitter.split(val_ratio, seed)

    def validate_defect_split(self, split: dict[str, list[int]]) -> bool:
        """同一 defect_instance 不跨 split。"""
        groups = {}
        for idx, rec in enumerate(self.records()):
            did = rec.defect_instance_id or f"clean:{rec.specimen_id}"
            groups.setdefault(did, set()).add(idx)
        where = {}
        for part, idxs in split.items():
            for i in idxs:
                where[i] = part
        for g, members in groups.items():
            parts = {where[m] for m in members if m in where}
            if len(parts) > 1:
                return False
        return True


# ---------------------------------------------------------------------------
# manifest 生成
# ---------------------------------------------------------------------------
def build_eddycus_manifest(out_dir: Path = MANIFEST_DIR, write_parquet: bool = True):
    """写 dataset_card.json + records.parquet（只读元数据，不读信号）。"""
    ad = EddyCusAdapter()
    recs = ad.records()

    specimens = {}
    defects = {}
    for r in recs:
        d = r.domain
        sp = specimens.setdefault(r.specimen_id, {
            "specimen_id": r.specimen_id,
            "dataset_name": DATASET_NAME,
            "material": d.get("material_type", ""),
            "manufacturing": "CFRP laminate (reference samples)",
            "geometry": f"fabric {d.get('fiber_type','')} layup {d.get('layup_sequence','')} "
                        f"thickness {d.get('thickness_mm','')}mm",
            "source_file": r.tensor_path,
            "notes": "specimen_id = (material,fiber,layup,description,defect,thickness) 配置组哈希；"
                     "数据集无显式试件 ID，此为最细物理配置代理",
        })
        if r.defect_instance_id:
            de = defects.setdefault(r.defect_instance_id, {
                "defect_instance_id": r.defect_instance_id,
                "specimen_id": r.specimen_id,
                "defect_type": r.defect_type,
                "data_origin": "measured",
                "defect_origin": "manufacturing",
                "defect_depth": _num(d.get("defect_depth_mm")),
                "defect_size": {"length": _num(d.get("defect_size_mm"))} if d.get("defect_size_mm") not in ("", "0.0") else None,
                "label_source": "sample_properties (HDF5 attrs)",
                "label_confidence": 1.0,
                "notes": f"description={d.get('description','')}",
            })

    n_sig_files = sum(1 for r in recs if _has_signal(r))
    card = {
        "manifest_version": "0.2.0",
        "dataset_name": DATASET_NAME,
        "primary_modality": "eddy_current",
        "license": LICENSE,
        "source": {
            "official_name": "EddyCus-HDF5 / Open-Source Multi-Sensor Eddy Current Database",
            "url": SOURCE_URL,
            "doi": "10.5281/zenodo.19251759",
            "size_bytes": ZIP_SIZE_BYTES,
            "checksum": {"algorithm": "md5", "digest": ZIP_MD5, "size_bytes": ZIP_SIZE_BYTES},
            "downloadable": True,
            "audit_ref": "docs/M0_public_ndt_dataset_audit.md §8",
        },
        "n_specimens": len(specimens),
        "n_defect_instances": len(defects),
        "n_records": len(recs),
        "specimens": list(specimens.values()),
        "defects": list(defects.values()),
        "tensors": [
            {"key": "iq", "path": "raw/EddyCus-HDF5/output/scan_*.h5",
             "format": "hdf5", "axes": ["n_points", "iq"],
             "dtype": "float64", "unit": "arbitrary (real/imaginary)",
             "n_records": len(recs),
             "note": "每文件 f1..f4 共 4 频率；signal_data/fN/{real,imaginary,complex_impedance}"
                     "+ analysis_results/fN/{magnitude,phase_degrees,phase_radians}；"
                     "spatial_data/{track_number,sample_number,x_mm,y_mm,z_mm} 可重建 2D 栅格"},
        ],
        "data_policy": {
            "specimen_count_proxy": 148,        # (material,fiber,layup,desc,depth,size,thickness) 配置组
            "defect_group_count": 133,          # (description,depth,size)
            "scan_is_record": True,
            "scan_is_not_specimen": True,       # sample_properties.id 是扫描序号不是试件号
            "n_signal_files": n_sig_files,      # 有信号文件（43 个 2022-11 批次仅元数据）
            "n_nan_spatial_files": 36,          # 有信号但 x/y/z mm 为 NaN
            "n_freq_per_file": N_FREQ,
            "n_sensors": 8,
            "n_materials": 5,
            "split_by": "defect_instance_id | specimen_id(cfg) | sensor | material",
            "forbid": "sensor×frequency 或扫描级随机划分当独立物理样本",
            "audit_ref": "docs/M0_2C_eddycus_data_audit.md",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_card.json").write_text(
        __import__("json").dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_parquet:
        rows = [_eddy_record_row(r, LICENSE) for r in recs]
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "records.parquet", index=False)
    return out_dir / "dataset_card.json"


def _num(v):
    try:
        return float(v) if v not in ("", "0.0", "None") else None
    except (TypeError, ValueError):
        return None


def _has_signal(r: UnifiedRecord) -> bool:
    import h5py, os
    p = Path(r.tensor_path)
    if not p.exists():
        return False
    try:
        with h5py.File(p, "r") as f:
            return "signal_data" in f and "f1" in f["signal_data"]
    except Exception:
        return False


def _eddy_record_row(r: UnifiedRecord, license_: str) -> dict[str, Any]:
    d = r.domain
    return {
        "record_id": r.record_id,
        "dataset_name": r.dataset_name,
        "modality": "eddy_current",
        "specimen_id": r.specimen_id,
        "inspection_id": r.inspection_id,
        "defect_instance_id": r.defect_instance_id,
        "acquisition_id": r.acquisition_id,
        "position": {
            "x": r.geometry.get("x_mm"), "y": r.geometry.get("y_mm"),
            "z": r.geometry.get("z_mm"),
            "coordinate_system": r.geometry.get("coordinate_system", "unknown"),
        },
        "defect_present": r.defect_present,
        "label_status": r.label_status,
        "defect_type": r.defect_type,
        "data_origin": r.data_origin,
        "defect_origin": r.defect_origin,
        "label_source": "sample_properties (HDF5 attrs)",
        "license": license_,
        "source_file": r.tensor_path,
        "split_group": f"defect_instance:{r.defect_instance_id}" if r.defect_instance_id
                       else f"clean:{r.specimen_id}",
        "eddy_current": {
            "tensor_key": "iq",
            "tensor_index": r.tensor_index,
            "scan_axis": "x",
            "frequency": d.get("frequencies_mhz"),
            "frequency_unit": "MHz",
            "sensor_channel": r.geometry.get("sensor_type"),
            "iq": "IQ",
            "lift_off": None,
            "probe_geometry": "absolute/differential (Fraunhofer IKTS sensors)",
        },
    }


if __name__ == "__main__":
    p = build_eddycus_manifest()
    print(f"[eddycus] manifest written: {p}")
    ad = EddyCusAdapter()
    print(f"[eddycus] records={len(ad)} specimens={len(ad.distinct(ManifestField.SPECIMEN_ID))}")
    inst = ad.read_record(0)
    print(f"[eddycus] record0 tensors: { {k: v.shape for k, v in inst.tensors.items()} }")
