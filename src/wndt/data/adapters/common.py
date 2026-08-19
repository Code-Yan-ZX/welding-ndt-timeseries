"""M0-2A 统一数据读取层：流式读取 + 统一记录结构 + manifest 生成工具。

三个公开超声数据集（PENELOPE / ML-NDT / NDT_ML_Flaw）的 adapter 都基于
本模块：

1. ``StreamingNpyReader`` / ``StreamingRawReader``：内存映射 / 随机偏移读取，
   任何时刻只把**被读取的那条记录**载入内存，绝不把整份 tensor 一次展开。
2. ``UnifiedRecord``：adapter 统一输出结构 —— tensor + 一组跨数据集字段
   （dataset_name / specimen_id / defect_instance_id / acquisition_id /
   data_origin / defect_origin / label_status / source geometry /
   axes/units / domain metadata）。不强制把三种数据插值成同一二维图片。
3. ``write_dataset_card`` / ``write_records_parquet``：按 M0-1.5 schema
   （data/manifests/templates/ndt_manifest_schema.json）生成顶层 dataset card
   JSON 与大规模 records.parquet。

约定：
- 原始数据一律放 ``data/raw/`` 且不纳入 git（见 .gitignore）。
- manifest（dataset_card.json + records.parquet）放 ``data/manifests/<ds>/``，
  纳入 git（仅元数据，不含原始信号）。
- tensor 路径以 ``data/processed/`` 或 ``data/raw/`` 为基准的字符串存于
  records 的 source_file / tensor 引用中，读取时再解析为绝对路径。
"""
from __future__ import annotations

import hashlib
import json
import mmap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]          # repo root
MANIFESTS = REPO / "data" / "manifests"
PROCESSED = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"


def manifest_dir_for(dataset_name: str) -> Path:
    """dataset_name -> manifest 目录（PENELOPE 的 card 目录为 penelope）。"""
    aliases = {"penelope_paut": "penelope"}
    return MANIFESTS / aliases.get(dataset_name, dataset_name)


# ---------------------------------------------------------------------------
# 流式读取
# ---------------------------------------------------------------------------
class StreamingNpyReader:
    """基于内存映射 (mmap) 的 .npy 按行读取器。

    ``shape`` / ``dtype`` 在构造时从文件头读出（不加载数据）；``read(i)``
    只把第 ``i`` 行切出来。适用于大数组如 PAUT (N, beam, time)、
    ML-NDT 体积、NDT_ML_Flaw 条带，避免 ``np.load`` 整份进内存。

    注意：npy 的 C 连续行读取是最优路径；若 axis0 上逐行随机访问频繁，
    这是最优做法（行内连续）。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            self._data_offset = f.tell()
        self._shape = shape
        self._dtype = dtype
        self._fortran = fortran
        self._mm = None
        self._fp = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def n_records(self) -> int:
        return int(self._shape[0])

    def _open(self):
        if self._mm is None:
            self._fp = open(self.path, "rb")
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def read(self, index: int) -> np.ndarray:
        """返回第 ``index`` 条记录（axis0 一行）的新数组。"""
        if index < 0 or index >= self.n_records:
            raise IndexError(f"record index {index} out of range [0, {self.n_records})")
        self._open()
        row_bytes = int(np.prod(self._shape[1:], dtype=np.int64)) * self._dtype.itemsize
        start = self._data_offset + index * row_bytes
        buf = self._mm[start:start + row_bytes]
        row = np.frombuffer(buf, dtype=self._dtype).reshape(self._shape[1:])
        return row.copy()

    def read_slice(self, start: int, stop: int) -> np.ndarray:
        """批量读取 [start, stop) 记录（连续段，比逐条 read 更高效）。"""
        if start < 0 or stop > self.n_records or stop <= start:
            raise IndexError(f"invalid slice [{start}, {stop}) in [0, {self.n_records})")
        self._open()
        row_bytes = int(np.prod(self._shape[1:], dtype=np.int64)) * self._dtype.itemsize
        start_b = self._data_offset + start * row_bytes
        n = stop - start
        buf = self._mm[start_b:start_b + n * row_bytes]
        return np.frombuffer(buf, dtype=self._dtype).reshape(n, *self._shape[1:]).copy()

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StreamingRawReader:
    """按需流式读取（用于 NDT_ML_Flaw 压缩包内取指定条带）。

    不是把整个 .xz/.lzma 解压到磁盘：单遍流式解压（lzma.open）逐条带推进，
    只把**被请求的条带**载入内存，其余字节读后即弃。对 6.88 GB/批的原始
    数据，被请求 32 条带时峰值内存 ≈ 32×6.9 MB ≈ 220 MB。
    """

    @staticmethod
    def read_rows_from_compressed(
        comp_path: Path,
        n_rows: int,
        row_bytes: int,
        rows: Sequence[int],
    ) -> dict[int, bytes]:
        """单遍解压 ``comp_path``，返回 {row_index: raw_bytes}。

        ``row_bytes``：解压后每条带固定字节数（等宽条带才可跳过推进）。
        """
        import lzma
        target = set(rows)
        out: dict[int, bytes] = {}
        with lzma.open(comp_path, "rb") as f:
            for i in range(n_rows):
                raw = f.read(row_bytes)
                if len(raw) < row_bytes:
                    break                     # 文件尾（容忍尺寸微差）
                if i in target:
                    out[i] = raw
        return out


# ---------------------------------------------------------------------------
# 统一记录结构
# ---------------------------------------------------------------------------
@dataclass
class UnifiedRecord:
    """adapter 统一输出的一条记录（跨数据集字段对齐，不强制同尺寸图片）。"""

    record_id: str
    dataset_name: str
    specimen_id: str
    # 物理 / 缺陷归属
    defect_instance_id: Optional[str] = None      # 独立缺陷；背景记录为 None
    acquisition_id: Optional[str] = None          # 同一缺陷/位置的重复采集
    inspection_id: Optional[str] = None           # 一次检查 / 一个原始文件
    # 来源与标签
    data_origin: str = "unknown"                  # measured / simulated / derived
    defect_origin: str = "unknown"                # manufacturing/service/artificial_*/simulated
    label_status: str = "unknown"                 # positive / negative / ignore / unknown
    defect_present: bool = False
    defect_type: Optional[str] = None
    # source geometry + axes/units + 域元数据
    geometry: dict[str, Any] = field(default_factory=dict)      # 源几何（mm / 帧 / 像素）
    axes: list[str] = field(default_factory=list)                # tensor 轴顺序
    units: dict[str, str] = field(default_factory=dict)          # 每轴单位
    domain: dict[str, Any] = field(default_factory=dict)         # 域元数据（角度/频率/材料…）
    # tensor 定位（懒加载：不在构造时读数据）
    tensor_path: Optional[str] = None
    tensor_index: Optional[int] = None
    tensor_slice: Optional[list[int]] = None
    # 附加元数据（数据集特有字段保留）
    extra: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self, license_: str, source_file: str) -> dict[str, Any]:
        """转成 M0-1.5 schema 的 records 行（JSON/parquet 用的扁平 dict）。"""
        rec = {
            "record_id": self.record_id,
            "dataset_name": self.dataset_name,
            "modality": "ultrasonic",
            "specimen_id": self.specimen_id,
            "inspection_id": self.inspection_id,
            "defect_instance_id": self.defect_instance_id,
            "acquisition_id": self.acquisition_id,
            "position": {
                "x": self.geometry.get("x_mm"),
                "y": self.geometry.get("y_mm"),
                "z": self.geometry.get("z_mm"),
                "coordinate_system": self.geometry.get("coordinate_system", "unknown"),
            },
            "defect_present": self.defect_present,
            "label_status": self.label_status,
            "defect_type": self.defect_type,
            "data_origin": self.data_origin,
            "defect_origin": self.defect_origin,
            "license": license_,
            "source_file": source_file,
            "ultrasonic": {
                "tensor_key": "tensor",
                "tensor_index": self.tensor_index,
                "tensor_slice": self.tensor_slice,
                "scan_axis": self.axes[0] if self.axes else "x",
                "beam_angle": self.domain.get("beam_angle"),
                "depth": self.domain.get("depth_axis", "time"),
            },
        }
        rec["ultrasonic"].update(self.extra.get("ultrasonic", {}))
        return rec


def unified_fields(r: UnifiedRecord) -> dict[str, Any]:
    """返回统一字段扁平 dict（用于数据审计统计）。"""
    return {
        "dataset_name": r.dataset_name,
        "specimen_id": r.specimen_id,
        "defect_instance_id": r.defect_instance_id,
        "acquisition_id": r.acquisition_id,
        "data_origin": r.data_origin,
        "defect_origin": r.defect_origin,
        "label_status": r.label_status,
        "defect_present": r.defect_present,
        "defect_type": r.defect_type,
        "geometry": r.geometry,
        "axes": r.axes,
        "units": r.units,
        "domain": r.domain,
    }


# ---------------------------------------------------------------------------
# manifest 生成
# ---------------------------------------------------------------------------
def checksum_file(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> dict:
    """文件完整性校验（流式，不整体读入）。"""
    h = hashlib.new(algo)
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            size += len(b)
    return {"algorithm": algo, "digest": h.hexdigest(), "size_bytes": size}


def write_dataset_card(
    dataset_name: str,
    primary_modality: str,
    license_: str,
    source: dict[str, Any],
    n_specimens: int,
    n_defect_instances: int,
    n_records: int,
    specimens: list[dict[str, Any]],
    defects: list[dict[str, Any]],
    tensors: list[dict[str, Any]],
    out_dir: Path,
    extra: Optional[dict[str, Any]] = None,
    records_ref_format: str = "parquet",
) -> Path:
    """写顶层 dataset card（不内嵌大规模 records，走 records_ref）。"""
    card = {
        "manifest_version": "0.2.0",
        "dataset_name": dataset_name,
        "primary_modality": primary_modality,
        "license": license_,
        "source": source,
        "n_specimens": n_specimens,
        "n_defect_instances": n_defect_instances,
        "n_records": n_records,
        "records_ref": {
            "path": f"records.{records_ref_format}",
            "format": records_ref_format,
            "n_records": n_records,
        },
        "specimens": specimens,
        "defects": defects,
        "tensors": tensors,
    }
    if extra:
        card.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "dataset_card.json"
    p.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_records_parquet(records: Sequence[UnifiedRecord], out_dir: Path, license_: str,
                          source_file: str) -> Path:
    """records.parquet：每个 UnifiedRecord 一行（含 tensor 引用，不含信号本身）。"""
    rows = [r.to_manifest_record(license_, source_file) for r in records]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "records.parquet"
    df.to_parquet(p, index=False)
    return p
