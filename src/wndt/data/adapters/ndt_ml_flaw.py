"""NDT_ML_Flaw（VTT / koomas）B-scan 条带适配器（M0-2A）。

来源：https://github.com/koomas/NDT_ML_Flaw
结构：``datasets/`` 下 17 批 —— 7 批真实裂纹（``.xz``，batch_013..019）+ 10 批
CIVA 仿真（``.lzma``，batch_201..210），每批配一个 ``.txt`` 元数据。每批
解压后为 1000 条 B-scan 条带 × 480(深度) × 7168(扫描轴) uint16，单批 ~6.88 GB
原始（17 批共 ~117 GB），压缩后仅 ~236 MB。

关键口径（与审计 docs/M0_public_ndt_dataset_audit.md 一致，经实测确认）：
- **1 个独立试件**（P41，异种金属焊缝）；
- **6 个独立缺陷**：P41_01..05（真实裂纹）+ P41_06_notch（EDM 人工缺陷）；
  CIVA 仿真批（201..210）是对同类型裂纹的仿真增强；
- **~17,000 条带 ≠ 17,000 个独立缺陷**：条带是同一缺陷沿扫描轴的采集 /
  增强（真实批含 augmentation 0.4–1.0 幅度缩放）；同一批混合多个缺陷
  （batch_013 同时含 P41_01 与 P41_06_notch 等），``defect_instance_id``
  由 txt 的 defect_type 列给出；
- 真实批标签 7 列 ``[Flaw 0/1, 增强量, 缺陷深度, 缺陷位置, 原始尺寸 mm, 索引, 缺陷类型]``，
  仿真批 6 列（无缺陷类型）。

数据访问红线（任务要求）：
- **禁止完整解压全部 .xz/.lzma 到磁盘** —— 全部走流式解压 + 按需读取；
- ``read_strip`` 单遍解压单个批，只把被请求的条带载入内存（读后即弃/小缓存）；
- ``_xz_uncompressed_size`` 只读 xz index 得到未压缩字节数，不触碰数据体。

License 说明：仓库为 LGPL-3.0，对"数据"授权语义模糊（数据非代码）。
本仓库不重新分发原始数据，仅记录 license_warning，不阻塞研究使用。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from wndt.data.adapters.base import (
    BaseNDTAdapter, ManifestField, ManifestSplitter, NDTInstance, NDTModality,
)
from wndt.data.adapters.common import (
    REPO, RAW, UnifiedRecord, StreamingRawReader,
    write_dataset_card, write_records_parquet,
)

DATASET_NAME = "ndt_ml_flaw"
DATA_ROOT = RAW / "NDT_ML_Flaw"
MANIFEST_DIR = REPO / "data" / "manifests" / "ndt_ml_flaw"
LICENSE = "LGPL-3.0"
SOURCE_URL = "https://github.com/koomas/NDT_ML_Flaw"
SPECIMEN_ID = "P41"                     # 异种金属焊缝试件
STRIP_SHAPE = (480, 7168)               # (深度, 扫描轴) uint16 —— 已实测锁定
N_STRIPS_PER_BATCH = 1000


# ---------------------------------------------------------------------------
# 压缩包原始字节数（只读头部/index，不解压数据）
# ---------------------------------------------------------------------------
def _read_lzma_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """LZMA-style varint (每字节 7 bit, 高位 0x80 为续)。返回 (值, 新 pos)。"""
    val = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _xz_uncompressed_size(path: Path) -> int | None:
    """从 .xz stream footer + index 读总未压缩字节数（不解压数据）。"""
    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size < 24:
            return None
        f.seek(file_size - 12)
        footer = f.read(12)
        if footer[-2:] != b"YZ":
            return None
        # backward_size 字段 = 实际字节数/4 - 1
        bs_field = int.from_bytes(footer[4:8], "little")
        backward_size = (bs_field + 1) * 4
        index_pos = file_size - 12 - backward_size
        if index_pos < 0:
            return None
        f.seek(index_pos)
        index = f.read(backward_size + 12)
        if not index or index[0] != 0x00:
            return None
        # number_of_records (varint)；每个 record 为
        # [Unpadded Size, Uncompressed Size]（XZ spec §3.4，顺序如此）
        n_recs, pos = _read_lzma_varint(index, 1)
        total = 0
        for _ in range(n_recs):
            _unpadded, pos = _read_lzma_varint(index, pos)
            usize, pos = _read_lzma_varint(index, pos)
            total += usize
        return total


class NDTMLFlawAdapter(BaseNDTAdapter):
    """NDT_ML_Flaw 条带适配器：17 批 × 1000 条 = 17,000 记录，1 试件 16 缺陷实例。"""

    dataset_name = DATASET_NAME
    modality = NDTModality.ULTRASONIC

    def __init__(
        self,
        data_root: str | Path = DATA_ROOT,
        manifest_path: str | Path | None = None,
    ):
        super().__init__(manifest_path=manifest_path or MANIFEST_DIR / "dataset_card.json",
                         data_root=data_root)
        self.data_root = Path(data_root)
        self.datasets_dir = self.data_root / "datasets"
        self._records: Optional[list[UnifiedRecord]] = None
        self._batch_index: dict[str, dict[str, Any]] = {}
        self._strip_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 元数据解析（完整解析全部 .txt，不做任何解压）
    # ------------------------------------------------------------------
    def _list_batches(self) -> list[Path]:
        return sorted(self.datasets_dir.glob("*.xz")) + sorted(self.datasets_dir.glob("*.lzma"))

    def _batch_id(self, comp: Path) -> str:
        return comp.stem

    @staticmethod
    def _parse_txt(txt: Path, is_simulated: bool) -> list[dict[str, Any]]:
        """解析一个 .txt 元数据 → 该批 1000 条带的标签行。

        真实批 7 列，末列 defect_type 为字符串（P41_01 / P41_06_notch / "-"）；
        仿真批 6 列全数值。前 6 列统一转 float，末列保持字符串。
        """
        rows = []
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            row: dict[str, Any] = {"flaw": None, "augmentation": None,
                                   "depth": None, "position": None,
                                   "size_mm": None, "index": None,
                                   "defect_type": None}
            try:
                nums = [float(x) for x in parts[:6]]
            except ValueError:
                continue
            if not nums:
                continue
            (row["flaw"], row["augmentation"], row["depth"], row["position"],
             row["size_mm"], row["index"]) = nums[:6]
            if len(parts) >= 7:                 # 真实批才有 defect_type 列
                row["defect_type"] = parts[6].strip()
            rows.append(row)
        return rows

    @staticmethod
    def _raw_batch_bytes(comp: Path) -> int | None:
        """压缩包内原始字节数（只读 xz index / lzma header，不解压数据）。

        注意：本仓库的 ``.lzma`` 文件实际是 **XZ 格式**（magic ``fd377a585a00``，
        CIVA 仿真批由 XZ 压缩但扩展名为 .lzma），按扩展名走 LZMA-alone header
        会解析出垃圾字节数。一律先按 magic 判 XZ，再用 LZMA-alone 兜底。
        """
        try:
            with comp.open("rb") as f:
                magic = f.read(6)
            if magic == b"\xfd7zXZ\x00":
                return _xz_uncompressed_size(comp)
            hdr = comp.open("rb").read(13)
            if len(hdr) >= 13:
                import struct
                size = struct.unpack("<Q", hdr[5:13])[0]
                return None if size == 0xFFFFFFFFFFFFFFFF else int(size)
            return None
        except Exception:
            return None

    def _batch_info(self, comp: Path) -> dict[str, Any]:
        """单个批：压缩文件 + txt 标签 + 来源类型（real/simulated）。"""
        bid = self._batch_id(comp)
        txt = comp.with_suffix(".txt")
        is_sim = comp.suffix == ".lzma"
        rows = self._parse_txt(txt, is_sim) if txt.exists() else []
        return {
            "batch_id": bid,
            "comp_path": comp,
            "is_simulated": is_sim,
            "n_strips": len(rows) if rows else N_STRIPS_PER_BATCH,
            "rows": rows,
            "raw_batch_bytes": self._raw_batch_bytes(comp),
        }

    def _build_records(self) -> list[UnifiedRecord]:
        records: list[UnifiedRecord] = []
        for comp in self._list_batches():
            bi = self._batch_info(comp)
            self._batch_index[bi["batch_id"]] = bi
            n = bi["n_strips"]
            for s in range(n):
                row = bi["rows"][s] if s < len(bi["rows"]) else {}
                flaw = bool(int(row["flaw"])) if row.get("flaw") is not None else None
                # 缺陷实例：
                # - 真实批：用 txt 的 defect_type 列（同一批可含多个缺陷，
                #   如 batch_013 同时含 P41_01 与 P41_06_notch），clean 行为 "-"
                # - 仿真批：每批一个 CIVA 仿真实例
                if bi["is_simulated"]:
                    defect_id = f"ndtmf:{SPECIMEN_ID}:civa:{bi['batch_id']}"
                    data_origin, defect_origin = "simulated", "simulated"
                    defect_type = "civa_simulated_crack"
                    label_status = "positive" if flaw else "negative"
                else:
                    dt = str(row.get("defect_type") or "").strip()
                    if dt.startswith("P41_06"):
                        defect_type = "edm_notch"
                        defect_origin = "artificial_edm"
                    elif dt.startswith("P41_"):
                        defect_type = "crack"
                        defect_origin = "manufacturing"
                    else:
                        defect_type = None
                        defect_origin = "unknown"
                    data_origin = "measured"
                    label_status = "positive" if flaw else "negative"
                    # defect_id 只在缺陷条带（flaw=1）时指向具体缺陷；clean 为 None
                    defect_id = (f"ndtmf:{SPECIMEN_ID}:{dt}"
                                 if (flaw and defect_type) else None)

                records.append(UnifiedRecord(
                    record_id=f"ndtmf:{bi['batch_id']}:strip{s}",
                    dataset_name=DATASET_NAME,
                    specimen_id=SPECIMEN_ID,
                    defect_instance_id=defect_id,
                    acquisition_id=bi["batch_id"],
                    inspection_id=bi["batch_id"],
                    data_origin=data_origin,
                    defect_origin=defect_origin,
                    label_status=label_status,
                    defect_present=bool(flaw),
                    defect_type=defect_type,
                    geometry={
                        "depth_voxel": row.get("depth"),
                        "position_voxel": row.get("position"),
                        "size_mm": row.get("size_mm"),
                        "augmentation": row.get("augmentation"),
                        "coordinate_system": "strip_pixels",
                    },
                    axes=["depth", "scan"],
                    units={"depth": "voxel_480", "scan": "voxel_7168"},
                    domain={
                        "dataset": "NDT_ML_Flaw",
                        "batch": bi["batch_id"],
                        "is_simulated": bi["is_simulated"],
                        "strip_shape": list(STRIP_SHAPE),
                        "raw_dtype": "uint16",
                        "raw_batch_bytes": bi["raw_batch_bytes"],
                        "compressed_bytes": comp.stat().st_size,
                    },
                    tensor_path=f"data/raw/NDT_ML_Flaw/datasets/{comp.name}",
                    tensor_index=s,
                    extra={
                        "ultrasonic": {"tensor_key": "strip", "strip_index": s},
                        "batch": bi["batch_id"],
                        "row": {k: v for k, v in row.items() if v is not None},
                    },
                ))
        self._records = records
        return records

    # ------------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------------
    def records(self) -> list[UnifiedRecord]:
        if self._records is None:
            self._build_records()
        return self._records

    def load_manifest(self) -> list[NDTInstance]:
        if self._records is None:
            self._build_records()
        out = []
        for r in self._records:
            out.append(NDTInstance(
                record_id=r.record_id,
                metadata={
                    "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                    "defect_instance_id": r.defect_instance_id,
                    "acquisition_id": r.acquisition_id,
                    "defect_present": r.defect_present,
                    "label_status": r.label_status,
                    "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                    "domain": r.domain, "axes": r.axes, "units": r.units,
                    "split_group": f"defect:{r.defect_instance_id or 'clean'}",
                }))
        return out

    def __len__(self) -> int:
        return len(self.records())

    def read_record(self, i: int) -> NDTInstance:
        """单条记录（含 tensor：一条 480×7168 uint16 条带，流式解压）。"""
        r = self.records()[i]
        strip = self.read_strip(i)
        return NDTInstance(
            record_id=r.record_id,
            metadata={
                "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                "defect_instance_id": r.defect_instance_id,
                "acquisition_id": r.acquisition_id,
                "label_status": r.label_status, "defect_present": r.defect_present,
                "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                "axes": r.axes, "units": r.units, "domain": r.domain,
                "geometry": r.geometry, "extra": r.extra,
                "split_group": f"defect:{r.defect_instance_id or 'clean'}",
            },
            tensors={"strip": strip},
        )

    # -- 流式读取（不完整解压）-------------------------------------------
    def _resolve_comp(self, r: UnifiedRecord) -> Path:
        p = Path(r.tensor_path)
        if p.is_absolute():
            return p
        # tensor_path 以仓库根为基准（如 raw/NDT_ML_Flaw/datasets/x.xz）
        if (REPO / p).exists():
            return REPO / p
        return self.data_root / p

    def read_strip(self, i: int) -> np.ndarray:
        """读取第 ``i`` 条带（B-scan 480×7168 uint16）。单遍流式解压该批。"""
        r = self.records()[i]
        comp = self._resolve_comp(r)
        cache_key = f"{r.acquisition_id}:{r.tensor_index}"
        if cache_key in self._strip_cache:
            return self._strip_cache[cache_key]
        bi = self._batch_index.get(r.acquisition_id) or self._batch_info(comp)
        n = bi["n_strips"]
        row_bytes = int(np.prod(STRIP_SHAPE)) * 2
        got = StreamingRawReader.read_rows_from_compressed(
            comp, n, row_bytes, [r.tensor_index])
        if r.tensor_index not in got:
            raise EOFError(f"strip {r.tensor_index} not found in {comp.name}")
        arr = np.frombuffer(got[r.tensor_index], dtype=np.uint16).reshape(STRIP_SHAPE)
        if len(self._strip_cache) > 8:     # 轻量 LRU 上限
            self._strip_cache.clear()
        self._strip_cache[cache_key] = arr
        return arr

    def read_batch_strips(self, batch_id: str, rows: Sequence[int]) -> list[tuple[int, np.ndarray]]:
        """单遍解压一个批，返回请求的若干条带（随机抽样友好）。"""
        comp = self.datasets_dir / f"{batch_id}.xz" if batch_id in self._real_ids() \
            else self.datasets_dir / f"{batch_id}.lzma"
        if not comp.exists():
            cands = list(self.datasets_dir.glob(f"{batch_id}.*"))
            if not cands:
                raise FileNotFoundError(batch_id)
            comp = cands[0]
        bi = self._batch_info(comp)
        row_bytes = int(np.prod(STRIP_SHAPE)) * 2
        got = StreamingRawReader.read_rows_from_compressed(
            comp, bi["n_strips"], row_bytes, list(rows))
        out = []
        for idx in rows:
            if idx in got:
                out.append((idx, np.frombuffer(got[idx], dtype=np.uint16).reshape(STRIP_SHAPE)))
        return out

    def _real_ids(self) -> set[str]:
        return {c.stem for c in self.datasets_dir.glob("*.xz")}

    # -- split ----------------------------------------------------------
    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42):
        if protocol == "defect":
            # 按独立缺陷/仿真实例划分：同一缺陷的全部条带不跨 split。
            splitter = ManifestSplitter(self.load_manifest(), ManifestField.DEFECT_INSTANCE_ID)
            return splitter.split(val_ratio=val_ratio, test_ratio=0.2, seed=seed)
        if protocol == "specimen":
            # 单一试件 → 无标本级划分，报错（避免假多试件）
            raise ValueError(
                "NDT_ML_Flaw has a single specimen (P41); specimen-level split "
                "is meaningless. Use 'defect' protocol (group by defect instance).")
        if protocol == "batch":
            rng = np.random.default_rng(seed)
            batches = sorted({r.acquisition_id for r in self.records()})
            rng.shuffle(batches)
            n_val = max(1, round(len(batches) * val_ratio))
            n_test = max(1, round(len(batches) * 0.2))
            split = {"train": [], "val": [], "test": []}
            for part, bl in (("train", batches[:-(n_val + n_test) or None]),
                             ("val", batches[len(batches) - n_val - n_test:len(batches) - n_test]),
                             ("test", batches[-n_test:])):
                split[part] = [i for i, r in enumerate(self.records())
                               if r.acquisition_id in bl]
            return split
        raise NotImplementedError(protocol)

    def validate_defect_split(self, split: dict[str, list[int]]) -> bool:
        by_def: dict[str, list[int]] = {}
        for i, r in enumerate(self.records()):
            d = r.defect_instance_id or "clean"
            by_def.setdefault(d, []).append(i)
        for d, idx in by_def.items():
            parts = {p for p, arr in split.items() if any(i in arr for i in idx)}
            assert len(parts) == 1, f"defect {d} spans {parts} (leak!)"
        return True


# ---------------------------------------------------------------------------
# manifest 生成
# ---------------------------------------------------------------------------
def build_ndt_ml_flaw_manifest(out_dir: Path = MANIFEST_DIR, write_parquet: bool = True):
    """生成 NDT_ML_Flaw dataset card + records.parquet。"""
    ad = NDTMLFlawAdapter()
    records = ad.records()

    # 缺陷实例清单
    defects = []
    seen = set()
    for r in records:
        if r.defect_instance_id and r.defect_instance_id not in seen:
            seen.add(r.defect_instance_id)
            ds = {}
            if r.geometry.get("size_mm") is not None:
                ds["length"] = r.geometry["size_mm"]
            defects.append({
                "defect_instance_id": r.defect_instance_id,
                "specimen_id": SPECIMEN_ID,
                "defect_type": r.defect_type,
                "data_origin": r.data_origin,
                "defect_origin": r.defect_origin,
                "defect_location": {"coordinate_system": "strip_pixels",
                                    **{k: v for k, v in {"x": r.geometry.get("position_voxel")}.items()
                                       if v is not None}},
                "defect_size": ds,
            })

    comps = ad._list_batches()
    n_real = sum(1 for c in comps if c.suffix == ".xz")
    n_sim = sum(1 for c in comps if c.suffix == ".lzma")
    total_raw = sum(ad._raw_batch_bytes(c) or 0 for c in comps)
    total_comp = sum(c.stat().st_size for c in comps)

    card_path = write_dataset_card(
        dataset_name=DATASET_NAME,
        primary_modality="ultrasonic",
        license_=LICENSE,
        source={
            "official_name": "NDT_ML_Flaw (VTT)",
            "url": SOURCE_URL,
            "commit": _git_head(),
            "license": LICENSE,
            "license_warning": "LGPL-3.0 针对代码；对'数据'的授权语义模糊。本仓库不重新分发原始数据，不阻塞研究使用。",
            "size_bytes": total_comp,
            "notes": f"{n_real} real .xz + {n_sim} simulated .lzma; raw ~{total_raw/1e9:.0f} GB compressed {total_comp/1e6:.0f} MB",
        },
        n_specimens=1,
        n_defect_instances=len(defects),
        n_records=len(records),
        specimens=[{
            "specimen_id": SPECIMEN_ID,
            "dataset_name": DATASET_NAME,
            "material": "dissimilar metal weld",
            "manufacturing": "weld (VTT test block P41)",
            "geometry": "single weld block; ~17000 B-scan strips",
            "source_file": "data/raw/NDT_ML_Flaw/datasets",
        }],
        defects=defects,
        tensors=[{
            "key": "strip", "path": "raw/NDT_ML_Flaw/datasets/<batch>.(xz|lzma)",
            "format": "raw", "axes": ["depth", "scan"],
            "dtype": "uint16", "unit": "raw_amplitude",
            "n_records": len(records),
        }],
        out_dir=out_dir,
        extra={
            "data_policy": {
                "specimen_count": 1,
                "strip_is_record": True,
                "strip_is_not_defect": True,
                "n_batches": len(comps),
                "n_real_batches": n_real,
                "n_simulated_batches": n_sim,
                "split_by": "defect_instance_id",
                "streaming_only": True,
                "no_full_extraction": True,
            },
            "provenance": {"steps": ["parse 17 .txt metadata (no decompression)",
                                     "stream-decompress per strip on demand"],
                           "software": "src/wndt/data/adapters/ndt_ml_flaw.py"},
        },
    )
    if write_parquet:
        rec_path = write_records_parquet(records, out_dir, LICENSE,
                                         source_file="data/raw/NDT_ML_Flaw/datasets/*.xz|lzma")
    else:
        rec_path = out_dir / "records.parquet"
    print(f"[ndt_ml_flaw] card: {card_path}")
    print(f"[ndt_ml_flaw] recs: {rec_path}  ({len(records)} records)")
    print(f"[ndt_ml_flaw] specimens=1 defect_instances={len(defects)} batches={len(comps)}")
    return card_path, rec_path


def _git_head() -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(DATA_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


if __name__ == "__main__":
    build_ndt_ml_flaw_manifest()
