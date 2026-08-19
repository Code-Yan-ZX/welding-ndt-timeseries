"""ML-NDT（VTT）PAUT 体积适配器（M0-2A）。

来源：https://github.com/iikka-v/ML-NDT （Virkkunen et al., 2019, arXiv:1903.11399）
结构：每个 UUID 批次目录含 ``.bins``（原始 3D 超声体积 UInt16 256×256×100 =
100 帧 × 256×256 B-scan，13.1 MB）+ ``.meta`` + ``.jsons``（缺陷元数据）+
``.labels``（100 行 [flaw 0/1, equivalent flaw size]）。

关键口径（与审计 docs/M0_public_ndt_dataset_audit.md 一致）：
- **1 个独立试件**：316L 奥氏体管道单对焊接头；
- **3 条真实热疲劳裂纹**（深度 1.6/4.0/8.6 mm，Trueflaw 制造）+ eFlaw
  幅度缩放/重植入的 virtual flaws（仿真增强）；
- **201 个 volume ≠ 201 个试件**：volume 是同一试件上的一次采集，
  ``defect_instance_id`` 标识真正的独立缺陷，volume/frame 是重复/增强；
- split 必须按 defect_instance_id 划分（跨 volume 同一个缺陷属于同一单元）。

本适配器：
- 解析 .labels/.jsons/.meta 构建 manifest（volume 级记录，frame ID 保留）；
- 支持按 volume / flaw 流式读取（mmap 或逐 volume 载入，绝不整仓 2.6 GB
  一次进内存）；
- 不把 201 volume 当作 201 独立试件；独立 specimen=1。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from wndt.data.adapters.base import (
    BaseNDTAdapter, ManifestField, ManifestSplitter, NDTInstance, NDTModality,
)
from wndt.data.adapters.common import (
    REPO, RAW, UnifiedRecord, checksum_file,
    write_dataset_card, write_records_parquet,
)

DATASET_NAME = "ml_ndt"
DATA_ROOT = RAW / "ML-NDT"
MANIFEST_DIR = REPO / "data" / "manifests" / "ml_ndt"
LICENSE = "LGPL-3.0"
SOURCE_URL = "https://github.com/iikka-v/ML-NDT"
VOLUME_SHAPE_RAW = (256, 256, 100)  # 原始 .bins 布局：256x256 空间 x 100 帧(末轴)
VOLUME_SHAPE = (100, 256, 256)      # 统一输出布局：frame-first (帧, y, x)
N_FRAMES = 100
SPECIMEN_ID = "mlndt_pipe_weld"    # 单一 316L 奥氏体管道焊接接头
SPECIMEN_GEOMETRY = "austenitic stainless steel pipe weld (single butt joint)"


class MLNDTAdapter(BaseNDTAdapter):
    """ML-NDT 体积适配器：201 volume 记录，1 独立试件，3+ 独立缺陷。"""

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
        self._records: Optional[list[UnifiedRecord]] = None
        self._batch_dirs: list[Path] = []

    # ------------------------------------------------------------------
    # 元数据解析
    # ------------------------------------------------------------------
    def _list_batches(self) -> list[Path]:
        """找出全部 .bins volume（<root>/data/{training,validation}/ 扁平文件）。

        每个 .bins 是一个 volume，UUID 为 volume_id；不按子目录组织。
        兼容 <root>/{training,validation}/ 与 <root>/data/{training,validation}/。
        """
        if not self._batch_dirs:
            dirs = []
            for base in (self.data_root / "data", self.data_root):
                for sub in ("training", "validation"):
                    d = base / sub
                    if d.is_dir():
                        dirs.extend(sorted(d.glob("*.bins")))
            self._batch_dirs = dirs
        return self._batch_dirs

    def _batch_metadata(self, bdir: Path) -> dict[str, Any]:
        """解析一个 volume 的 .meta / .jsons(流) / .labels。

        ``bdir`` 传 .bins 文件路径（或 volume_id），同名 .meta/.jsons/.labels
        位于同一目录。
        """
        bins_path = Path(bdir) if Path(bdir).suffix == ".bins" else self._volume_path(bdir)
        volume_id = bins_path.stem
        meta: dict[str, Any] = {}
        m = bins_path.with_suffix(".meta")
        if m.exists():
            try:
                meta = json.loads(m.read_text())
            except Exception:
                meta = {"raw": m.read_text()[:2000]}
        labels_path = bins_path.with_suffix(".labels")
        j = bins_path.with_suffix(".jsons")
        jd = self._parse_jsons_stream(j) if j.exists() else []
        return {
            "meta": meta,
            "jsons": jd,
            "labels": self._parse_labels(labels_path) if labels_path.exists() else [],
            "volume_id": volume_id,
        }

    @staticmethod
    def _parse_labels(labels_path: Path) -> list[dict[str, Any]]:
        """.labels：100 行 [flaw 0/1, equivalent_flaw_size]。"""
        rows = []
        for line in labels_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append({
                "flaw": int(parts[0]),
                "equivalent_flaw_size": float(parts[1]) if len(parts) > 1 else None,
            })
        return rows

    @staticmethod
    def _parse_jsons_stream(json_path: Path) -> list[dict[str, Any]]:
        """.jsons 是**每帧一个**拼接 JSON 对象的流（100 个对象），非单 JSON。

        每个对象 ``{"flaws":[{"original_location":"652-708", ...}]}``；帧无
        缺陷时对象可能为 ``{"flaws":[]}``。字段是 JSON 数字的字符串表示。
        """
        data = json_path.read_text()
        dec = json.JSONDecoder()
        objs = []
        idx = 0
        while idx < len(data):
            while idx < len(data) and data[idx] in " \n\r\t":
                idx += 1
            if idx >= len(data):
                break
            try:
                obj, end = dec.raw_decode(data, idx)
            except json.JSONDecodeError:
                break
            objs.append(obj)
            idx = end
        return objs

    @staticmethod
    def _flaw_events(jsons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从 .jsons（每帧一个对象）提取该 volume 的**去重缺陷事件**。

        ML-NDT 的 volume 是"单试件 + 3 真实裂纹 + eFlaw 重植入"的密集复合：
        每帧一个缺陷事件，各自带 source size（1.6/4.0/8.6 之一）与 factor
        （幅度缩放，factor<1 为 virtual）。以 (size, factor, original_location)
        去重，字段字符串转数值。
        """
        seen: dict[tuple, dict[str, Any]] = {}
        for obj in jsons or []:
            for f in obj.get("flaws") or []:
                try:
                    efs = float(f.get("equivalent_flawsize", 0) or 0)
                    factor = float(f.get("factor", 1.0) or 1.0)
                    size = float(f.get("size", 0) or 0)
                    loc = str(f.get("original_location") or "")
                    loc_this = str(f.get("location") or "")
                except (TypeError, ValueError):
                    continue
                key = (round(size, 2), round(factor, 4), loc)
                if key in seen:
                    continue
                seen[key] = {
                    "source_size_mm": size,
                    "factor": factor,
                    "original_location": loc,
                    "location_this": loc_this,
                    "equivalent_flawsize": efs,
                    "max_amplitude": float(f.get("max_amplitude", 0) or 0),
                    "is_virtual": abs(factor - 1.0) > 1e-6,
                }
        return list(seen.values())

    @staticmethod
    def _source_crack_id(size_mm: float) -> str:
        """真实源裂纹的稳定 id（按 source size：1.6 / 4.0 / 8.6 mm）。"""
        return f"mlndt:{SPECIMEN_ID}:crack{size_mm:.1f}"

    def _build_records(self) -> list[UnifiedRecord]:
        records: list[UnifiedRecord] = []
        for bdir in self._list_batches():
            bm = self._batch_metadata(bdir)
            volume_id = bm["volume_id"]
            labels = bm["labels"]
            jd = bm["jsons"]
            events = self._flaw_events(jd)     # 去重缺陷事件（~每 volume 数十个）

            # volume 级标签：任一帧有缺陷
            n_flaw_frames = sum(1 for r in labels if r["flaw"] == 1) if labels else None
            has_flaw = n_flaw_frames and n_flaw_frames > 0

            # 独立缺陷实例 = 真实源裂纹（按 source size）；virtual 是 eFlaw
            # 对真实裂纹的幅度缩放重植入，不当作独立物理缺陷，但保留事件明细。
            source_sizes = sorted({e["source_size_mm"] for e in events if e["source_size_mm"] > 0})
            n_virtual = sum(1 for e in events if e["is_virtual"])
            n_real_events = len(events) - n_virtual
            primary_defect = (
                self._source_crack_id(source_sizes[0]) if source_sizes else None)

            # 帧范围缺陷位置（.jsons original_location 如 "652-708"）
            loc = None
            if events:
                orig = events[0].get("original_location") or ""
                m = re.match(r"(\d+)\s*-\s*(\d+)", str(orig).strip())
                if m:
                    loc = {"frame_start": int(m.group(1)), "frame_end": int(m.group(2)),
                           "coordinate_system": "frame_index"}

            n_frames = len(labels) if labels else N_FRAMES
            records.append(UnifiedRecord(
                record_id=f"mlndt:{volume_id}",
                dataset_name=DATASET_NAME,
                specimen_id=SPECIMEN_ID,
                defect_instance_id=primary_defect,
                acquisition_id=volume_id,               # volume = 一次采集
                inspection_id=volume_id,
                data_origin="measured",
                defect_origin="service" if primary_defect else "unknown",
                label_status="positive" if has_flaw else "negative",
                defect_present=bool(has_flaw),
                defect_type="thermal_fatigue_crack",
                geometry={**(loc or {}),
                          "n_frames": n_frames,
                          "volume_id": volume_id,
                          "coordinate_system": loc["coordinate_system"] if loc else "frame_index"},
                axes=["frame", "y", "x"],
                units={"frame": "index", "y": "voxel", "x": "voxel"},
                domain={
                    "dataset": "ML-NDT",
                    "probe": "Zetec Dynaray 64/64PR-Lite + Imasonic 1.5MHz matrix + ADUX577A wedge",
                    "scan_resolution_mm": 0.21,
                    "n_frames": n_frames,
                    "volume_shape": list(VOLUME_SHAPE),
                    "raw_dtype": "uint16",
                },
                tensor_path=f"data/raw/ML-NDT/{volume_id}",
                tensor_index=None,
                extra={
                    "ultrasonic": {
                        "tensor_key": "volume", "volume_id": volume_id,
                        "n_frames": n_frames, "equivalent_flawsize": None,
                    },
                    "n_flaw_frames": n_flaw_frames,
                    "source_sizes_mm": source_sizes,
                    "n_virtual_events": n_virtual,
                    "n_real_events": n_real_events,
                    "flaw_events": events,
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
        """单条记录（含 tensor：整个 volume 100×256×256 uint16 = 13.1 MB）。"""
        r = self.records()[i]
        vol = self.read_volume(i)
        labels = self._batch_metadata(r.acquisition_id)["labels"]
        return NDTInstance(
            record_id=r.record_id,
            metadata={
                "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                "defect_instance_id": r.defect_instance_id,
                "acquisition_id": r.acquisition_id,
                "label_status": r.label_status, "defect_present": r.defect_present,
                "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                "axes": r.axes, "units": r.units, "domain": r.domain,
                "geometry": r.geometry,
                "frame_labels": [x["flaw"] for x in labels],
                "split_group": f"defect:{r.defect_instance_id or 'clean'}",
            },
            tensors={"volume": vol},
        )

    # -- 流式读取 -------------------------------------------------------
    def _volume_path(self, volume_id: str) -> Path:
        for base in (self.data_root / "data", self.data_root):
            for sub in ("training", "validation"):
                p = base / sub / f"{volume_id}.bins"
                if p.exists():
                    return p
        raise FileNotFoundError(f"no .bins for volume {volume_id} under {self.data_root}")

    def read_volume(self, i: int) -> np.ndarray:
        """流式读取第 ``i`` 个 volume（单次只载入这一个，13.1 MB）。

        原始 .bins 布局为 ``(256, 256, 100)``（空间 x 空间 x 帧，末轴为帧），
        统一输出为 frame-first ``(100, 256, 256)``。
        """
        r = self.records()[i]
        vid = r.acquisition_id
        p = self._volume_path(vid)
        raw = np.fromfile(p, dtype=np.uint16, count=np.prod(VOLUME_SHAPE_RAW))
        # 允许少量尾部 padding（文件可能比理论大小略大/小）
        raw = raw.reshape(VOLUME_SHAPE_RAW)
        return np.moveaxis(raw, -1, 0)          # (256,256,100) -> (100,256,256)

    def read_frame(self, i: int, frame: int) -> np.ndarray:
        """读取第 ``i`` 个 volume 的第 ``frame`` 帧 B-scan (256,256)。"""
        vol = self.read_volume(i)
        return vol[frame]

    def read_volume_slice(self, start: int, stop: int) -> list[tuple[str, np.ndarray]]:
        """批量读 [start, stop) 个 volume（逐 volume 载入，绝不整仓）。"""
        out = []
        for i in range(start, stop):
            r = self.records()[i]
            out.append((r.record_id, self.read_volume(i)))
        return out

    def read_by_flaw(self, defect_instance_id: str) -> list[tuple[int, np.ndarray]]:
        """按独立缺陷读取其全部 volume（同一缺陷的多次采集/增强）。"""
        idx = [i for i, r in enumerate(self.records())
               if r.defect_instance_id == defect_instance_id]
        return [(i, self.read_volume(i)) for i in idx]

    # -- split ----------------------------------------------------------
    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42):
        if protocol == "defect":
            # 按独立缺陷划分：同一缺陷的全部 volume 落在同一 split。
            # 缺陷单元过少（3 真实 + 若干 virtual）时用缺陷级，防泄露。
            splitter = ManifestSplitter(self.load_manifest(), ManifestField.DEFECT_INSTANCE_ID)
            return splitter.split(val_ratio=val_ratio, test_ratio=0.2, seed=seed)
        if protocol == "volume":       # 仅诊断用（会跨缺陷泄漏）
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(self.records())).tolist()
            n_val = max(1, round(len(perm) * val_ratio))
            n_test = max(1, round(len(perm) * 0.2))
            return {
                "train": perm[:-(n_val + n_test) or None],
                "val": perm[len(perm) - n_val - n_test:len(perm) - n_test],
                "test": perm[-n_test:],
            }
        raise NotImplementedError(protocol)

    def validate_defect_split(self, split: dict[str, list[int]]) -> bool:
        """不变量：同一 defect_instance 的所有 volume 必须落在同一 split。"""
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
def build_ml_ndt_manifest(out_dir: Path = MANIFEST_DIR, write_parquet: bool = True):
    """生成 ML-NDT dataset card + records.parquet。"""
    ad = MLNDTAdapter()
    records = ad.records()

    # 独立缺陷 = 3 条真实源裂纹（1.6/4.0/8.6 mm）；virtual 事件作为增强统计。
    source_sizes = sorted({e["source_size_mm"]
                           for r in records for e in r.extra.get("flaw_events", [])
                           if e["source_size_mm"] > 0})
    defects = []
    for s in source_sizes:
        did = MLNDTAdapter._source_crack_id(s)
        # 该源裂纹的 virtual 事件样本（首个 volume 的明细作参考）
        sample = next((e for r in records for e in r.extra.get("flaw_events", [])
                       if e["source_size_mm"] == s and e["is_virtual"]), None)
        defects.append({
            "defect_instance_id": did,
            "specimen_id": SPECIMEN_ID,
            "defect_type": "thermal_fatigue_crack",
            "data_origin": "measured",
            "defect_origin": "service",
            "defect_size": {"length": s},
            "equivalent_flawsize": sample["equivalent_flawsize"] if sample else None,
            "is_virtual": False,
            "factor": 1.0,
            "notes": f"real crack ~{s}mm; many eFlaw virtual re-insertions derive from it",
        })
    n_virtual_total = sum(r.extra.get("n_virtual_events", 0) for r in records)

    # volume 级 checksum（首个 .bins 示例）
    bins = []
    for r in records:
        p = ad._volume_path(r.acquisition_id)
        bins.append(p)
    chk = checksum_file(bins[0]) if bins else None

    card_path = write_dataset_card(
        dataset_name=DATASET_NAME,
        primary_modality="ultrasonic",
        license_=LICENSE,
        source={
            "official_name": "ML-NDT (VTT)",
            "url": SOURCE_URL,
            "commit": _git_head(),
            "license": LICENSE,
            "size_bytes": sum(p.stat().st_size for p in bins),
            "notes": "201 volumes x (100,256,256) uint16; 1 specimen; 3 real thermal fatigue cracks + virtual flaws",
        },
        n_specimens=1,
        n_defect_instances=len(defects),
        n_records=len(records),
        specimens=[{
            "specimen_id": SPECIMEN_ID,
            "dataset_name": DATASET_NAME,
            "material": "316L austenitic stainless steel",
            "manufacturing": "single butt pipe weld",
            "geometry": SPECIMEN_GEOMETRY,
            "source_file": "data/raw/ML-NDT",
        }],
        defects=defects,
        tensors=[{
            "key": "volume", "path": "raw/ML-NDT/<volume_id>/*.bins",
            "format": "raw", "axes": ["frame", "y", "x"],
            "dtype": "uint16", "unit": "raw_amplitude",
            "n_records": len(records),
        }],
        out_dir=out_dir,
        extra={
            "data_policy": {
                "specimen_count": 1,       # 201 volume ≠ 201 specimen
                "volume_is_acquisition": True,
                "real_defect_count": 3,    # 热疲劳裂纹 1.6/4.0/8.6 mm
                "virtual_event_total": n_virtual_total,   # eFlaw 重植入事件总数
                "split_by": "defect_instance_id",
                "train_val": "train 199 / val 2 (original repo split)",
                "raw_volume_checksum": chk,
            },
            "provenance": {"steps": ["read .bins as uint16 (100,256,256)",
                                     "parse .labels/.jsons/.meta"],
                           "software": "src/wndt/data/adapters/ml_ndt.py"},
        },
    )
    if write_parquet:
        rec_path = write_records_parquet(records, out_dir, LICENSE,
                                         source_file="data/raw/ML-NDT/*.bins")
    else:
        rec_path = out_dir / "records.parquet"
    print(f"[ml_ndt] card: {card_path}")
    print(f"[ml_ndt] recs: {rec_path}  ({len(records)} records)")
    print(f"[ml_ndt] specimens=1 defect_instances={len(defects)} volumes={len(records)}")
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
    build_ml_ndt_manifest()
