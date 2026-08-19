"""PENELOPE（SAW Open Repository）PAUT 适配器（M0-2A）。

数据源：仓库已处理的 ``data/processed/paut/``（90 族 / DataGroup 0，71° /
49 波束 / 3500 采样，max-pool 到 512）+ 原始 ``data/raw/saw/ZENODO_Penelope/``
下的 ``2. ndt_data/``（.nde + defects_xlocation.xlsx）。

本适配器：
- 覆盖 5 个完整 PAUT coupon（PP3–PP7），3000 位置级记录；
- 每位置：record_id / specimen_id(coupon) / inspection_id(.nde 文件名) /
  x[mm] / label_status / 原始 .nde + DataGroup/view / tensor key+index /
  beam 数 / 深度采样数 / 角度 / preprocessing provenance；
- 从 defects_xlocation.xlsx 构建独立 defect instance（每 coupon 每标注行），
  并按与 ``scripts/paut_preprocess.py`` 相同的局部缺陷口径把位置映射到
  缺陷实例（>50mm 贯穿缺陷视为背景）；
- 按 specimen / view 读取；单条/批量 B-scan；specimen-level split；
- tensor 用 mmap 流式读取，绝不把 3000×49×512 整份载入内存。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from wndt.data.adapters.base import (
    BaseNDTAdapter, ManifestField, ManifestSplitter, NDTInstance, NDTModality,
)
from wndt.data.adapters.common import (
    REPO, UnifiedRecord, StreamingNpyReader, checksum_file,
    write_dataset_card, write_records_parquet,
)

DATASET_NAME = "penelope_paut"
PROCESSED = REPO / "data" / "processed" / "paut"
RAW_ROOT = REPO / "data" / "raw" / "saw" / "ZENODO_Penelope"
MANIFEST_DIR = REPO / "data" / "manifests" / "penelope"
LICENSE = "CC-BY-4.0"
BIG_DEFECT_MM = 50.0          # 与 scripts/paut_preprocess.py 口径一致
BEAM_ANGLE_DEG = 71.0         # DataGroup 0 折射角
N_BEAMS = 49
N_DEPTH = 512                 # 处理后深度采样数（原 3500 max-pool）
GROUP = 0                     # .nde DataGroup 索引
VIEW = "90/G0(71deg,49beam)"

# 每 coupon 的 90 族 .nde 文件名（与 meta_summary.json 一致）
NDE_FILE = {
    "PP3": "PAUT_90.nde",
    "PP4": "PAUT_90+.nde",
    "PP5": "PAUT_90+.nde",
    "PP6": "PAUT_90+.nde",
    "PP7": "1163421_PP7_030325_90.nde",
}
# 规范 split：train=PP3-5 / val=PP6 / test=PP7（与仓库实验口径一致）
CANONICAL_SPLIT = {"train": ["PP3", "PP4", "PP5"], "val": ["PP6"], "test": ["PP7"]}
DEFECT_CODES = {1: "Porosity", 2: "Lack of fusion", 3: "Slag inclusion",
                4: "Metallic inclusion", 5: "Projections", 6: "Cracks"}


class PENELOPEAdapter(BaseNDTAdapter):
    """PENELOPE SAW PAUT 适配器：5 coupon × 位置级记录 = 3000 样本。"""

    dataset_name = DATASET_NAME
    modality = NDTModality.ULTRASONIC

    def __init__(
        self,
        processed_dir: str | Path = PROCESSED,
        raw_root: str | Path = RAW_ROOT,
        manifest_path: str | Path | None = None,
    ):
        super().__init__(manifest_path=manifest_path or MANIFEST_DIR / "dataset_card.json",
                         data_root=processed_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_root = Path(raw_root)
        self.summary = json.loads((self.processed_dir / "meta_summary.json").read_text())
        self._records: Optional[list[UnifiedRecord]] = None
        self._ascans: Optional[StreamingNpyReader] = None
        self._env: Optional[StreamingNpyReader] = None

    # ------------------------------------------------------------------
    # 元数据构建
    # ------------------------------------------------------------------
    def _load_arrays(self):
        self._meta_label = np.load(self.processed_dir / "meta_label.npy")
        self._meta_coupon = np.load(self.processed_dir / "meta_coupon.npy")
        self._meta_pos = np.load(self.processed_dir / "meta_pos.npy")
        self._meta_type = np.load(self.processed_dir / "meta_defect_type.npy")
        self._coupon_order = [c for c in ("PP3", "PP4", "PP5", "PP6", "PP7")
                              if c in self.summary.get("per_coupon", {})]
        # 每个 coupon 在 ascans.npy 中的行偏移（按 preprocess 顺序）
        self._offsets: dict[str, int] = {}
        cursor = 0
        for c in self._coupon_order:
            self._offsets[c] = cursor
            cursor += self.summary["per_coupon"][c]["n_pos"]

    def _defect_instances(self, coupon: str) -> list[dict[str, Any]]:
        """从 defects_xlocation.xlsx 构建独立缺陷实例（每标注行一个）。

        返回列表，每项含 defect_instance_id / code / bead / x_init / x_end /
        length_mm。>50mm 贯穿缺陷保留为 instance 但标记 label_status=ignore
        （与预处理口径一致：位置标签视为背景）。
        """
        ndt = self.raw_root / coupon / "2. ndt_data"
        f = ndt / "defects_xlocation.xlsx"
        if not f.exists():
            return []
        df = pd.read_excel(f, sheet_name=coupon)
        df = df.rename(columns=lambda s: str(s).strip())
        out = []
        for i, r in df.iterrows():
            xi, xe = float(r["x_init [mm]"]), float(r["x_end [mm]"])
            if xe < xi:                       # 数据录入反转 -> 交换
                xi, xe = xe, xi
            length = abs(float(r["x_end [mm]"]) - float(r["x_init [mm]"]))
            code = int(r["defect"])
            out.append({
                "defect_instance_id": f"penelope:{coupon}:row{i}",
                "code": code,
                "defect_type": DEFECT_CODES.get(code, f"code{code}"),
                "bead": int(r["bead"]) if pd.notna(r["bead"]) else None,
                "x_init_mm": xi, "x_end_mm": xe,
                "length_mm": length,
                "localized": length < BIG_DEFECT_MM,
            })
        return out

    def _build_records(self) -> list[UnifiedRecord]:
        self._load_arrays()
        records: list[UnifiedRecord] = []
        rec_idx = 0
        for c in self._coupon_order:
            pc = self.summary["per_coupon"][c]
            n_pos = int(pc["n_pos"])
            offset = float(pc["offset_mm"])
            res = float(pc["res_mm"])
            n_samples_raw = int(pc["n_samples"])
            nde = pc.get("nde", NDE_FILE.get(c, "?"))

            # 缺陷实例（该 coupon 的标注行）
            defects = self._defect_instances(c)
            # 位置 -> defect_instance_id：与 preprocess 相同口径
            # （局部缺陷重叠，dominant = max code）
            pos_defect = np.full(n_pos, None, dtype=object)
            for d in defects:
                if not d["localized"]:
                    continue
                i0 = max(0, int(round((d["x_init_mm"] - offset) / res)))
                i1 = min(n_pos - 1, int(round((d["x_end_mm"] - offset) / res)))
                if i1 < i0:
                    continue
                # 仅在"该位置尚无更高 code 缺陷"时占用（与 max dominant 一致）
                for p in range(i0, i1 + 1):
                    cur = pos_defect[p]
                    if cur is None or d["code"] > cur[1]:
                        pos_defect[p] = (d["defect_instance_id"], d["code"])

            for p in range(n_pos):
                gi = self._offsets[c] + p           # 全局行索引
                lab = int(self._meta_label[gi])
                tcode = int(self._meta_type[gi])
                x_mm = offset + p * res
                did = pos_defect[p][0] if pos_defect[p] is not None else None
                records.append(UnifiedRecord(
                    record_id=f"{c}_{p:04d}",
                    dataset_name=self.dataset_name,
                    specimen_id=c,
                    inspection_id=nde,
                    defect_instance_id=did,
                    acquisition_id=f"{c}:90:G0",      # 90 族 / G0 单次采集
                    data_origin="measured",
                    defect_origin="manufacturing",
                    label_status="positive" if lab else "negative",
                    defect_present=bool(lab),
                    defect_type=DEFECT_CODES.get(tcode) if tcode else None,
                    geometry={
                        "x_mm": x_mm, "coordinate_system": "coupon_local",
                        "scan_resolution_mm": res, "offset_mm": offset,
                    },
                    axes=["beam", "time"],
                    units={"beam": "uCoordinate_mm", "time": "sample_512"},
                    domain={
                        "beam_angle": BEAM_ANGLE_DEG,
                        "n_beams": N_BEAMS,
                        "n_depth": N_DEPTH,
                        "n_samples_raw": n_samples_raw,
                        "group": GROUP, "view": VIEW,
                        "nde_file": nde,
                    },
                    tensor_path="data/processed/paut/ascans.npy",
                    tensor_index=gi,
                    tensor_slice=None,
                    extra={
                        "ultrasonic": {
                            "tensor_key": "ascans", "record_index": gi,
                            "beam_angle": BEAM_ANGLE_DEG,
                            "beam_count": N_BEAMS, "depth_samples": N_DEPTH,
                            "group": GROUP, "view": VIEW,
                        },
                        "coupon_pos": p, "global_index": gi,
                    },
                ))
                rec_idx += 1
        self._records = records
        return records

    # ------------------------------------------------------------------
    # BaseNDTAdapter 接口
    # ------------------------------------------------------------------
    def load_manifest(self) -> list[NDTInstance]:
        """返回 NDTInstance 列表（tensor 懒加载，不实际读信号）。"""
        if self._records is None:
            self._build_records()
        out = []
        for r in self._records:
            meta = {
                "record_id": r.record_id,
                "dataset_name": r.dataset_name,
                "specimen_id": r.specimen_id,
                "inspection_id": r.inspection_id,
                "defect_instance_id": r.defect_instance_id,
                "acquisition_id": r.acquisition_id,
                "defect_present": r.defect_present,
                "label_status": r.label_status,
                "data_origin": r.data_origin,
                "defect_origin": r.defect_origin,
                "defect_type": r.defect_type,
                "position": {"x": r.geometry.get("x_mm"),
                             "coordinate_system": "coupon_local"},
                "axes": r.axes, "units": r.units, "domain": r.domain,
                "geometry": r.geometry,
                "split_group": f"specimen:{r.specimen_id}",
            }
            out.append(NDTInstance(record_id=r.record_id, metadata=meta))
        return out

    def records(self) -> list[UnifiedRecord]:
        if self._records is None:
            self._build_records()
        return self._records

    def __len__(self) -> int:
        return len(self.records())

    # -- 流式 tensor 读取 ------------------------------------------------
    def _open_readers(self):
        if self._ascans is None:
            self._ascans = StreamingNpyReader(self.processed_dir / "ascans.npy")
        if self._env is None:
            self._env = StreamingNpyReader(self.processed_dir / "env.npy")

    def read_record(self, i: int) -> NDTInstance:
        """单条记录（含 tensor）——只载入该条。"""
        r = self.records()[i]
        self._open_readers()
        bscan = self._ascans.read(r.tensor_index)     # (49, 512)
        env = self._env.read(r.tensor_index)          # (512,)
        return NDTInstance(
            record_id=r.record_id,
            metadata={
                "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                "defect_instance_id": r.defect_instance_id,
                "acquisition_id": r.acquisition_id,
                "label_status": r.label_status, "defect_present": r.defect_present,
                "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                "axes": r.axes, "units": r.units, "domain": r.domain,
                "geometry": r.geometry, "split_group": f"specimen:{r.specimen_id}",
            },
            tensors={"bscan": bscan, "env": env},
        )

    def read_batch(self, indices: Sequence[int], contiguous: bool = False) -> list[NDTInstance]:
        """批量读取；相邻索引可用 read_slice 一次 mmap 切片。"""
        self._open_readers()
        if contiguous and indices:
            lo, hi = min(indices), max(indices)
            if list(indices) == list(range(lo, hi + 1)):
                arr = self._ascans.read_slice(lo, hi + 1)
                envs = self._env.read_slice(lo, hi + 1)
                out = []
                for k, gi in enumerate(indices):
                    r = self.records()[gi]
                    out.append(NDTInstance(
                        record_id=r.record_id,
                        metadata={"dataset_name": r.dataset_name,
                                  "specimen_id": r.specimen_id,
                                  "defect_instance_id": r.defect_instance_id,
                                  "label_status": r.label_status,
                                  "defect_present": r.defect_present,
                                  "split_group": f"specimen:{r.specimen_id}"},
                        tensors={"bscan": arr[k], "env": envs[k]},
                    ))
                return out
        return [self.read_record(i) for i in indices]

    def read_by_specimen(self, specimen_id: str) -> list[NDTInstance]:
        """按 specimen（coupon）读取全部记录（含 tensor）。"""
        idx = [i for i, r in enumerate(self.records()) if r.specimen_id == specimen_id]
        return self.read_batch(idx, contiguous=True)

    def read_by_view(self, view: str = VIEW) -> list[NDTInstance]:
        """按 view 读取。当前 adapter 只含 90/G0 单视图；多视图见
        ``read_multiview``。未知 view 直接报错避免误用。"""
        if view != VIEW:
            raise ValueError(
                f"view {view!r} unsupported; this adapter covers {VIEW!r} only")
        return self.read_batch(list(range(len(self.records()))), contiguous=True)

    def read_multiview(self, i: int):
        """读取多视图实例（若 data/processed/paut/ascans_mv.npy 存在）。

        返回 (bscan_mv (4,49,512), meta)。4 视图见 paut_preprocess_multiview.py：
        90/G0(71°) 270/G0(71°) 90/G1(47°) 270/G1(47°)。
        """
        mv = self.processed_dir / "ascans_mv.npy"
        if not mv.exists():
            return None
        if getattr(self, "_mv", None) is None:
            self._mv = StreamingNpyReader(mv)
        gi = self.records()[i].tensor_index
        return self._mv.read(gi)     # (4, 49, 512)

    # -- split ----------------------------------------------------------
    def split_indices(self, protocol: str, val_ratio: float = 0.2, seed: int = 42):
        if protocol == "specimen":
            # 规范 split：PP3-5 / PP6 / PP7（仓库既有口径，训练/评测一致性）
            split = {"train": [], "val": [], "test": []}
            for part, coupons in CANONICAL_SPLIT.items():
                for c in coupons:
                    split[part].extend(
                        [i for i, r in enumerate(self.records()) if r.specimen_id == c])
            return split
        if protocol == "record_random":
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

    def validate_specimen_split(self, split: dict[str, list[int]]) -> bool:
        """不变量：同一 specimen 的所有记录必须落在同一 split。"""
        for c in self._coupon_order:
            idx = [i for i, r in enumerate(self.records()) if r.specimen_id == c]
            parts = {p for p, arr in split.items() if any(i in arr for i in idx)}
            assert len(parts) == 1, f"specimen {c} spans {parts} (leak!)"
        return True


# ---------------------------------------------------------------------------
# manifest 生成入口
# ---------------------------------------------------------------------------
def build_penelope_manifest(out_dir: Path = MANIFEST_DIR,
                            write_parquet: bool = True) -> tuple[Path, Path]:
    """生成 PENELOPE dataset card + records.parquet + 校验文件清单。"""
    ad = PENELOPEAdapter()
    records = ad.records()

    # specimens
    per_coupon = ad.summary["per_coupon"]
    specimens = []
    for c in ad._coupon_order:
        pc = per_coupon[c]
        ndt = ad.raw_root / c / "2. ndt_data"
        nde = ndt / pc.get("nde", NDE_FILE.get(c, "?"))

        specimens.append({
            "specimen_id": c,
            "dataset_name": DATASET_NAME,
            "material": "SAW X-joint steel weld (AIMEN)",
            "manufacturing": "submerged arc welding (SAW)",
            "geometry": "weld coupon ~600mm scan line",
            "source_file": str(nde.relative_to(ad.raw_root)) if nde.exists() else None,
        })

    # defects（每 coupon 每标注行）
    defects = []
    for c in ad._coupon_order:
        for d in ad._defect_instances(c):
            defects.append({
                "defect_instance_id": d["defect_instance_id"],
                "specimen_id": c,
                "defect_type": d["defect_type"],
                "data_origin": "measured",
                "defect_origin": "manufacturing",
                "defect_location": {"x": d["x_init_mm"], "coordinate_system": "coupon_local"},
                "defect_size": {"length": d["length_mm"]},
                "label_source": "defects_xlocation.xlsx",
            })

    # tensors
    tensors = [
        {"key": "ascans", "path": "paut/ascans.npy", "format": "npy",
         "axes": ["n_records", "beam", "time"], "dtype": "float32",
         "unit": "rectified_amplitude", "n_records": len(records)},
        {"key": "env", "path": "paut/env.npy", "format": "npy",
         "axes": ["n_records", "time"], "dtype": "float32",
         "unit": "rectified_amplitude_max_over_beams", "n_records": len(records)},
    ]

    # provenance / source 校验
    src_nde = ad.raw_root / "PP3" / "2. ndt_data" / "PAUT_90.nde"
    src_checksum = checksum_file(src_nde) if src_nde.exists() else None
    n_pos = sum(int(pc["n_pos"]) for pc in per_coupon.values())
    n_def = sum(int(pc["defect_pos"]) for pc in per_coupon.values())

    card_path = write_dataset_card(
        dataset_name=DATASET_NAME,
        primary_modality="ultrasonic",
        license_=LICENSE,
        source={
            "official_name": "PENELOPE / Submerged Arc Welding Open Repository",
            "url": "https://doi.org/10.5281/zenodo.15083865",
            "size_bytes": 12_679_424_288,
            "commit": "n/a (Zenodo zip)",
            "notes": "ZENODO_Penelope_vs2.zip, 12.7 GB, MD5 verified at download",
            "raw_root": "data/raw/saw/ZENODO_Penelope",
        },
        n_specimens=len(specimens),
        n_defect_instances=len(defects),
        n_records=len(records),
        specimens=specimens,
        defects=defects,
        tensors=tensors,
        out_dir=out_dir,
        extra={
            "data_policy": {
                "label_policy": "localized defects (< 50mm) only; blanket defects treated as background",
                "view": VIEW,
                "group": GROUP,
                "beam_angle_deg": BEAM_ANGLE_DEG,
                "n_beams": N_BEAMS,
                "depth_samples": N_DEPTH,
                "raw_samples": 3500,
                "per_coupon_counts": {
                    c: {"n_pos": int(per_coupon[c]["n_pos"]),
                        "n_defect_pos": int(per_coupon[c]["defect_pos"]),
                        "n_clean_pos": int(per_coupon[c]["n_pos"] - per_coupon[c]["defect_pos"]),
                        "defect_rate": round(float(per_coupon[c]["defect_rate"]), 4)}
                    for c in per_coupon},
                "canonical_split": CANONICAL_SPLIT,
                "raw_nde_checksum": src_checksum,
            },
            "provenance": {
                "steps": [
                    "read .nde DataGroup 0 amplitude (n_pos, 49, 3500) int16",
                    "max-pool depth 3500 -> 512 (rectified envelope keeps echo peaks)",
                    "defects_xlocation.xlsx -> per-position localized-defect label (<50mm)",
                ],
                "software": "scripts/paut_preprocess.py",
            },
        },
    )

    if write_parquet:
        rec_path = write_records_parquet(
            records, out_dir, LICENSE,
            source_file="data/processed/paut/*.npy + raw .nde")
    else:
        rec_path = out_dir / "records.parquet"

    print(f"[penelope] card : {card_path}")
    print(f"[penelope] recs : {rec_path}  ({len(records)} records)")
    print(f"[penelope] specimens={len(specimens)} defect_instances={len(defects)} "
          f"records={len(records)} (defect positions {n_def}/{n_pos})")
    return card_path, rec_path


if __name__ == "__main__":
    build_penelope_manifest()
