"""M0-3 真实焊缝多源超声数据适配器（Strathclyde FMC/PAUT，CC BY 4.0）。

数据源（详见 ``data/manifests/external_weld_ut/download_manifest.json`` 与
``docs/M0_3_external_weld_ut_audit.md``）：

- A ``Lack_of_fusion_FMC_DORT_2016.mat``  : 316L 焊缝 lack-of-fusion FMC
- B ``FMC_2012_04_26_at_16_16.mat``       : Inconel 82/182 中心线裂纹 FMC
- C ``FMC_RR3_2_25MHz_3mmsdh.mat``       : 304SS MMA 焊缝 3mm SDH FMC
- D ``PAUT.zip``                          : 相控阵探头定位图像（zip 内格式待审计）

**独立性纪律（M0-3 §三）**：
- 每个 .mat 一般只含**一个物理试件**的若干采集；Tx×Rx、scan position、
  重复扫查**必须共享 group_id**，禁止当独立试件；
- ``specimen_id`` 只按真正独立的物理试件 / 采集配置分配；
- 若全部真实独立焊缝试件 < 10，dataset card 与报告必须标注
  ``exploratory external pretraining source``。

**表示（M0-3 §四）**：
- FMC 保留 **Rx × time** 二维物理结构：每个 transmit event（Tx 索引）= 1 个
  SSL view，view 继承原 specimen/group_id；不把 Tx×Rx×T 无解释 flatten 成
  普通独立样本；
- 变长输入：``read_view`` 返回 ``(1, Rx, T)`` + ``(Rx, T)`` valid mask，
  batch 内按 (Rx, T) bucket + padding，recon loss 只算 masked∩valid；
- PAUT 保持 beam/focal-law × time 二维表示（source D 审计后定）。

本适配器：
- 每条记录 = 1 个 transmit view（FMC）或 1 幅 PAUT 图像；record_id 携带
  group_id（= 物理试件/采集配置）；
- ``read_view(i)`` 懒加载第 i 个 view 的 (1,Rx,T) + valid；
- ``unit_keys(unit)`` 提供 specimen / defect / acquisition 分组键；
- ``build_manifest`` 生成 dataset_card.json + records.parquet。
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from wndt.data.adapters.base import (
    BaseNDTAdapter, ManifestField, NDTInstance, NDTModality,
)
from wndt.data.adapters.common import (
    REPO, RAW, UnifiedRecord, checksum_file, write_dataset_card,
    write_records_parquet,
)

DATASET_NAME = "external_weld_ut"
DATA_ROOT = RAW / "external_weld_ut"
MANIFEST_DIR = REPO / "data" / "manifests" / "external_weld_ut"
LICENSE = "CC BY 4.0"

# 数据源静态信息（与 download_manifest.json 一致；审计后按真实结构更新）
SOURCES = {
    "A": {"mat": "Lack_of_fusion_FMC_DORT_2016.mat", "doi": "10.15129/086404bd-eb69-429b-978c-2c35cdbfcf87",
          "material": "316L SS (austenitic weld)", "weld": "316L plate weld",
          "defect": "lack-of-fusion crack (50° to x-axis)",
          "layout": "FMC_new (T=10000, Tx=128, Rx=128) int32", "n_specimens": 1},
    "B": {"mat": "FMC_2012_04_26_at_16_16.mat", "doi": "10.15129/179e1b38-e701-443d-b995-a4449851330c",
          "material": "Inconel 82/182 + 316L + carbon steel", "weld": "Inconel 82/182 weld",
          "defect": "centreline vertical rough crack 12mm",
          "layout": "FMC_new (T=10000, Tx=45, Rx=45) int32", "n_specimens": 1},
    "C": {"mat": "FMC_RR3_2_25MHz_3mmsdh.mat", "doi": "10.15129/60b6a5b8-e78e-4742-8414-aaba9399a9c8",
          "material": "304SS", "weld": "MMA weld", "defect": "3mm side drilled hole",
          "layout": "fmc (Tx=128, Rx=128, T=976) float64", "n_specimens": 1},
    "D": {"mat": "PAUT.zip", "doi": "10.15129/bfb5a77d-dabe-4be4-82c9-b10e8c237dea",
          "material": "steel weld (unspecified)", "weld": "weld (probe localisation)",
          "defect": "none (probe localisation / weld material ID)",
          "layout": "389 × PAUT B-scan txt (time=762, beam=401)", "n_specimens": 1},
}


# ---------------------------------------------------------------------------
# MATLAB 读取（v5 / v7.3 自动，进程内按路径缓存）
# ---------------------------------------------------------------------------
_MAT_CACHE: dict[Path, dict[str, Any]] = {}


def load_mat(path: Path) -> dict[str, Any]:
    """读取 .mat -> 变量字典。v5 用 scipy；v7.3（HDF5）用 h5py 转 numpy。

    v7.3 数组转置（HDF5 默认行主序 vs MATLAB 列主序）并返回简单 dtype 数组；
    v5 的 cell/struct 递归展开为普通 ndarray（object）。进程内按路径缓存
    （同一 .mat 被多 view 复用时不重复解析）。
    """
    path = Path(path)
    if path in _MAT_CACHE:
        return _MAT_CACHE[path]
    try:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=False, struct_as_record=False)
        out = {k: v for k, v in m.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out: dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            def read(name, obj):
                if isinstance(obj, h5py.Dataset):
                    out[name.split("/")[-1]] = np.asarray(obj[...]).T
            f.visititems(read)
    _MAT_CACHE[path] = out
    return out


# ---------------------------------------------------------------------------
# FMC 数组检测（按真实数据审计结果：A/B 时间轴在 axis0，C 在 axis-1）
# ---------------------------------------------------------------------------
def _time_axis_smoothness(a: np.ndarray) -> float:
    """轴的时间平滑度：mean|diff|。时间轴相邻采样高度相关 -> 值小。"""
    a = a.astype(np.float64)
    s = tuple(min(d, 512) for d in a.shape)
    sub = a[tuple(slice(0, x) for x in s)]
    return float(np.abs(np.diff(sub, axis=0)).mean())


def canonical_fmc(fmc: np.ndarray) -> np.ndarray:
    """把任意 FMC 布局规范化为 ``(Tx, Rx, T)``（时间轴移到最后）。

    真实数据确认两种布局：
    - A/B ``FMC_new`` (T, Tx, Rx) = (10000, 128/45, 128/45) int32：时间在 axis0；
    - C ``fmc`` (Tx, Rx, T) = (128, 128, 976) float64：时间在 axis-1。
    时间轴判定：axis0 与 axis-1 中相邻采样更平滑者（mean|diff| 更小）为时间轴
    （超声 A-scan 时间序列平滑，而阵元/通道间不连续）。
    """
    fmc = np.asarray(fmc)
    if fmc.ndim == 2:
        return fmc[None]                          # (Rx, T) 单 transmit
    if fmc.ndim != 3:
        raise ValueError(f"FMC 数组必须是 2D/3D，got {fmc.shape}")
    d0 = _time_axis_smoothness(fmc)
    d1 = _time_axis_smoothness(np.moveaxis(fmc, -1, 0))
    if d0 <= d1:
        # 时间在 axis0：(T, Tx, Rx) -> moveaxis(0, -1) = (Tx, Rx, T)
        return np.moveaxis(fmc, 0, -1)
    return fmc                                   # 时间已在 axis-1：(Tx, Rx, T)


def detect_fmc_arrays(mat: dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 mat 变量中探测 (FMC 数组, time 向量)。

    - FMC 数组：三维数值数组（shape 任一大维 > 10）——真实数据为 (T,Tx,Rx)
      或 (Tx,Rx,T)，统一经 ``canonical_fmc`` 规范化为 (Tx,Rx,T)；
    - time 向量：1D（或 MATLAB 行/列向量 (1,N)/(N,1)）数值数组，候选名含
      time/t/dt 或长度 == FMC 时间轴长。真实数据 .mat 内无 time 向量（None）。
    """
    fmc_raw: Optional[np.ndarray] = None
    time_vec: Optional[np.ndarray] = None
    for k, v in mat.items():
        a = np.asarray(v)
        # 数值数组（float/complex/int；真实 A/B 为 int32 ADC 计数）
        if a.dtype.kind in "fci" and a.ndim == 3 and a.shape[-1] > 10:
            if fmc_raw is None or a.size > fmc_raw.size:
                fmc_raw = a
        # time 向量：1D 或 MATLAB 行向量 (1, N) / 列向量 (N, 1)
        flat = a.reshape(-1) if (a.ndim in (1, 2) and 1 in a.shape) else a
        if a.dtype.kind in "fci" and flat.ndim == 1 and flat.size > 10:
            tl = k.lower()
            if any(s in tl for s in ("time", "timestep", "t_axis", "t_vec")):
                if time_vec is None or "time" in tl:
                    time_vec = flat
    if fmc_raw is None:
        return None, time_vec
    return canonical_fmc(fmc_raw), time_vec


def group_id_for(source: str, mat: dict[str, Any],
                 fmc: np.ndarray) -> str:
    """物理试件/采集配置 -> group_id。

    **禁止把 Tx×Rx / scan position / 切片数当独立试件**：同一 .mat 文件里的
    FMC 采集属于同一物理试件（或同一次采集配置），group_id 必须按真正独立的
    物理单元分配。默认一个 .mat = 一个 group（审计确认后再细分）。
    """
    return f"{DATASET_NAME}:{source}:spec1"


# ---------------------------------------------------------------------------
# 视图索引
# ---------------------------------------------------------------------------
@dataclass
class WeldUTView:
    source: str                 # A/B/C/D
    tx: int                     # transmit event / 文件索引（view 维度）
    rx: int                     # 接收阵元数（FMC）或 beam 数（PAUT）
    t: int                      # 时间采样数
    group_id: str               # 物理试件/采集配置（split/group 键）
    record_id: str
    mat_relpath: str
    defect_type: str | None
    material: str


def _paut_txt_views(zip_path: Path) -> list[dict[str, Any]]:
    """PAUT.zip 内全部 .txt B-scan（每文件 = 1 个 view，time×beam -> beam×time）。

    所有 txt 网格同构（审计确认 762×401）；形状从第一个文件读取一次缓存。
    """
    import zipfile
    out: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".txt"))
        shape = None
        for name in names:
            stem = Path(name).stem                    # 如 A01
            letter, num = stem[0], int(stem[1:])
            if shape is None:
                raw = z.read(name).decode("utf-8", errors="replace")
                rows = [ln for ln in raw.split("\n") if ln.strip()]
                shape = (len(rows[0].split("\t")), len(rows))   # (beam, time)
            out.append({
                "name": name, "letter": letter, "num": num,
                "group_id": f"{DATASET_NAME}:D:spec1",
                "record_id": f"{DATASET_NAME}:D:{stem}",
                "rx": shape[0], "t": shape[1],
            })
    return out


def build_view_index(data_root: Path = DATA_ROOT,
                     sources: Sequence[str] = ("A", "B", "C", "D")) -> list[WeldUTView]:
    """每个 FMC transmit event / 每张 PAUT B-scan = 1 个 view。

    A/B/C 布局经 ``canonical_fmc`` 规范化为 (Tx, Rx, T)；D 为 (beam, time)。
    全部 view 继承物理试件的 group_id（一个 .mat / 一个 zip = 一个 group，
    审计确认后再细分）。
    """
    views: list[WeldUTView] = []
    for sid in sources:
        info = SOURCES[sid]
        p = data_root / sid / info["mat"]
        if not p.exists():
            continue
        gid = group_id_for(sid, {}, np.empty((0,)))
        if sid == "D":
            for e in _paut_txt_views(p):
                views.append(WeldUTView(
                    source=sid, tx=e["num"], rx=e["rx"], t=e["t"],
                    group_id=e["group_id"], record_id=e["record_id"],
                    mat_relpath=f"{sid}/{info['mat']}#{e['name']}",
                    defect_type=info["defect"], material=info["material"],
                ))
            continue
        mat = load_mat(p)
        fmc, _tv = detect_fmc_arrays(mat)
        if fmc is None:
            continue
        fmc = np.asarray(fmc)                        # (Tx, Rx, T)
        for tx in range(fmc.shape[0]):
            views.append(WeldUTView(
                source=sid, tx=tx, rx=int(fmc.shape[1]), t=int(fmc.shape[2]),
                group_id=gid,
                record_id=f"{DATASET_NAME}:{sid}:tx{tx:03d}",
                mat_relpath=f"{sid}/{info['mat']}",
                defect_type=info["defect"], material=info["material"],
            ))
    views.sort(key=lambda v: (v.source, v.tx))
    return views


def read_view(data_root: Path, v: WeldUTView) -> tuple[np.ndarray, np.ndarray]:
    """读取一个 view -> ``(1, Rx/T? , T) float32`` + ``(Rx, T) bool``。

    - FMC (A/B/C)：``(Rx, T)`` = canonical[tx]；
    - PAUT (D)：``(beam, time)`` = txt 网格转置（time×beam -> beam×time）。
    逐 view 独立 median/MAD robust 归一化（valid 像素上；真实数据全密，
    valid 恒 True）。
    """
    if v.source == "D":
        import zipfile
        zip_path, entry = v.mat_relpath.split("#", 1)
        with zipfile.ZipFile(data_root / zip_path) as z:
            raw = z.read(entry).decode("utf-8", errors="replace")
        rows = [ln for ln in raw.split("\n") if ln.strip()]
        a = np.array([list(map(float, ln.split("\t"))) for ln in rows],
                     dtype=np.float32)               # (time, beam)
        a = a.T                                     # (beam, time)
    else:
        mat = load_mat(data_root / v.mat_relpath)
        fmc, _tv = detect_fmc_arrays(mat)
        a = np.asarray(fmc)[v.tx].astype(np.float32)     # (Rx, T)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = 1.4826 * mad + 1e-6
    x = (a - med) / scale
    valid = np.ones(x.shape, dtype=bool)
    return x[None], valid


# ---------------------------------------------------------------------------
# BaseNDTAdapter 接口（供 manifest / 划分 / 测试复用）
# ---------------------------------------------------------------------------
class ExternalWeldUTAdapter(BaseNDTAdapter):
    """M0-3 外部真实焊缝 FMC 适配器（view 级；split 按 group_id）。"""

    def __init__(self, manifest_path: str | Path | None = None,
                 data_root: str | Path | None = None):
        super().__init__(manifest_path=manifest_path or MANIFEST_DIR / "dataset_card.json",
                         data_root=data_root or DATA_ROOT)
        self.dataset_name = DATASET_NAME
        self._views: Optional[list[WeldUTView]] = None

    def views(self) -> list[WeldUTView]:
        if self._views is None:
            self._views = build_view_index(Path(self.data_root))
        return self._views

    def load_manifest(self) -> list[NDTInstance]:
        return [
            NDTInstance(
                record_id=v.record_id,
                metadata={
                    "dataset_name": DATASET_NAME, "specimen_id": v.group_id,
                    "acquisition_id": f"{v.source}:{v.mat_relpath}",
                    "defect_instance_id": f"{v.source}:defect",
                    "defect_present": True,
                    "defect_type": v.defect_type,
                    "domain": {"material": v.material, "source": v.source},
                    "geometry": {"n_rx": v.rx, "n_t": v.t, "tx_index": v.tx},
                    "split_group": f"specimen:{v.group_id}",
                })
            for v in self.views()
        ]

    def __len__(self) -> int:
        return len(self.views())

    def unit_keys(self, unit: str = "specimen") -> list[str]:
        return [v.group_id for v in self.views()]

    def split_indices(self, protocol: str = "specimen", val_ratio: float = 0.2,
                      seed: int = 42) -> dict[str, list[int]]:
        from wndt.data.adapters.eddycus import split_by_unit_keys
        return split_by_unit_keys(self.unit_keys(), val_ratio=val_ratio, seed=seed)


# ---------------------------------------------------------------------------
# manifest 生成
# ---------------------------------------------------------------------------
def build_manifest(out_dir: Path = MANIFEST_DIR, data_root: Path = DATA_ROOT,
                   write_parquet: bool = True) -> Path:
    """写 dataset_card.json + records.parquet（只读结构，不逐 view 展开信号）。"""
    ad = ExternalWeldUTAdapter(data_root=data_root)
    views = ad.views()
    specimens: dict[str, dict] = {}
    for v in views:
        specimens.setdefault(v.group_id, {
            "specimen_id": v.group_id,
            "dataset_name": DATASET_NAME,
            "material": v.material,
            "manufacturing": SOURCES[v.source]["weld"],
            "geometry": f"FMC rx={v.rx} t={v.t}",
            "source_file": v.mat_relpath,
            "notes": "group_id = 物理试件/采集配置（一个 .mat = 一个 group，"
                     "审计确认后细分）；Tx×Rx/scan/重复扫查共享 group_id，禁止当独立试件",
        })
    n_indep = len(specimens)
    card = {
        "manifest_version": "0.2.0",
        "dataset_name": DATASET_NAME,
        "primary_modality": "ultrasonic",
        "license": LICENSE,
        "source": {
            "official_name": "Strathclyde weld FMC/PAUT datasets (A/B/C/D)",
            "url": "https://pureportal.strath.ac.uk/ (see download_manifest.json)",
            "doi": [SOURCES[s]["doi"] for s in SOURCES],
            "size_bytes": sum(p.stat().st_size for s in SOURCES
                              for p in [data_root / s / SOURCES[s]["mat"]]
                              if p.exists()),
            "downloadable": True,
            "audit_ref": "docs/M0_3_external_weld_ut_audit.md",
        },
        "n_specimens": n_indep,
        "n_defect_instances": len({v.defect_type for v in views}),
        "n_records": len(views),
        "specimens": list(specimens.values()),
        "defects": [],
        "tensors": [
            {"key": "view", "path": "raw/external_weld_ut/{A,B,C}/*.mat",
             "format": "matlab (v5 or v7.3)", "axes": ["rx", "time"],
             "dtype": "float", "unit": "voltage (ADC)",
             "n_records": len(views),
             "note": "每 transmit event = 1 view (Rx,T)；group_id 继承物理试件；"
                     "变长输入 + valid mask；recon loss 只算 masked∩valid"},
        ],
        "data_policy": {
            "specimen_count": n_indep,
            "view_is_not_specimen": True,
            "view_is_transmit_event": True,
            "split_by": "specimen_id (group_id)",
            "forbid": "Tx×Rx / scan position / 切片数当独立试件；缺陷标签用于 SSL 采样捷径",
            "exploratory": n_indep < 10,
            "exploratory_note": ("真实独立焊缝试件 < 10 -> exploratory external "
                                 "pretraining source，不称为 foundation-scale dataset")
            if n_indep < 10 else None,
            "audit_ref": "docs/M0_3_external_weld_ut_audit.md",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_parquet:
        rows = []
        for v in views:
            rows.append({
                "record_id": v.record_id, "dataset_name": DATASET_NAME,
                "modality": "ultrasonic", "specimen_id": v.group_id,
                "inspection_id": v.mat_relpath, "acquisition_id": f"{v.source}:{v.mat_relpath}",
                "defect_instance_id": f"{v.source}:defect",
                "defect_present": True, "label_status": "positive",
                "defect_type": v.defect_type, "data_origin": "measured",
                "defect_origin": "manufacturing",
                "license": LICENSE, "source_file": f"data/raw/external_weld_ut/{v.mat_relpath}",
                "ultrasonic": {"tensor_key": "view", "tensor_index": v.tx,
                               "scan_axis": "rx", "depth": "time",
                               "beam_angle": None, "n_rx": v.rx, "n_t": v.t,
                               "source": v.source, "material": v.material},
            })
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_dir / "records.parquet", index=False)
    return out_dir / "dataset_card.json"


if __name__ == "__main__":
    p = build_manifest()
    print(f"[{DATASET_NAME}] manifest written: {p}")
    ad = ExternalWeldUTAdapter()
    print(f"[{DATASET_NAME}] views={len(ad)} groups={len(ad.unit_keys('specimen'))}")
