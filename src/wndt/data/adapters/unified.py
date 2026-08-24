"""M0-2A 统一数据读取层：把三个外部超声 adapter 收口到同一接口。

统一输出字段（每个 adapter 的 ``read_record(i)`` 都已具备）：
``tensor / dataset_name / specimen_id / defect_instance_id / acquisition_id /
data_origin / defect_origin / label_status / source geometry / axes/units /
domain metadata``。

本模块只做**接口收口与统计**，绝不把三种数据插值成同一二维图片 ——
各数据集的 stem（``dataset_stems.py``）负责把原生形状编成 token embedding。
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from wndt.data.adapters.base import NDTInstance      # noqa: E402
from wndt.data.adapters.eddycus import EddyCusAdapter  # noqa: E402
from wndt.data.adapters.ml_ndt import MLNDTAdapter    # noqa: E402
from wndt.data.adapters.ndt_ml_flaw import NDTMLFlawAdapter  # noqa: E402
from wndt.data.adapters.penelope import PENELOPEAdapter  # noqa: E402

ADAPTERS = {
    "penelope_paut": PENELOPEAdapter,
    "ml_ndt": MLNDTAdapter,
    "ndt_ml_flaw": NDTMLFlawAdapter,
    "eddycus": EddyCusAdapter,
}


def build_adapter(dataset_name: str, **kw):
    if dataset_name not in ADAPTERS:
        raise KeyError(f"unknown adapter {dataset_name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[dataset_name](**kw)


@dataclass
class UnifiedStats:
    """数据审计统计（一个数据集的独立单元数 / 标签分布 / 来源）。"""

    dataset_name: str
    n_records: int
    n_specimens: int
    n_defect_instances: int
    label_dist: Counter
    data_origin: Counter
    defect_origin: Counter
    records_per_specimen: dict[str, int]
    records_per_defect: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_records": self.n_records,
            "n_specimens": self.n_specimens,
            "n_defect_instances": self.n_defect_instances,
            "label_status": dict(self.label_dist),
            "data_origin": dict(self.data_origin),
            "defect_origin": dict(self.defect_origin),
            "records_per_specimen": self.records_per_specimen,
            "records_per_defect": self.records_per_defect,
        }


def stat_dataset(adapter) -> UnifiedStats:
    """统计一个 adapter 的独立单元 / 标签分布（不读取任何信号 tensor）。"""
    recs = adapter.records()
    label = Counter()
    dorig = Counter()
    deforig = Counter()
    spec = defaultdict(int)
    defe = defaultdict(int)
    n_def = set()
    for r in recs:
        label[r.label_status] += 1
        dorig[r.data_origin] += 1
        deforig[r.defect_origin] += 1
        spec[r.specimen_id] += 1
        if r.defect_instance_id:
            defe[r.defect_instance_id] += 1
            n_def.add(r.defect_instance_id)
    return UnifiedStats(
        dataset_name=adapter.dataset_name,
        n_records=len(recs),
        n_specimens=len(spec),
        n_defect_instances=len(n_def),
        label_dist=label,
        data_origin=dorig,
        defect_origin=deforig,
        records_per_specimen=dict(spec),
        records_per_defect=dict(defe),
    )


def read_random(adapter, n: int, seed: int = 0) -> list[NDTInstance]:
    """随机读取 ``n`` 条记录（含 tensor）——smoke/QA 用。

    对支持按批流式读取的 adapter（NDT_ML_Flaw），按批分组读取，避免对每个
    条带都完整解压一遍 6.88 GB 的批文件。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(adapter.records()), size=n, replace=False).tolist()
    if hasattr(adapter, "read_batch_strips"):
        # 按批分组：batch -> [全局索引...]
        groups: dict[str, list[int]] = {}
        recs = adapter.records()
        for i in idx:
            groups.setdefault(recs[i].acquisition_id, []).append(i)
        out: list[NDTInstance] = []
        for bid, gidx in groups.items():
            batch_records = [i for i, r in enumerate(recs) if r.acquisition_id == bid]
            pos_map = {i: pos for pos, i in enumerate(batch_records)}
            strips = adapter.read_batch_strips(bid, [pos_map[i] for i in gidx])
            for (pos, arr), gi in zip(strips, gidx):
                r = recs[gi]
                out.append(NDTInstance(
                    record_id=r.record_id,
                    metadata={
                        "dataset_name": r.dataset_name, "specimen_id": r.specimen_id,
                        "defect_instance_id": r.defect_instance_id,
                        "acquisition_id": r.acquisition_id,
                        "label_status": r.label_status, "defect_present": r.defect_present,
                        "data_origin": r.data_origin, "defect_origin": r.defect_origin,
                        "domain": r.domain, "geometry": r.geometry, "extra": r.extra,
                    },
                    tensors={"strip": arr},
                ))
        return out
    return [adapter.read_record(int(i)) for i in idx]


def tensor_report(instance: NDTInstance) -> dict[str, Any]:
    """单实例 tensor 质量报告：shape/dtype/范围/NaN/Inf。"""
    out = {}
    for k, v in instance.tensors.items():
        arr = np.asarray(v)
        out[k] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "mean": float(arr.mean()) if arr.size else None,
            "nan": int(np.isnan(arr).sum()),
            "inf": int(np.isinf(arr).sum()),
        }
    return out


def check_split_no_leak(adapter, protocol: str, unit_field: str = "defect_instance_id",
                        seed: int = 42) -> bool:
    """验证按物理单元划分后，同一单元不跨 split（防泄露）。"""
    split = adapter.split_indices(protocol, seed=seed)
    units: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(adapter.records()):
        u = getattr(r, unit_field) or f"clean:{r.dataset_name}"
        units[u].append(i)
    for u, idx in units.items():
        parts = {p for p, arr in split.items() if any(i in arr for i in idx)}
        assert len(parts) == 1, f"unit {u} spans splits {parts} (leak!)"
    return True
