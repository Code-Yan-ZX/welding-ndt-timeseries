"""PENELOPE PAUT (超声, 目标域核心基准) loader。

数据: data/processed/paut/
  - ascans.npy        (3000, 49, 512) float32  49 波束 × 512 深度 (max-pool 降采样)
  - meta_coupon.npy   (3000,) <U8              coupon 试件 id (PP3..PP7)
  - meta_label.npy    (3000,) int64            0/1 (局部缺陷; ≥50mm 贯穿作背景)
  - meta_defect_type.npy  (3000,) int64        缺陷类型码
  - meta_pos.npy      (3000,) int64            轴向位置
划分: Protocol V2 coupon LOOCV (严格非PP4 逐折均值) — 由 evaluation/probe 处理。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from general_ndt.datasets.registry import register_dataset
from general_ndt.datasets.schema import GeneralNDTSample

DEFAULT_ROOT = "data/processed/paut"


def _load_meta(root: Path, name: str) -> np.ndarray:
    return np.load(root / name, mmap_mode="r")


@register_dataset("penelope_paut")
def load_penelope_paut(config: dict | None = None) -> list[GeneralNDTSample]:
    cfg = config or {}
    root = Path(cfg.get("root", DEFAULT_ROOT))
    sample_limit = cfg.get("sample_limit", None)  # smoke/审计用小样本
    label_type = cfg.get("label_type", "binary")

    if not (root / "ascans.npy").exists():
        raise FileNotFoundError(f"PENELOPE processed 数据未找到: {root} (先跑 preprocess 或检查路径)")

    ascans = np.load(root / "ascans.npy", mmap_mode="r")
    coupon = np.asarray(_load_meta(root, "meta_coupon.npy"))
    label = np.asarray(_load_meta(root, "meta_label.npy"))
    defect_type = np.asarray(_load_meta(root, "meta_defect_type.npy"))
    pos = np.asarray(_load_meta(root, "meta_pos.npy"))

    n = ascans.shape[0]
    idx = list(range(n))
    if sample_limit:
        # 保 coupon 覆盖, 每 coupon 取 ceil(limit / n_coupons) 个
        coupons = sorted(set(coupon))
        per = max(1, int(np.ceil(sample_limit / len(coupons))))
        idx = []
        for c in coupons:
            ci = [i for i in range(n) if coupon[i] == c][:per]
            idx.extend(ci)
        idx = idx[:sample_limit]

    summary = {}
    meta_summary_path = root / "meta_summary.json"
    if meta_summary_path.exists():
        summary = json.loads(meta_summary_path.read_text())

    samples: list[GeneralNDTSample] = []
    for i in idx:
        samples.append(
            GeneralNDTSample(
                sample_id=f"penelope_paut:{i}",
                signal=np.asarray(ascans[i]),        # (49, 512) = 1D 形态 (C,T)
                shape_kind="1d",
                modality="ultrasonic",
                specimen_id=str(coupon[i]),
                sensor_id="90deg_49beams",
                sampling_rate=1.0,                   # 深度索引 (名义), 非 Hz
                spatial_coordinates=np.asarray([float(pos[i])]),
                label=int(label[i]),
                label_type=label_type,
                defect_type=str(defect_type[i]) if defect_type[i] > 0 else "clean",
                split_group=f"coupon:{coupon[i]}",
                metadata={
                    "dataset": "penelope_paut",
                    "license": "CC-BY-4.0",
                    "beams": int(ascans[i].shape[0]),
                    "depth": int(ascans[i].shape[1]),
                    "summary": summary,
                },
            )
        )
    return samples
