"""external_weld_ut (Strathclyde 真实焊缝 FMC/PAUT, 无标签预训练语料) loader。

数据: data/processed/external_weld_ut/ (scripts/preprocess_external_weld_ut.py 产出)
  - signals.npy       (690, 16, 512) float32  1D 形态 (C=接收/位置子采样, T=时间)
  - manifest.parquet  sample_id / group_id (A/B/C/D 独立试件) / view_index / padded_time ...
  - meta_summary.json 审计引用 + 参数 + 逐组统计

定位 (F 系列): tier B 无标签预训练语料 (4 独立试件, 无逐位置标签) —— 仅预训练,
禁止任何下游评测 / 跨试件泛化 claim。与 PENELOPE 同为 ultrasonic 模态,
per_modality_stem 架构下共享 ultrasonic stem (有意设计: 外部数据直接塑造探针所用 stem)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from general_ndt.datasets.registry import register_dataset
from general_ndt.datasets.schema import GeneralNDTSample

DEFAULT_ROOT = "data/processed/external_weld_ut"


@register_dataset("external_weld_ut")
def load_external_weld_ut(config: dict | None = None) -> list[GeneralNDTSample]:
    cfg = config or {}
    root = Path(cfg.get("root", DEFAULT_ROOT))
    sample_limit = cfg.get("sample_limit", None)
    groups = list(cfg.get("groups", ["A", "B", "C", "D"]))

    sig_path = root / "signals.npy"
    if not sig_path.exists():
        raise FileNotFoundError(
            f"external_weld_ut processed 数据未找到: {root} (先跑 scripts/preprocess_external_weld_ut.py)")

    signals = np.load(sig_path, mmap_mode="r")
    manifest = pd.read_parquet(root / "manifest.parquet")
    summary = {}
    summary_path = root / "meta_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    df = manifest[manifest.group_id.isin(groups)].reset_index()
    if sample_limit:
        # 保组覆盖: 每组 ceil(limit / n_groups) 个 (同 penelope.py 模式)
        per = max(1, int(np.ceil(sample_limit / len(groups))))
        keep = []
        for g in groups:
            keep.extend(df.index[df.group_id == g][:per])
        df = df.loc[sorted(keep)].iloc[:sample_limit]

    samples: list[GeneralNDTSample] = []
    for _, row in df.iterrows():
        sig = np.asarray(signals[row["index"]])              # (16, 512) float32
        n_time_raw = int(row["n_time_raw"])
        valid_mask = None
        if bool(row["padded_time"]):
            # (C, T) 全形 mask (schema __post_init__ 要求同形; 时间 <401 无效)
            valid_mask = np.zeros(sig.shape, dtype=bool)
            valid_mask[:, :n_time_raw] = True
        samples.append(
            GeneralNDTSample(
                sample_id=str(row["sample_id"]),
                signal=sig,
                shape_kind="1d",
                modality="ultrasonic",
                specimen_id=str(row["group_id"]),
                sensor_id=f"{row['group_id']}_rxgrid",
                sampling_rate=1.0,                           # 名义时间索引, 非 Hz
                label=None,
                label_type="none",
                split_group=str(row["split_group"]),
                valid_mask=valid_mask,
                metadata={
                    "dataset": "external_weld_ut",
                    "license": "CC-BY-4.0",
                    "admission": "tier B 无标签预训练, 禁止评测 (4 独立试件)",
                    "source_file": str(row["source_file"]),
                    "view_index": int(row["view_index"]),
                    "n_time_raw": n_time_raw,
                    "summary": summary,
                },
            )
        )
    return samples
