"""EddyCus-HDF5 (涡流 ECT, 跨模态目标域) loader。

数据: data/raw/EddyCus-HDF5/output/scan_*.h5 + data/manifests/eddycus/records.parquet
manifest 提供: specimen_id(配置组 148) / defect_type / split_group / source_file。
h5 提供: signal_data/f{1..4}/{real,imaginary} (I/Q 双通道 × 4 频率)。

最小表示 (v0): 每扫描 → (8, N) 1D 形态, 8 通道 = 4 频率 × (I, Q), N = 信号点数
(等间隔子采样到 max_points)。2D C-scan 栅格重建列为后续扩展 (保留 spatial 坐标元数据)。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from general_ndt.datasets.registry import register_dataset
from general_ndt.datasets.schema import GeneralNDTSample

DEFAULT_H5_DIR = "data/raw/EddyCus-HDF5/output"
DEFAULT_MANIFEST = "data/manifests/eddycus/records.parquet"
DEFAULT_FREQS = ("f1", "f2", "f3", "f4")


def _load_scan(path: Path, max_points: int | None = None) -> tuple[np.ndarray, int, int]:
    """读取一个扫描的 I/Q 双通道 × 4 频率 → (8, N) float32; 返回 (signal, N, n_freq)。"""
    import h5py

    with h5py.File(path, "r") as h:
        channels = []
        for f in DEFAULT_FREQS:
            g = h[f"signal_data/{f}"]
            real = g["real"][:]
            imag = g["imaginary"][:]
            channels.append(real)
            channels.append(imag)
        n = channels[0].shape[0]
        sig = np.stack(channels, axis=0).astype(np.float32)  # (8, N)
    if max_points and n > max_points:
        step = int(np.ceil(n / max_points))
        sig = sig[:, ::step]
        n = sig.shape[1]
    return sig, n, len(DEFAULT_FREQS)


@register_dataset("eddycus")
def load_eddycus(config: dict | None = None) -> list[GeneralNDTSample]:
    cfg = config or {}
    h5_dir = Path(cfg.get("h5_dir", DEFAULT_H5_DIR))
    manifest_path = Path(cfg.get("manifest", DEFAULT_MANIFEST))
    max_points = cfg.get("max_points", 2048)    # v0 子采样上限 (默认 2048 点)
    sample_limit = cfg.get("sample_limit", None)

    if not manifest_path.exists():
        raise FileNotFoundError(f"EddyCus manifest 未找到: {manifest_path} (先跑 eddycus adapter)")
    if not h5_dir.exists():
        raise FileNotFoundError(f"EddyCus h5 目录未找到: {h5_dir}")

    df = pd.read_parquet(manifest_path)
    if "eddy_current" not in df.columns:
        # 兼容: 有些 manifest 版本把频率/传感器信息放在 eddy_current 子对象
        pass
    records = df.to_dict("records")
    if sample_limit:
        records = records[:sample_limit]

    samples: list[GeneralNDTSample] = []
    for rec in records:
        src = rec.get("source_file")
        if not src or not Path(src).exists():
            continue
        try:
            sig, n, n_freq = _load_scan(Path(src), max_points=max_points)
        except Exception as exc:  # 个别 h5 损坏则跳过, 记录
            if cfg.get("strict", False):
                raise
            continue
        ed = rec.get("eddy_current") or {}
        freq = ed.get("frequency")
        freq = ed.get("frequency_mhz") if freq is None else freq
        freq_values = list(freq) if freq is not None else []
        samples.append(
            GeneralNDTSample(
                sample_id=str(rec.get("record_id")),
                signal=sig,                       # (8, N) = 1D 形态 (C,T), C=4freq×IQ
                shape_kind="1d",
                modality="eddy_current",
                specimen_id=str(rec.get("specimen_id")),
                sensor_id=ed.get("sensor_channel"),
                sampling_rate=1.0,                # 扫描索引 (名义)
                spatial_coordinates=None,
                label=int(bool(rec.get("defect_present"))),
                label_type=cfg.get("label_type", "binary"),
                defect_type=str(rec.get("defect_type")),
                split_group=str(rec.get("split_group")),
                metadata={
                    "dataset": "eddycus",
                    "license": "CC-BY-4.0",
                    "n_points": n,
                    "n_freq": n_freq,
                    "frequencies_mhz": freq_values,
                    "data_origin": str(rec.get("data_origin")),
                    "defect_origin": str(rec.get("defect_origin")),
                },
            )
        )
    return samples
