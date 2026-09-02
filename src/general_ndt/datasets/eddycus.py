"""EddyCus-HDF5 (涡流 ECT, 跨模态目标域) loader — Phase 2A 双表示。

数据: data/raw/EddyCus-HDF5/output/scan_*.h5 + data/manifests/eddycus/records.parquet
manifest 提供: specimen_id(**推断配置组, 非显式试件**) / defect_type / split_group / source_file。
h5 提供: signal_data/f{1..4}/{real,imaginary} (I/Q 双通道 × 4 频率) + spatial_data/
(track_number, sample_number, x_mm, y_mm, z_mm)。

两种表示 (Phase 2A, 禁止把 2048 点一维子采样当唯一正式表示):
A. exploratory_flat_1d (默认):
   - 每扫描 → (8, N) 1D 形态, 8 通道 = 4 频率 × (I, Q), N = 等间隔子采样到 max_points。
   - **仅用于工程 smoke**; 不支持 spatial-region claim。
B. native_grid_2d:
   - 按 track/sample 编号 scatter 恢复 H×W 栅格 (H=max_track, W=max_sample);
   - I/Q 与频率作通道 → (8, H, W); 空洞以 valid_mask 保留;
   - 支持 spatial-region masking。
   - 若 track/sample 不足以无歧义重建 (非 1 起始整数 / 重复坐标 / 缺失) → 报告阻塞,
     该文件跳过并记入 metadata 统计, **绝不伪造 reshape**。

⚠ 准入 (2026-09-02): specimen_id 是代码对 (material,fiber,layup,description,defect,thickness)
的哈希 → **148 为推断配置组, 非显式物理试件**。本 loader 只作无标签预训练 +
cross-sensor/cross-material 探索 (exploratory), 不得声称 cross-specimen 泛化。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from general_ndt.datasets.registry import register_dataset
from general_ndt.datasets.schema import GeneralNDTSample

DEFAULT_H5_DIR = "data/raw/EddyCus-HDF5/output"
DEFAULT_MANIFEST = "data/manifests/eddycus/records.parquet"
DEFAULT_FREQS = ("f1", "f2", "f3", "f4")
REPRESENTATIONS = ("exploratory_flat_1d", "native_grid_2d")


def normalize_sensor(s: str | None) -> str:
    """归一化 sensor_type: 去引号/去空格 (raw 有 `"S13131 ..."` 与 `6,1MHz` 格式化污染)。"""
    if not s:
        return ""
    return re.sub(r"\s+", "", str(s).replace('"', ""))


def _read_iq_channels(path: Path) -> np.ndarray:
    """读取 4 频率 × I/Q → (8, N) float64。"""
    import h5py

    with h5py.File(path, "r") as h:
        chans = []
        for f in DEFAULT_FREQS:
            g = h[f"signal_data/{f}"]
            chans.append(np.asarray(g["real"][:]))
            chans.append(np.asarray(g["imaginary"][:]))
    return np.stack(chans, axis=0)


def _load_flat_1d(path: Path, max_points: int | None = None) -> tuple[np.ndarray, int]:
    """A: (8, N) 等间隔子采样。"""
    sig = _read_iq_channels(path).astype(np.float32)
    n = sig.shape[1]
    if max_points and n > max_points:
        step = int(np.ceil(n / max_points))
        sig = sig[:, ::step]
        n = sig.shape[1]
    return sig, n


class BlockingError(Exception):
    """native_grid_2d 重建阻塞 (记录后跳过, 不伪造 reshape)。"""


def _load_native_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """B: scatter 重建 (8, H, W) + valid_mask (H, W)。

    返回 (signal, valid, info)。track/sample 不足以无歧义重建时抛 BlockingError
    (由调用方跳过并记录, 不伪造 reshape)。
    """
    import h5py

    with h5py.File(path, "r") as h:
        if "spatial_data" not in h:
            raise BlockingError(f"{path.name}: 缺少 spatial_data, 无法重建 2D 栅格")
        sp = h["spatial_data"]
        if "track_number" not in sp or "sample_number" not in sp:
            raise BlockingError(f"{path.name}: 缺少 track_number/sample_number")
        trk = np.asarray(sp["track_number"][()])
        smp = np.asarray(sp["sample_number"][()])
        sig = _read_iq_channels(path)  # (8, N)

    def _integral_1based(a: np.ndarray) -> bool:
        if a.size == 0 or a.dtype.kind not in "iuf":
            return False
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            return False
        if np.any(a < 1):
            return False
        return bool(np.all(np.abs(a - np.rint(a)) < 1e-9))

    if not (_integral_1based(trk) and _integral_1based(smp)):
        raise BlockingError(
            f"{path.name}: track/sample 非 1 起始整数, 无法无歧义重建 2D 栅格"
        )
    trk_i = np.rint(trk).astype(np.int64)
    smp_i = np.rint(smp).astype(np.int64)
    if trk_i.size != sig.shape[1]:
        raise BlockingError(f"{path.name}: track/sample 长度 {trk_i.size} != 信号长度 {sig.shape[1]}")
    pairs = set(zip(trk_i.tolist(), smp_i.tolist()))
    if len(pairs) != trk_i.size:
        raise BlockingError(f"{path.name}: 重复 (track,sample) 坐标, 无法无歧义重建")

    H, W = int(trk_i.max()), int(smp_i.max())
    sig2d = np.zeros((sig.shape[0], H, W), dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    sig2d[:, trk_i - 1, smp_i - 1] = sig
    valid[trk_i - 1, smp_i - 1] = True
    info = {"H": H, "W": W, "n_points": trk_i.size, "n_holes": int(H * W - trk_i.size)}
    return sig2d, valid, info


@register_dataset("eddycus")
def load_eddycus(config: dict | None = None) -> list[GeneralNDTSample]:
    cfg = config or {}
    h5_dir = Path(cfg.get("h5_dir", DEFAULT_H5_DIR))
    manifest_path = Path(cfg.get("manifest", DEFAULT_MANIFEST))
    max_points = cfg.get("max_points", 2048)         # 仅 exploratory_flat_1d 子采样上限
    representation = cfg.get("representation", "exploratory_flat_1d")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"未知 representation: {representation} (可选 {REPRESENTATIONS})")
    sample_limit = cfg.get("sample_limit", None)
    strict = cfg.get("strict", False)

    if not manifest_path.exists():
        raise FileNotFoundError(f"EddyCus manifest 未找到: {manifest_path} (先跑 eddycus adapter)")
    if not h5_dir.exists():
        raise FileNotFoundError(f"EddyCus h5 目录未找到: {h5_dir}")

    df = pd.read_parquet(manifest_path)
    records = df.to_dict("records")
    if sample_limit:
        records = records[:sample_limit]

    blocked = []
    samples: list[GeneralNDTSample] = []
    for rec in records:
        src = rec.get("source_file")
        if not src or not Path(src).exists():
            continue
        ed = rec.get("eddy_current") or {}
        freq = ed.get("frequency") if ed.get("frequency") is not None else ed.get("frequency_mhz")
        freq_values = list(freq) if freq is not None else []
        sensor_raw = ed.get("sensor_channel")
        sensor_norm = normalize_sensor(sensor_raw)
        try:
            if representation == "exploratory_flat_1d":
                sig, n = _load_flat_1d(Path(src), max_points=max_points)
                shape_kind = "1d"
                valid_mask = None
                grid_info = {"n_points": n}
            else:
                sig, valid_mask, grid_info = _load_native_grid(Path(src))
                shape_kind = "2d"
        except BlockingError as exc:
            blocked.append({"file": Path(src).name, "reason": str(exc)})
            if strict:
                raise
            continue
        except Exception as exc:
            blocked.append({"file": Path(src).name, "reason": f"读取失败: {exc}"})
            if strict:
                raise
            continue
        samples.append(
            GeneralNDTSample(
                sample_id=str(rec.get("record_id")),
                signal=sig,
                shape_kind=shape_kind,
                modality="eddy_current",
                specimen_id=str(rec.get("specimen_id")),   # 推断配置组, 非显式试件
                sensor_id=sensor_norm or sensor_raw,
                sampling_rate=1.0,
                spatial_coordinates=None,
                label=int(bool(rec.get("defect_present"))),
                label_type=cfg.get("label_type", "binary"),
                defect_type=str(rec.get("defect_type")),
                split_group=str(rec.get("split_group")),
                valid_mask=valid_mask,
                metadata={
                    "dataset": "eddycus",
                    "license": "CC-BY-4.0",
                    "representation": representation,
                    "n_freq": len(DEFAULT_FREQS),
                    "frequencies_mhz": freq_values,
                    "sensor_norm": sensor_norm,
                    "data_origin": str(rec.get("data_origin")),
                    "defect_origin": str(rec.get("defect_origin")),
                    "admission": "B/C pending admission; specimen_id = inferred config group",
                    **grid_info,
                },
            )
        )
    if blocked:
        samples_meta = samples[0].metadata if samples else {}
        samples_meta.setdefault("blocked_grid", []).extend(blocked)
    return samples
