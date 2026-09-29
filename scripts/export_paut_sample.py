#!/usr/bin/env python3
"""Export 1-2 anonymized PAUT positions as shareable sample waveforms.

Each exported position is the 49-beam x 512-sample downsampled A-scan volume
(max-pool 3500 -> 512, rectified envelope; identical to the preprocessing in
scripts/paut_preprocess.py) plus its specimen / scan-position / defect-label
metadata.  Coupon IDs are anonymized (specimen-A/specimen-B); the mapping to
the real coupon IDs is deliberately withheld here.

Outputs (data/exports/paut_sample/):
  paut_sample_<specimen>_pos<idx>.npz   waveform (49,512) float32 + metadata
  paut_sample_<specimen>_pos<idx>.csv   same waveform, 49 rows x 512 cols
  metadata.json                         combined metadata for both positions
  README.md                             format description
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAUT = REPO / "data/processed/paut"
OUT = REPO / "data/exports/paut_sample"

DEFECT_CODES = {0: "clean", 1: "Porosity", 2: "Lack of fusion",
                3: "Slag inclusion", 4: "Metallic inclusion",
                5: "Projections", 6: "Cracks"}

# (global index in ascans.npy, anonymized specimen id)
PICKS = [
    (270, "specimen-A"),   # defect: Cracks (PP3 local pos 270)
    (2399, "specimen-B"),  # clean (PP7 local pos 0)
]


def main() -> None:
    ascans = np.load(PAUT / "ascans.npy", mmap_mode="r")
    lab = np.load(PAUT / "meta_label.npy")
    typ = np.load(PAUT / "meta_defect_type.npy")
    coup = np.load(PAUT / "meta_coupon.npy")
    pos = np.load(PAUT / "meta_pos.npy")
    summary = json.load(open(PAUT / "meta_summary.json"))

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for gidx, anon in PICKS:
        wave = np.asarray(ascans[gidx], dtype=np.float32)   # (49, 512)
        coupon = str(coup[gidx])
        info = summary["per_coupon"][coupon]
        # meta_pos stores the GLOBAL dataset index; recover the within-coupon
        # scan position so x_mm maps onto the coupon's scan axis.
        local_pos = int(gidx - np.nonzero(coup == coupon)[0][0])
        x_mm = info["offset_mm"] + local_pos * info["res_mm"]
        rec = {
            "sample_id": f"paut_sample_{anon}_pos{local_pos}",
            "specimen_id": anon,                 # anonymized; real ID withheld
            "scan_position_index": local_pos,
            "scan_position_mm": round(x_mm, 3),
            "defect_label": int(lab[gidx]),      # 1=localized defect, 0=clean
            "defect_type_code": int(typ[gidx]),
            "defect_type": DEFECT_CODES[int(typ[gidx])],
            "n_beams": int(wave.shape[0]),
            "n_samples": int(wave.shape[1]),
            "beam_angle_deg": 71,
            "acquisition": "Evident/Olympus OmniScan X3 phased-array UT, "
                           "rectified amplitude, max-pool downsampled "
                           "3500 -> 512 per beam",
            "units": "amplitude (raw int16-scale rectified envelope, float32)",
        }
        stem = OUT / rec["sample_id"]
        np.savez_compressed(stem.with_suffix(".npz"),
                            waveform=wave,
                            specimen_id=np.array(anon),
                            scan_position_index=np.int64(rec["scan_position_index"]),
                            scan_position_mm=np.float64(rec["scan_position_mm"]),
                            defect_label=np.int64(rec["defect_label"]),
                            defect_type_code=np.int64(rec["defect_type_code"]))
        # CSV: 49 rows (beams) x 512 cols (samples), header = sample index
        header = ",".join(str(i) for i in range(wave.shape[1]))
        np.savetxt(stem.with_suffix(".csv"), wave, delimiter=",",
                   fmt="%.6g", header=header, comments="")
        records.append(rec)
        print(f"exported {rec['sample_id']}  label={rec['defect_label']} "
              f"({rec['defect_type']})  x={rec['scan_position_mm']}mm")

    meta = {
        "description": "Anonymized PAUT weld-inspection sample waveforms "
                       "(49 beams x 512 samples per position)",
        "anonymization": "coupon IDs replaced by specimen-A/B; mapping to "
                         "real coupon IDs withheld",
        "positions": records,
    }
    with open(OUT / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    readme = """# PAUT 匿名样本波形(49 束 x 512 点)

从相控阵超声(Phased-Array UT)焊缝检测数据集中导出的 2 个匿名扫描位置样本,
每个位置为 71 度角、49 束波束的 A-scan 幅度体,逐束 3500 点 max-pool 降采样到
512 点(整流包络,max-pool 保留回波峰值)。

## 文件

| 文件 | 内容 |
|---|---|
| `paut_sample_<specimen>_pos<idx>.npz` | `waveform (49, 512) float32` + 元数据标量 |
| `paut_sample_<specimen>_pos<idx>.csv` | 同一波形,49 行(束) x 512 列(采样点),首行为采样点序号 |
| `metadata.json` | 两个位置的完整元数据 |

## 样本

| sample_id | 试件(匿名) | 扫描位置 | 缺陷标签 | 缺陷类型 |
|---|---|---|---|---|
| paut_sample_specimen-A_pos270 | specimen-A | idx 270 (350.0 mm) | 1 | Cracks(裂纹) |
| paut_sample_specimen-B_pos0 | specimen-B | idx 0 (80.0 mm) | 0 | clean |

- 试件 ID 已匿名(specimen-A/B),与真实试件的对应关系此处不提供。
- 缺陷标签口径:位置 x 范围与任一**局部化**缺陷(轴向长度 < 50 mm)重叠记 1,
  否则 0;贯穿全焊缝的整体缺陷按背景处理(与主实验预处理一致)。
- 幅度为原始整流幅度(float32,raw int16 量纲),未做归一化。

## 读取示例

```python
import numpy as np
d = np.load("paut_sample_specimen-A_pos270.npz")
wave = d["waveform"]          # (49, 512) float32
print(d["defect_label"])      # 1
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
