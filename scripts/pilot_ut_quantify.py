#!/usr/bin/env python3
"""Pilot-A: ECNDT2026 UT 数据量化（ndt_public_benchmark_pilot）。

官方 repo/Zenodo 只发布了原始 .m2k + 代码；`data/configs/inspection_info.json`
与 `data/configs/_annotations.coco.json`（所有标签来源）从未公开发布（repo 的
.gitignore 忽略了整个 data/，Zenodo 15115255 仅含 data.zip=m2k，归档
19410833 仅含 code.zip=repo 拷贝）。因此**严格复现在脚本 1/2 即被阻塞**。

本脚本做可复现的部分：
1. 遍历 58 个 m2k 记录轻量元数据（time_grid、gain 等）；
2. 抽样全量读取代表性文件，记录 ascan 形状（shots/角度/时间）、水程等；
3. 记录复现阻塞清单与代码审读发现的泄漏/协议问题。

输出: experiments/results/ndt_pilot/ut_ecndt2026_quantify.json
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UT = REPO / "data" / "raw" / "ut_ecndt2026_repo"
OUT = REPO / "experiments" / "results" / "ndt_pilot"
UT_PY = UT / ".venv_ut" / "bin" / "python"

CFG = {"freq_transd": 5, "bw_transd": 0.5, "tp_transd": "gaussian"}


def probe_file(path: Path, full: bool) -> dict:
    code = f"""
import json, numpy as np
from framework import file_m2k
cfg = {CFG!r}
d = file_m2k.read(r"{path}", read_ascan={"False" if not full else "True"}, **cfg)
ip = d.inspection_params
out = {{
  "time_grid_shape": list(np.asarray(d.time_grid).shape),
  "t_start_us": float(np.asarray(d.time_grid[0]).ravel()[0]),
  "dt_us": float((np.asarray(d.time_grid[1]) - np.asarray(d.time_grid[0])).ravel()[0]),
  "gain_hw_db": float(ip.gain_hw) if ip.gain_hw is not None else None,
  "water_path_mm": float(ip.water_path) if getattr(ip, "water_path", None) is not None else None,
  "n_angles": int(len(ip.angles)) if getattr(ip, "angles", None) is not None else None,
}}
if {str(full)}:
    ad = np.asarray(d.ascan_data)
    out["ascan_shape"] = list(ad.shape)
    out["ascan_dtype"] = str(ad.dtype)
print(json.dumps(out))
"""
    r = subprocess.run([str(UT_PY), "-c", code], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr else "unknown"}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    t0 = time.time()
    files = sorted(UT.glob("ut_data/data/*/*.m2k")) + sorted(UT.glob("ut_data/data/*/*/*.m2k"))
    entries = {}
    for p in files:
        rel = str(p.relative_to(UT))
        full = p.name in {"pit_inspection.m2k", "no_collimation.m2k",
                          "active_dir_single_element_v3.m2k", "2nd_row_v1.m2k"}
        entries[rel] = probe_file(p, full)
        print(rel, entries[rel], flush=True)

    summary = {
        "runtime_s": round(time.time() - t0, 1),
        "n_m2k_files": len(files),
        "folders": sorted({p.parent.name for p in files}),
        "files": entries,
        "reproduction_blockers": {
            "missing_configs": "data/configs/inspection_info.json 不存在于任何公开渠道 "
                               "(repo data/ 为空且被 .gitignore 忽略; Zenodo 15115255 "
                               "data.zip 仅含 m2k; 归档 19410833 code.zip 为 repo 拷贝)",
            "missing_annotations": "data/configs/_annotations.coco.json (COCO 缺陷掩码, "
                                   "唯一标签来源) 同样缺失 → 无法构建 contain_flaw 标签, "
                                   "脚本 2 起全部无法运行, 无法复现论文指标",
            "missing_pretrained": "REPRODUCING.md 声称可在 Zenodo 获取预训练模型从 "
                                  "6_test.py 开始 → 记录中无模型文件",
            "impact": "严格按作者 protocol 复现 不可行 (阻塞于脚本 1 读取 "
                      "inspection_info.json); paper-vs-repro 指标对照表无法建立",
        },
        "protocol_findings_from_code_reading": {
            "unit": "S-scan tile: 每次采集(shot) 的 S-scan (time×181角度) 划分为 19×9=171 个 tile",
            "label": "tile 级二值标签: COCO 掩码缩放 /5 后按 tile 内缺陷像素 >1% 判正",
            "split": "半监督: 全部 flaw tile 进 test 池, non-flaw tile 按 tile 随机 70/15/15 "
                     "(random_state=42) 分 train/val-test 池 → 同一 shot 的相邻 tile 同时出现 "
                     "在 train(normal) 与 test → 存在 specimen/scan 级相邻位置泄漏(设计使然)",
            "threshold_baseline_leak": "5_threshold_guess.py 的 max_tiles 阈值在整个 dataset "
                                       "(含 train) 上取每 shot 最大值 → 阈值基线自身泄漏",
            "lof": "pyod LOF novelty=True, n_neighbors=15, p=1, contamination=7%, "
                   "predict_with_rejection(T=32, delta=0.05); GridSearchCV 默认关闭 "
                   "(CV_GRIDSEARCH=False), 超参硬编码",
            "features": "tile 统计(6) + tiles_idx one-hot(170) + PCA50 + FFT 幅值统计(4)",
            "test_val_split": "test 池再随机 70/30 分 test/validation; validation 含 flaw "
                              "(contamination 打印), 仅用于 predict_with_rejection 一致性, 未用于调参",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ut_ecndt2026_quantify.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for v in entries.values() if "error" not in v)
    print(f"\nprobed ok {ok}/{len(files)}; full-read: "
          f"{[k for k, v in entries.items() if v.get('ascan_shape')]}")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
