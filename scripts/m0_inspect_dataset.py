#!/usr/bin/env python
"""M0-2A 数据集检查 / smoke：对任意已接入数据集做数据 QA。

用法：
    python scripts/m0_inspect_dataset.py penelope_paut [--n 32] [--seed 0]
    python scripts/m0_inspect_dataset.py ml_ndt        [--n 8]
    python scripts/m0_inspect_dataset.py ndt_ml_flaw   [--n 32]

输出：
1. 数据集统计（记录数 / 独立试件 / 独立缺陷 / 标签 / 来源分布）；
2. 随机读取 n 条记录：tensor shape / dtype / 范围 / NaN / Inf；
3. 一次 dataset-specific stem forward，输出统一 embedding shape；
4. 验证同一 specimen/flaw 不跨 split。

纯 CPU，不下载，不训练。NDT_ML_Flaw 走流式解压，不完整展开原始数据。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from wndt.data.adapters.unified import (
    build_adapter, stat_dataset, read_random, tensor_report, check_split_no_leak,
)
from wndt.models.multimodal.dataset_stems import build_dataset_stem


def run(dataset_name: str, n: int, seed: int) -> None:
    print(f"=== M0-2A inspect: {dataset_name} | n={n} seed={seed} ===")
    ad = build_adapter(dataset_name)

    # 1. 统计
    st = stat_dataset(ad)
    print(f"[stats] records={st.n_records} specimens={st.n_specimens} "
          f"defect_instances={st.n_defect_instances}")
    print(f"[stats] label_status={dict(st.label_dist)}")
    print(f"[stats] data_origin={dict(st.data_origin)}")
    print(f"[stats] defect_origin={dict(st.defect_origin)}")
    print(f"[stats] records_per_specimen={st.records_per_specimen}")

    # 2. 随机读取 + tensor QA（部分数据集下载中，placeholder 批量可能读失败）
    try:
        samples = read_random(ad, n, seed=seed)
    except (EOFError, FileNotFoundError) as e:
        print(f"[smoke] partial data — streaming failed: {e}")
        print(f"[stem] {dataset_name}: skipped (no readable data)")
        print("=== done (partial) ===")
        return
    shapes = {}
    for i, inst in enumerate(samples):
        rep = tensor_report(inst)
        for k, r in rep.items():
            shapes.setdefault(k, r["shape"])
            print(f"[sample {i:3d}] {k}: shape={r['shape']} dtype={r['dtype']} "
                  f"range=[{r['min']:.3g},{r['max']:.3g}] nan={r['nan']} inf={r['inf']}")
    print(f"[tensor] shapes seen: {shapes}")

    # 3. stem forward（取一条样本的前若干帧；体积数据集用 frame/volume stem）
    first_tensor = next(iter(samples[0].tensors.values()))
    stem_name = dataset_name
    if dataset_name == "ml_ndt":
        stem_name = "ml_ndt_volume" if first_tensor.ndim == 3 else "ml_ndt"
    stem = build_dataset_stem(stem_name)
    stem.eval()
    with torch.no_grad():
        x = torch.from_numpy(first_tensor.astype(np.float32)).unsqueeze(0)
        emb = stem(x)
    print(f"[stem] {stem_name}: input {tuple(x.shape)} -> embedding {tuple(emb.shape)}")

    # 4. split 不泄露校验（按 protocol 对应的物理单元）
    if dataset_name == "penelope_paut":
        protocol, unit_field = "specimen", "specimen_id"
    else:
        protocol, unit_field = "defect", "defect_instance_id"
    try:
        check_split_no_leak(ad, protocol, unit_field=unit_field, seed=seed)
        print(f"[split] {protocol} split: same-unit-not-across-splits OK")
    except Exception as e:
        print(f"[split] {protocol} split check: {type(e).__name__}: {e}")

    print(f"=== done: {dataset_name} ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["penelope_paut", "ml_ndt", "ndt_ml_flaw"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.n, args.seed)


if __name__ == "__main__":
    main()
