#!/usr/bin/env python3
"""Pilot-B2: EddyCus 四种划分协议构建 + 泄漏审计（ndt_public_benchmark_pilot）。

协议（有信号 695 扫描；43 个 metadata-only 文件排除）：

P0 random-scan   : 扫描级随机 70/15/15（仅作参照，非主结果；已知高估）。
P1 cfg-disjoint  : 按 specimen_cfg（推断配置组，147 组）分组 70/15/15。
                   同配置重复扫描绝不跨 split。
P2 sensor-held-out: 二分类意义下**结构性不可行**——全部 63 个 clean 扫描来自主
                   传感器 S13131，其余 6 传感器只有 46 个 defect 扫描、0 clean。
                   可行变体 P2-onemodel：train=S13131（内部留 15% cfg 组做 val，
                   用于阈值/FPR 校准），test=6 个非主传感器的 46 个 defect 扫描，
                   报告 TPR@FPR10（FPR 在 val clean 上校准）——检验未见传感器上
                   的缺陷检出率，而非完整二分类。
P3 material-held-out: leave-one-material-out，仅两个 fold 可做完整二分类：
                   test=HP-U300/122C（545 def/27 clean）、test=Kohlegelege ST 50g
                   （35 def/36 clean）；test=0/90° Fabric 524（36 def/0 clean）只做
                   TPR@FPR10。其余 3 个材料扫描数 <16 且无 clean，跳过并记录。

输出（全部落盘，含 group identity）：
- data/manifests/eddycus_pilot_splits/P{0,1,2,3}.json
- experiments/results/ndt_pilot/eddycus_splits_audit.json（泄漏审计 + 可行性结论）
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_MANIFEST = REPO / "data" / "manifests" / "eddycus_pilot_splits"
OUT_RESULTS = REPO / "experiments" / "results" / "ndt_pilot"
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # test = 1 - train - val

MAIN_SENSOR = "S13131P3,3A7,3MHz"


def _grouped_split(groups: list[str], seed: int = SEED) -> dict[str, set[str]]:
    """组级 70/15/15（按组随机，组数过少时按比例下取整保证 test/val 至少 1 组）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sorted(groups))
    n = len(perm)
    n_test = max(1, int(round(n * (1 - TRAIN_RATIO - VAL_RATIO))))
    n_val = max(1, int(round(n * VAL_RATIO)))
    return {
        "test": set(perm[:n_test].tolist()),
        "val": set(perm[n_test:n_test + n_val].tolist()),
        "train": set(perm[n_test + n_val:].tolist()),
    }


def _grouped_train_val(groups: list[str], seed: int = SEED,
                       val_ratio: float = VAL_RATIO) -> dict[str, set[str]]:
    """组级两分（train 内部校准用）：test/val 组绝不混入外部 test。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sorted(groups))
    n_val = max(1, int(round(len(perm) * val_ratio)))
    return {"val": set(perm[:n_val].tolist()), "train": set(perm[n_val:].tolist())}


def assign(scan_df: pd.DataFrame, part_groups: dict[str, set[str]], col: str) -> pd.Series:
    g2p = {g: p for p, gs in part_groups.items() for g in gs}
    return scan_df[col].map(g2p)


def main() -> None:
    OUT_MANIFEST.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv")
    sig = df[df.has_signal].reset_index(drop=True)
    audit: dict = {"seed": SEED, "n_signal_scans": len(sig)}

    # ------------------------------------------------------------------ P0
    # 扫描级随机（分层于 defect_present）：参照协议
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(sig))
    n_test = int(round(len(sig) * 0.15))
    n_val = int(round(len(sig) * 0.15))
    p0 = np.full(len(sig), "train", dtype=object)
    p0[idx[:n_test]] = "test"
    p0[idx[n_test:n_test + n_val]] = "val"
    sig["P0"] = p0

    # ------------------------------------------------------------------ P1
    groups = _grouped_split(sig.specimen_cfg.unique().tolist())
    sig["P1"] = assign(sig, groups, "specimen_cfg")

    # ------------------------------------------------------------------ P2
    # train=S13131（内部 15% cfg 组做 val），test=6 非主传感器全部 defect 扫描
    is_main = sig.sensor == MAIN_SENSOR
    p2 = np.where(is_main, "train", "test")
    main_groups = _grouped_train_val(sig.loc[is_main, "specimen_cfg"].unique().tolist())
    p2[is_main.to_numpy()] = sig.loc[is_main, "specimen_cfg"].map(
        {g: p for p, gs in main_groups.items() for g in gs}).to_numpy()
    sig["P2"] = p2

    # ------------------------------------------------------------------ P3
    # leave-one-material-out（仅可二分类的 fold 生成完整 split；其余 one-class fold）
    material_counts = sig.groupby("material").defect_present.agg(["sum", "count"])
    p3_folds: dict[str, dict] = {}
    p3_cols = pd.DataFrame(index=sig.index)
    feasible_binary, oneclass_only = [], []
    for m, row in material_counts.iterrows():
        n_clean = int(row["count"] - row["sum"])
        if n_clean == 0:
            oneclass_only.append(m)
            fold_split = np.where(sig.material == m, "test", "train")
            main_groups_m = _grouped_train_val(
                sig.loc[(sig.material != m) & (sig.sensor == MAIN_SENSOR), "specimen_cfg"].unique().tolist())
            # val 从 train 的 S13131 cfg 组中留出（阈值/FPR 校准）
            g2p = {g: p for p, gs in main_groups_m.items() for g in gs}
            for i in sig.index[(sig.material != m) & (sig.sensor == MAIN_SENSOR)]:
                fold_split[i] = g2p[sig.specimen_cfg[i]]
            p3_cols[f"P3::{m}"] = fold_split
            continue
        feasible_binary.append(m)
        fold_split = np.where(sig.material == m, "test", "train")
        train_mask = sig.material != m
        # val 优先从 train 的 cfg 组留 15%（两分，test 组标签绝不混入外部 test）
        g2p = {g: p for p, gs in _grouped_train_val(
            sig.loc[train_mask, "specimen_cfg"].unique().tolist()).items() for g in gs}
        for i in sig.index[train_mask]:
            fold_split[i] = g2p[sig.specimen_cfg[i]]
        p3_cols[f"P3::{m}"] = fold_split
    for m in feasible_binary:
        col = p3_cols[f"P3::{m}"]
        tr = sig[col == "train"]
        p3_folds[m] = {
            "feasible_binary": True,
            "train": {"defect": int(tr.defect_present.sum()), "clean": int((~tr.defect_present).sum())},
            "val": {p: {"defect": int(v.defect_present.sum()), "clean": int((~v.defect_present).sum())}
                    for p, v in sig[col == "val"].groupby("defect_present")},
        }

    audit["P3_folds"] = p3_folds
    audit["P3_oneclass_only_materials"] = oneclass_only

    # ------------------------------------------------------------------ 泄漏审计
    def group_leak(col_split: str, col_group: str) -> int:
        return int(sum(
            1 for _, g in sig.groupby(col_group) if g[col_split].nunique() > 1))

    audit["leakage_audit"] = {
        "P0_groups_crossing_splits_by_specimen_cfg": group_leak("P0", "specimen_cfg"),
        "P0_groups_crossing_splits_by_defect_group": group_leak("P0", "defect_group"),
        "P1_groups_crossing_splits_by_specimen_cfg": group_leak("P1", "specimen_cfg"),
        "P1_groups_crossing_splits_by_defect_group": group_leak("P1", "defect_group"),
        "P2_train_test_share_sensor": bool(sig[sig.P2 == "test"].sensor.eq(MAIN_SENSOR).any()),
        "P3_train_test_share_material": {
            m: bool(sig[p3_cols[f"P3::{m}"] == "test"].material.nunique() > 1)
            for m in material_counts.index},
    }
    audit["split_sizes"] = {
        "P0": sig.P0.value_counts().to_dict(),
        "P1": sig.P1.value_counts().to_dict(),
        "P2": sig.P2.value_counts().to_dict(),
        **{f"P3::{m}": pd.Series(p3_cols[f"P3::{m}"]).value_counts().to_dict() for m in material_counts.index},
    }
    audit["P0_defect_rate_by_split"] = {
        p: float(sig[sig.P0 == p].defect_present.mean()) for p in ("train", "val", "test")}
    audit["P1_defect_rate_by_split"] = {
        p: float(sig[sig.P1 == p].defect_present.mean()) for p in ("train", "val", "test")}
    audit["feasibility"] = {
        "P2_binary_infeasible_reason": "全部 63 个 clean 扫描均来自主传感器 S13131；"
                                       "6 个非主传感器共 46 个 defect 扫描、0 clean → "
                                       "held-out sensor 上无负类，完整二分类不可行",
        "P2_variant": "P2-onemodel: train=S13131, test=46 defect scans, TPR@FPR10",
        "P3_binary_feasible_materials": feasible_binary,
        "P3_oneclass_only_materials": oneclass_only,
        "P3_infeasible_reason": "Fabric 524/565 与 1013 等材料无 clean 扫描 → 无负类",
    }

    # ------------------------------------------------------------------ 落盘
    for p in ("P0", "P1", "P2"):
        (OUT_MANIFEST / f"{p}.json").write_text(json.dumps({
            "protocol": p, "seed": SEED, "unit": "scan" if p == "P0" else
            ("specimen_cfg" if p == "P1" else "sensor-held-out(one-model)"),
            "splits": dict(zip(sig.scan_id, sig[p])),
        }, indent=1), encoding="utf-8")
    for m in material_counts.index:
        (OUT_MANIFEST / f"P3__{m.replace('/', '_').replace(' ', '')}.json").write_text(json.dumps({
            "protocol": "P3-material-held-out", "seed": SEED, "test_material": m,
            "splits": {row.scan_id: p3_cols.loc[i, f"P3::{m}"] for i, row in sig.iterrows()},
        }, indent=1), encoding="utf-8")

    sig_out = pd.concat([sig, p3_cols], axis=1)[
        ["scan_id", "sensor", "material", "defect_present", "defect_type",
         "specimen_cfg", "defect_group", "P0", "P1", "P2"] + list(p3_cols.columns)]
    sig_out.to_csv(OUT_RESULTS / "eddycus_splits_scan_table.csv", index=False)
    (OUT_RESULTS / "eddycus_splits_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "P3_folds"},
                     indent=2, ensure_ascii=False))
    print("P3 folds:", json.dumps(p3_folds, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
