#!/usr/bin/env python
"""General NDT Foundation — 正式 split manifest 生成 (Phase 2A Gate)。

对严格评测数据集生成**固定、可审计**的划分 manifest (按最小物理独立单元分组,
同 specimen/coupon 绝不跨 train/val/test), 输出 sample_id 级清单。

用法:
  python scripts/general_ndt_split_manifest.py [--dataset penelope_paut]
                                               [--exclude-specimens PP4]
                                               [--out artifacts/general_ndt/splits/...]

当前支持:
  - penelope_paut: coupon LOOCV (leave-one-coupon-out, 排除 PP4 用于主结果)
  - eddycus: 配置组 cross-config 划分 (exploratory; 非显式试件)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.registry import build_dataset                    # noqa: E402
from general_ndt.evaluation.probe import leave_one_specimen_split          # noqa: E402


def write_split_manifest(dataset: str, exclude: list[str], out: Path) -> dict:
    samples = build_dataset(dataset, {})
    if not samples:
        raise ValueError(f"数据集为空: {dataset}")
    specimens = sorted({s.specimen_id for s in samples if s.specimen_id})
    test_specs = [sp for sp in specimens if sp not in exclude]
    folds = leave_one_specimen_split(samples, test_specimens=test_specs)
    manifest = {
        "dataset": dataset,
        "protocol": "leave_one_specimen",
        "unit": "specimen_id",
        "excluded_specimens": exclude,
        "n_samples": len(samples),
        "n_specimens": len(specimens),
        "n_folds": len(folds),
        "folds": [],
    }
    for tr, va, te in folds:
        fold = {
            "test_specimens": sorted({samples[i].specimen_id for i in te}),
            "train_sample_ids": [samples[i].sample_id for i in tr],
            "val_sample_ids": [samples[i].sample_id for i in va],
            "test_sample_ids": [samples[i].sample_id for i in te],
        }
        # 校验: 同 specimen 不跨集合
        tr_sp = {samples[i].specimen_id for i in tr}
        va_sp = {samples[i].specimen_id for i in va}
        te_sp = {samples[i].specimen_id for i in te}
        assert not (tr_sp & te_sp) and not (va_sp & te_sp), f"泄漏: {tr_sp & te_sp}"
        manifest["folds"].append(fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="penelope_paut")
    ap.add_argument("--exclude-specimens", default="PP4")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exclude = [x for x in args.exclude_specimens.split(",") if x]
    out = args.out or (REPO / "artifacts" / "general_ndt" / "splits"
                       / f"{args.dataset}_loocv.json")
    m = write_split_manifest(args.dataset, exclude, out)
    print(f"[split-manifest] {args.dataset}: {m['n_folds']} folds "
          f"({m['n_samples']} samples, {m['n_specimens']} specimens, "
          f"exclude={m['excluded_specimens']})")
    for f in m["folds"]:
        print(f"  test={f['test_specimens']}: train={len(f['train_sample_ids'])} "
              f"val={len(f['val_sample_ids'])} test={len(f['test_sample_ids'])}")
    print(f"[split-manifest] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
