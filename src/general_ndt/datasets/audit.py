"""最小数据审计脚本: 样本数 / specimen 数 / shape 分布 / 标签分布 / ID 重复 / 泄漏检查。

用法 (CLI):
    python -m general_ndt.datasets.audit penelope_paut eddycus [--sample-limit N]

或作为函数调用:
    stats = audit_samples(samples)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from general_ndt.datasets.registry import build_dataset
from general_ndt.datasets.schema import GeneralNDTSample


def audit_samples(samples: Sequence[GeneralNDTSample]) -> dict:
    """对样本列表计算统计量。返回 dict, 可直接打印/序列化。"""
    n = len(samples)
    specimens = sorted({s.specimen_id for s in samples if s.specimen_id})
    shapes = Counter((s.signal.shape, s.shape_kind) for s in samples)
    modalities = Counter(s.modality for s in samples)
    label_types = Counter(s.label_type for s in samples)
    label_values = Counter(s.label for s in samples if s.label is not None)
    defect_types = Counter(s.defect_type for s in samples if s.defect_type)
    sample_ids = [s.sample_id for s in samples]
    id_dup = len(sample_ids) - len(set(sample_ids))
    per_specimen = Counter(s.specimen_id for s in samples if s.specimen_id)
    # 同 specimen 的通道数不一致 (1D 形态) → 提示潜在 chunk/通道不齐
    chan_hetero = {}
    for sp in specimens:
        ch = {s.signal.shape[0] for s in samples if s.specimen_id == sp}
        chan_hetero[sp] = sorted(ch)

    return {
        "n_samples": n,
        "n_specimens": len(specimens),
        "specimens": specimens,
        "per_specimen": dict(per_specimen),
        "shape_distribution": {f"{k}": v for k, v in sorted(shapes.items())},
        "modalities": dict(modalities),
        "label_types": dict(label_types),
        "label_distribution": {str(k): v for k, v in label_values.most_common()},
        "defect_type_distribution": dict(defect_types.most_common()),
        "duplicate_sample_ids": id_dup,
        "per_specimen_channel_shapes": chan_hetero,
    }


def check_specimen_leak(
    samples: Sequence[GeneralNDTSample], train_ids: Iterable[str], test_ids: Iterable[str]
) -> list[str]:
    """泄漏检查: 同一 specimen_id 不得同时出现在 train 与 test。返回违规 specimen 列表。"""
    train_sp = {s.specimen_id for s in samples if s.sample_id in set(train_ids)}
    test_sp = {s.specimen_id for s in samples if s.sample_id in set(test_ids)}
    overlap = sorted(train_sp & test_sp)
    return overlap


def check_split_disjointness(
    samples: Sequence[GeneralNDTSample], split_of: dict[str, str]
) -> list[str]:
    """通用泄漏检查: 若一个 specimen 出现在 ≥2 个 split, 记违规。返回违规 specimen 列表。"""
    seen: dict[str, set[str]] = {}
    for s in samples:
        sp = s.specimen_id
        if sp is None:
            continue
        seen.setdefault(sp, set()).add(split_of.get(s.sample_id, "?"))
    return [sp for sp, splits in seen.items() if len(splits) > 1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="general_ndt 数据审计")
    ap.add_argument("datasets", nargs="+", help="数据集名 (registry 中的键)")
    ap.add_argument("--sample-limit", type=int, default=None, help="每数据集限制样本数")
    args = ap.parse_args(argv)

    ok = True
    for name in args.datasets:
        try:
            samples = build_dataset(name, {"sample_limit": args.sample_limit})
        except Exception as exc:
            print(f"[audit] {name}: 加载失败 — {exc}")
            ok = False
            continue
        stats = audit_samples(samples)
        print(f"===== {name} =====")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        # 泄漏预检: 有 split_group 且同组跨 split 需外部划分后检查; 这里报告 split_group 唯一性
        sg = Counter(s.split_group for s in samples)
        print(f"  [audit] split_group 种类: {len(sg)} (应≈独立单元数, 用于严格划分)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
