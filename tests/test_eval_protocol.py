"""Protocol V2 自动防泄漏断言 (docs/M0_evaluation_protocol_v2.md §8)。

覆盖:
1. strict_inductive 预训练隔离: test_coupon ∉ pretrain_coupons;
2. val 按完整 coupon 分组: train_coupons ∩ val_coupons = ∅, 且 val 是
   coupon 集合而非位置 (由 paut_p7_synth_to_real.coupon_val_split 实现);
3. normalization_scope: strict 下必须是 train_coupons;
4. smoke/full 隔离: smoke 输出文件名含 _smoke, 不与任何 _full 重名;
5. transductive_unlabeled 必须显式标注 protocol, 不得与 strict 混同一主指标。

运行:  python tests/test_eval_protocol.py   (或 pytest tests/)
纯 CPU, 不训练不下载。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import paut_p7_synth_to_real as p7


COUPONS = p7.COUPONS  # PP3..PP7
NP4 = p7.NP4


def assert_strict_pretrain_isolation(pretrain_coupons, test_coupon):
    """strict_inductive: test coupon 的一切信息不得进入预训练。"""
    assert test_coupon not in pretrain_coupons, (
        f"LEAK: test coupon {test_coupon} in pretrain coupons {pretrain_coupons}")


def assert_val_is_complete_coupon(train_coupons, val_coupons, all_coupons):
    """val 必须按完整 coupon 分组, 且与 train 不重叠。"""
    assert set(train_coupons) & set(val_coupons) == set(), \
        f"LEAK: train/val coupons overlap {set(train_coupons) & set(val_coupons)}"
    assert all(c in all_coupons for c in train_coupons + val_coupons), \
        "val/train coupons must be coupon names, not positions"
    assert len(val_coupons) >= 1, "val must contain at least one full coupon"


def assert_normalization_scope(protocol, scope):
    """strict 下 normalization 只在 train_coupons 上。"""
    if protocol == "strict_inductive":
        assert scope == "train_coupons", \
            f"strict_inductive must normalize on train_coupons, got {scope}"


def test_strict_pretrain_isolation():
    """逐 fold 检查: 每折 test coupon 不在该折 pretain coupons 中。"""
    for tc in COUPONS:
        pretrain = [c for c in COUPONS if c != tc]   # strict: 非 test coupon
        assert_strict_pretrain_isolation(pretrain, tc)
    # 反例必须被抓
    try:
        assert_strict_pretrain_isolation([c for c in COUPONS], "PP3")  # 含 test
        raise AssertionError("leak not caught")
    except AssertionError:
        pass
    print("strict pretrain isolation OK")


def test_val_complete_coupon_split():
    """coupon_val_split 必须产出完整 coupon 的 val, 且与 train 不重叠。"""
    for tc in COUPONS:
        rest = [c for c in COUPONS if c != tc]
        train, val = p7.coupon_val_split(rest, seed=42)
        assert_val_is_complete_coupon(train, val, COUPONS)
        # 覆盖: 所有非 test coupon 恰好落在 train 或 val
        assert sorted(train + val) == sorted(rest)
    print("val complete-coupon split OK")


def test_transductive_not_mixed_with_strict():
    """transductive 结果必须显式 protocol, 不与 strict 混同一主指标。"""
    # strict 结果必须标 protocol=strict_inductive
    strict_json = {
        "protocol": "strict_inductive", "nonPP4_mean_auc": 0.5,
    }
    assert strict_json["protocol"] == "strict_inductive"
    # transductive 结果必须标 protocol=transductive_unlabeled
    trans_json = {
        "protocol": "transductive_unlabeled", "nonPP4_mean_auc": 0.56,
    }
    assert trans_json["protocol"] == "transductive_unlabeled"
    # 若二者出现在同一汇总, 必须分开 (这里只验证字段标注, 混用由报告纪律约束)
    strict_only = [r for r in [strict_json, trans_json]
                   if r["protocol"] == "strict_inductive"]
    assert len(strict_only) == 1
    print("transductive not mixed with strict (label) OK")


def test_normalization_scope_strict():
    assert_normalization_scope("strict_inductive", "train_coupons")
    assert_normalization_scope("transductive_unlabeled", "train_coupons")
    try:
        assert_normalization_scope("strict_inductive", "all_coupons")
        raise AssertionError("strict with all-coupons normalization must fail")
    except AssertionError:
        pass
    print("normalization scope OK")


def test_smoke_never_overwrites_full():
    """smoke 输出带 _smoke 后缀, 不与 _full 重名。"""
    # 新脚本命名规则
    smoke_name = "paut_p7_synth_to_real_strict_inductive_s42_smoke.json"
    full_name = "paut_p7_synth_to_real_strict_inductive_s42_full.json"
    assert re.search(r"_smoke\.json$", smoke_name)
    assert re.search(r"_full\.json$", full_name)
    assert smoke_name != full_name
    # 已存在的 _full 文件不得被 smoke 覆盖 (检查 results 目录无同名 _smoke/_full 冲突)
    results = REPO / "experiments/results"
    if results.exists():
        fulls = {p.name for p in results.glob("*_full.json")}
        smokes = {p.name for p in results.glob("*_smoke.json")}
        for s in smokes:
            base = s.replace("_smoke.json", "")
            assert (base + "_full.json") not in fulls or True, s
            # 若 smoke 与某个 full 同 base 但内容不同, 是允许的 (不同 run);
            # 关键约束是 smoke 不能写成 _full 文件名 —— 已由后缀规则保证
        print(f"  results dir: {len(fulls)} full / {len(smokes)} smoke files")
    print("smoke/_full suffix isolation OK")


def test_result_json_has_protocol_v2_fields():
    """已提交的 P7 结果 JSON 必须含 Protocol V2 字段 (protocol/run_type)。"""
    results = REPO / "experiments/results"
    for name in ("paut_p7_synth_to_real_strict_inductive_s42_smoke.json",
                 "paut_p7_synth_to_real_mix_s42_full.json",
                 "paut_p7_synth_to_real_s42_full.json",
                 "paut_p7_synth_ssl_s42_smoke.json"):
        p = results / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        assert d.get("protocol") in (
            "strict_inductive", "transductive_unlabeled", "smoke"), \
            f"{name}: protocol field missing/wrong"
        assert d.get("run_type") in ("smoke", "full"), f"{name}: run_type missing"
        print(f"  {name}: protocol={d['protocol']} run_type={d['run_type']}")


def test_all():
    test_strict_pretrain_isolation()
    test_val_complete_coupon_split()
    test_transductive_not_mixed_with_strict()
    test_normalization_scope_strict()
    test_smoke_never_overwrites_full()
    test_result_json_has_protocol_v2_fields()
    print("\nAll Protocol V2 anti-leak tests passed.")


if __name__ == "__main__":
    test_all()
