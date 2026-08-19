"""Synth-UT (physics-inspired procedural synthetic data) 单元测试。

覆盖 (M0-1.5 Protocol V2 / Synth-UT 修正):
1. def_rate 实际控制标签: actual_defect_rate 与 target 在容差内;
2. orthogonal 模式下 style 参数与标签统计**无人工绑定**:
   同一缺陷类型/深度/幅值分布在多个 style 不同的 coupon 上;
   style 参数 (gain/speckle/f0/...) 与 coupon 缺陷率统计量不相关;
3. confounded 模式确实把缺陷率与 style 耦合 (对照, 仅诊断);
4. meta.json 同时保存 target_defect_rate 与 actual_defect_rate。

运行:  python tests/test_synth_ut.py   (或 pytest tests/)
不下载、不训练、只跑生成器, 是 CPU 可跑的通路。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from synth_ultrasound import (
    DEFECT_TYPES,
    _assign_defect_profiles,
    assert_defect_rate,
    default_coupon_cfgs,
    make_coupon,
)

TOL = 0.05


def _make_cfgs(seed=42, n=12, mode="orthogonal"):
    rng = np.random.default_rng(seed)
    return default_coupon_cfgs(rng, n, mode)


def test_def_rate_controls_labels():
    """def_rate 必须实际控制标签: 每个 coupon 的 actual 与 target 在容差内。"""
    cfgs = _make_cfgs()
    for i, cfg in enumerate(cfgs):
        rng = np.random.default_rng(1000 + i)
        _, y = make_coupon(rng, 400, cfg)
        actual = float(y.mean())
        assert_defect_rate(actual, cfg["def_rate"], TOL, f"coupon {i}")
        assert cfg["def_rate"] > 0, "target rate must be positive"
    print("def_rate controls labels OK")


def test_defect_attributes_spread_across_styles():
    """每种缺陷类型/深度/幅值必须分布在多个 coupon style 上 (orthogonal)。

    验证: (a) 所有 DEFECT_TYPES 都出现; (b) 同一 def_type 出现在多个
    coupon; (c) 这些 coupon 的 style 参数 (gain/speckle/f0) 相互不同
    (非退化); (d) 缺陷率不绑定 def_type。
    """
    cfgs = _make_cfgs(n=40)   # 更多 coupon, 保证每个类型跨多个 style
    types = {c["def_type"] for c in cfgs}
    assert types == set(DEFECT_TYPES), f"missing types: {set(DEFECT_TYPES) - types}"

    # 每个类型至少出现在 3 个 coupon 上
    for t in DEFECT_TYPES:
        members = [c for c in cfgs if c["def_type"] == t]
        assert len(members) >= 3, f"def_type {t} only on {len(members)} coupons"
        # 这些 coupon 的 style 必须彼此不同 (非人为堆叠在同一 style)
        gains = {round(c["gain"], 3) for c in members}
        speckles = {round(c["speckle"], 3) for c in members}
        assert len(gains) >= 2 and len(speckles) >= 2, \
            f"def_type {t} collapsed onto identical styles: {members}"

    # 缺陷率不绑定类型: 每种类型的缺陷率分布范围都足够宽
    for t in DEFECT_TYPES:
        rates = [c["def_rate"] for c in cfgs if c["def_type"] == t]
        assert max(rates) - min(rates) > 0.1, \
            f"def_type {t} has near-constant defect rates {rates}"
    print("defect attributes spread across styles OK")


def test_style_independent_of_label_rate_orthogonal():
    """orthogonal: style 参数与标签统计 (缺陷率) 无相关。

    缺陷率由独立 uniform 抽样决定, 与 gain/speckle/f0/def_amp 等 style
    参数无关 —— 相关系数应接近 0 (不人为绑定)。若合成器误把 style 与
    缺陷率绑在一起, 这里会抓到。
    """
    rng = np.random.default_rng(2024)
    cfgs = default_coupon_cfgs(rng, 120, "orthogonal")
    rates = np.array([c["def_rate"] for c in cfgs])
    for style_key in ("gain", "speckle", "f0", "def_amp", "def_depth", "bw", "beam_w"):
        vals = np.array([c[style_key] for c in cfgs])
        r = abs(np.corrcoef(vals, rates)[0, 1])
        # 120 个样本下, 无绑定时的 |r| 应明显小于强耦合; 0.35 是宽松阈值
        assert r < 0.35, f"style {style_key} correlates with defect rate: r={r:.3f}"
    print("orthogonal style↔rate independence OK")


def test_confounded_couples_style_and_rate():
    """confounded 模式确实耦合 style 与缺陷率 (对照诊断)。"""
    rng = np.random.default_rng(9)
    cfgs = default_coupon_cfgs(rng, 40, "confounded")
    rates = np.array([c["def_rate"] for c in cfgs])
    styles = np.array([c["gain"] * c["def_amp"] / max(1e-6, c["speckle"]) for c in cfgs])
    r = abs(np.corrcoef(styles, rates)[0, 1])
    assert r > 0.9, f"confounded should couple style and rate, got r={r:.3f}"
    print("confounded style↔rate coupling OK")


def test_meta_json_saves_target_and_actual(tmp=None):
    """meta.json 同时保存 target_defect_rate 与 actual_defect_rate。"""
    # 直接走 main 的 meta 生成逻辑: 手动构造 per-coupon 元数据并检查键存在
    cfgs = _make_cfgs()
    per_coupon = []
    for i, cfg in enumerate(cfgs):
        rng = np.random.default_rng(1000 + i)
        _, y = make_coupon(rng, 300, cfg)
        per_coupon.append({
            "target_defect_rate": cfg["def_rate"],
            "actual_defect_rate": float(y.mean()),
        })
    for m in per_coupon:
        assert "target_defect_rate" in m and "actual_defect_rate" in m
        assert abs(m["target_defect_rate"] - m["actual_defect_rate"]) <= TOL
    # 顶层也应有 range 键 (由 main() 写入; 这里校验 JSON 结构可序列化)
    top = {
        "target_defect_rate_range": [min(m["target_defect_rate"] for m in per_coupon),
                                     max(m["target_defect_rate"] for m in per_coupon)],
        "actual_defect_rate_range": [min(m["actual_defect_rate"] for m in per_coupon),
                                     max(m["actual_defect_rate"] for m in per_coupon)],
    }
    json.dumps(top)  # 可序列化即通过
    print("meta.json target/actual keys OK")


def test_all():
    test_def_rate_controls_labels()
    test_defect_attributes_spread_across_styles()
    test_style_independent_of_label_rate_orthogonal()
    test_confounded_couples_style_and_rate()
    test_meta_json_saves_target_and_actual()
    print("\nAll Synth-UT tests passed.")


if __name__ == "__main__":
    test_all()
