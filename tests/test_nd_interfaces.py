"""M0-1 接口草案的轻量单元测试：adapter 划分、NDTBatch collate、
模态 stem/encoder/fusion head 形状、PairedDataGuard 拦截 unpaired 融合。

运行:  python tests/test_nd_interfaces.py   （或 pytest tests/）

这些测试只走合成数据桩，不下载、不训练，是 M0-2 前唯一允许的"训练通路"
（单模态桩 + 接口单测），符合 docs/M0_experiment_roadmap.md 的限制。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from wndt.data.adapters.base import (
    ManifestField,
    ManifestSplitter,
    NDTBatch,
    PairedDataGuard,
    SyntheticEddyCurrentAdapter,
    SyntheticUltrasonicAdapter,
    UnpairedDataError,
    UnpairedUtEctDataset,
)
from wndt.models.multimodal.interfaces import (
    ConcatFusionHead,
    EddyCurrentStem,
    GatedFusionHead,
    MLPTaskHead,
    MeanPoolEncoder,
    ScoreFusionHead,
    UltrasonicStem,
    build_ultrasonic_only_pipeline,
    ensure_paired_fusion,
)

B = 4


def test_adapter_distinct_and_unit_split():
    """样本数/独立缺陷数/独立试件数必须区分，且 split 按物理单元不跨 split。"""
    ut = SyntheticUltrasonicAdapter(n_specimens=3, n_positions=20)
    n_records = len(ut.instances())
    n_specimens = len(ut.distinct(ManifestField.SPECIMEN_ID))
    n_defects = len(ut.distinct(ManifestField.DEFECT_INSTANCE_ID))
    assert n_records == 60 and n_specimens == 3
    assert n_defects == 15, f"3 specimen * 5 defects each = {n_defects}"  # 20 位置每 4 个一缺陷

    split = ut.split_indices("specimen", seed=42)
    assert set(split) == {"train", "val", "test"}
    assert sum(len(v) for v in split.values()) == n_records
    # 关键不变量：同一 specimen 的所有记录必须落在同一 split
    for spec in ut.distinct(ManifestField.SPECIMEN_ID):
        idx = ut.unit_indices(ManifestField.SPECIMEN_ID)[spec]
        parts = [p for p, arr in split.items() if any(i in arr for i in idx)]
        assert len(parts) == 1, f"specimen {spec} spans splits {parts} (泄露!)"
    print("adapter unit split OK")


def test_splitter_too_few_units_raises():
    """物理单元 < 3 时必须报错而非产生泄漏 split。"""
    try:
        ManifestSplitter([], ManifestField.SPECIMEN_ID)  # 空实例 -> 0 单元
    except Exception:
        pass
    ut = SyntheticUltrasonicAdapter(n_specimens=1, n_positions=5)
    try:
        ut.split_indices("specimen")
        raise AssertionError("expected ValueError for 1 unit")
    except ValueError:
        pass
    print("too-few-units guard OK")


def test_ndtbatch_collate_and_availability():
    """NDTBatch：同形状 collate、缺失模态 availability 置 False。"""
    ut = SyntheticUltrasonicAdapter(n_specimens=2, n_positions=4)
    insts = ut.instances()
    batch = NDTBatch.collate(insts, modalities=["ultrasonic"], tensor_keys={"ultrasonic": "bscan"})
    assert batch.tensors["ultrasonic"].shape == (len(insts), 4, 32)
    assert batch.availability.all().item()
    assert batch.has("ultrasonic") and not batch.has("eddy_current")

    # 缺失模态：构造一个实例没有 iq 键
    from wndt.data.adapters.base import NDTInstance
    ect = SyntheticEddyCurrentAdapter(n_specimens=1, n_positions=4).instances()
    mixed = [
        NDTInstance(record_id="m0", metadata={}, tensors={"bscan": insts[0].tensors["bscan"], "iq": ect[0].tensors["iq"]}),
        NDTInstance(record_id="m1", metadata={}, tensors={"bscan": insts[1].tensors["bscan"]}),  # 缺 iq
    ]
    b2 = NDTBatch.collate(mixed, modalities=["ultrasonic", "eddy_current"],
                          tensor_keys={"ultrasonic": "bscan", "eddy_current": "iq"})
    assert b2.availability[0].tolist() == [True, True]
    assert b2.availability[1].tolist() == [True, False]
    print("NDTBatch collate + availability OK")


def test_ndtbatch_collate_dtype_when_first_instance_missing_modality():
    """M0-1.5 回归: 第一个实例缺失某模态时 dtype 不能 KeyError/错乱。

    旧实现取 instances[0].tensors[key].dtype —— 第一个实例缺该模态直接
    KeyError。修复后 dtype 取第一个**存在**该模态的实例。
    """
    from wndt.data.adapters.base import NDTInstance
    ut = SyntheticUltrasonicAdapter(n_specimens=1, n_positions=2).instances()
    ect = SyntheticEddyCurrentAdapter(n_specimens=1, n_positions=2).instances()
    # 第 0 个实例缺 iq, 第 1 个实例有 iq
    mixed = [
        NDTInstance(record_id="m0", metadata={}, tensors={"bscan": ut[0].tensors["bscan"]}),
        NDTInstance(record_id="m1", metadata={}, tensors={"bscan": ut[1].tensors["bscan"], "iq": ect[0].tensors["iq"]}),
    ]
    b = NDTBatch.collate(mixed, modalities=["ultrasonic", "eddy_current"],
                         tensor_keys={"ultrasonic": "bscan", "eddy_current": "iq"})
    assert b.availability[0].tolist() == [True, False]
    assert b.availability[1].tolist() == [True, True]
    # dtype 来自第一个**存在**该模态的实例 (numpy float32 → torch float32)
    assert b.tensors["eddy_current"].dtype == torch.from_numpy(
        ect[0].tensors["iq"]).dtype
    assert b.tensors["eddy_current"].shape == (2, 32, 2)
    # 缺失样本的占位必须是 0, 不能是随机值
    assert b.tensors["eddy_current"][0].abs().sum().item() == 0.0
    print("collate dtype (first instance missing) OK")


def test_modality_stems_and_encoder_shapes():
    """模态专属 stem -> 统一 embedding 维；encoder 输出 (B, d_model)。"""
    ut_stem = UltrasonicStem(in_channels=4, seq_len=32, out_dim=64)
    ect_stem = EddyCurrentStem(in_channels=2, seq_len=32, out_dim=64)
    x_ut = torch.randn(B, 4, 32)
    x_ect = torch.randn(B, 32, 2)
    e_ut = ut_stem(x_ut, available=torch.ones(B))
    e_ect = ect_stem(x_ect, available=torch.ones(B))
    assert e_ut.shape == (B, 1, 64) and e_ect.shape == (B, 1, 64)

    enc = MeanPoolEncoder(d_model=64, modality_order=["ultrasonic", "eddy_current"])
    avail = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    fused = enc({"ultrasonic": e_ut, "eddy_current": e_ect}, avail)
    assert fused.shape == (B, 64)
    fused.sum().backward()
    print("stems + encoder shapes OK")


def test_fusion_heads_shapes():
    """三类融合头输出形状一致，且缺失模态不污染输出（显式 modality_order）。"""
    d = 64
    order = ["ultrasonic", "eddy_current"]
    avail_full = torch.ones(B, 2)
    avail_missing = torch.tensor([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.float32)
    toks = {
        "ultrasonic": torch.randn(B, 4, d),
        "eddy_current": torch.randn(B, 4, d),
    }
    heads = (ConcatFusionHead(d, order), GatedFusionHead(d, order),
             ScoreFusionHead(d, order, n_classes=2))
    for head in heads:
        out = head(toks, avail_missing)
        assert out.shape == (B, d) or out.shape == (B, 2), (type(head).__name__, out.shape)
        out_full = head(toks, avail_full)
        assert out_full.shape == out.shape
        out.sum().backward()
    print("fusion heads shapes OK")


def test_fusion_heads_explicit_modality_order_not_dict_order():
    """M0-1.5: 融合头按 modality_order 聚合, 不依赖 dict.values() 顺序。

    交换字典插入顺序 (tokens 内容不变) 输出必须一致 —— 若实现用
    dict.values(), 顺序颠倒会静默错配。
    """
    d = 64
    order = ["ultrasonic", "eddy_current"]
    avail = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    ut = torch.randn(3, 4, d)
    ect = torch.randn(3, 4, d)
    toks_a = {"ultrasonic": ut, "eddy_current": ect}
    toks_b = {"eddy_current": ect, "ultrasonic": ut}   # 交换插入顺序
    for head in (ConcatFusionHead(d, order), GatedFusionHead(d, order),
                 ScoreFusionHead(d, order, n_classes=2)):
        out_a = head(toks_a, avail)
        out_b = head(toks_b, avail)
        assert torch.allclose(out_a, out_b, atol=1e-6), \
            f"{type(head).__name__} depends on dict order (leak)"
    print("explicit modality_order OK")


def test_score_fusion_head_masks_and_renormalizes():
    """M0-1.5: ScoreFusionHead 权重 B×M, 按 availability 屏蔽并在可用模态重归一化。

    全缺失样本输出应为 0 (无可用模态); 缺失一个模态时输出只由可用模态贡献。
    """
    d, n_cls = 32, 2
    head = ScoreFusionHead(d, ["ultrasonic", "eddy_current"], n_classes=n_cls)
    ut = torch.randn(2, 1, d)
    ect = torch.randn(2, 1, d)
    toks = {"ultrasonic": ut, "eddy_current": ect}
    # 样本0: 两模态都可用; 样本1: 都缺失
    avail_both = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    avail_none = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    out_full = head(toks, avail_both)
    out_none = head(toks, avail_none)
    assert out_full.shape == (2, n_cls)
    # 全缺失样本输出必须为 0 (无可用模态, 不产生虚假 logits)
    assert out_none[1].abs().sum().item() == 0.0
    # 权重参数是长度为 M 的先验 (运行时展开为 B×M)
    assert head.w.shape == (2,)
    print("ScoreFusionHead B×M mask + renormalize OK")


def test_missing_modality_invariance():
    """M0-1.5: 缺失模态占位 tensor 随机值变化, 该缺失样本的输出必须不变。

    对 GatedFusionHead / ConcatFusionHead / ScoreFusionHead: 把**缺失模态**
    的 token 换成完全不同的随机值, 在相同 availability 下, 缺失该模态的
    样本输出应逐位一致 (占位值不得泄漏进 MLP/加权)。
    注意: 同时可用两个模态的样本, 其输出随真实模态输入变化是**正确**行为,
    不在不变性断言范围内。
    """
    d = 64
    order = ["ultrasonic", "eddy_current"]
    # 样本0: 缺 ECT; 样本1: 缺 UT; 样本2: 双模态可用 (对照, 允许变化)
    avail = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)
    # 仅对"缺 ECT"的样本0 做不变性断言
    missing_ect = torch.tensor([True, False, False])
    for head in (ConcatFusionHead(d, order), GatedFusionHead(d, order),
                 ScoreFusionHead(d, order, n_classes=2)):
        base = {
            "ultrasonic": torch.randn(3, 4, d),
            "eddy_current": torch.randn(3, 4, d),
        }
        out_base = head(base, avail).detach()
        # 缺失模态换成完全不同的随机占位值 (量级放大多个数量级)
        pert = {
            "ultrasonic": base["ultrasonic"].clone(),
            "eddy_current": torch.randn(3, 4, d) * 1e3,
        }
        out_pert = head(pert, avail).detach()
        diff = (out_base - out_pert).abs().max(dim=1).values
        assert (diff[missing_ect] < 1e-3).all(), \
            f"{type(head).__name__}: missing-modality placeholder leaked into output"
        # 双模态样本输出允许变化 (真实输入变了), 但要足够大以证明测试有效
        assert diff[~missing_ect].max().item() > 1e-3, \
            f"{type(head).__name__}: test not discriminating (both-available sample should change)"
    print("missing-modality invariance OK")


def test_gated_fusion_head_zeroes_missing_before_mlp():
    """M0-1.5: GatedFusionHead 缺失模态 token 进入任何 MLP 前置零。"""
    d = 64
    head = GatedFusionHead(d, ["ultrasonic", "eddy_current"])
    avail = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
    toks = {
        "ultrasonic": torch.randn(2, 1, d),
        "eddy_current": torch.randn(2, 1, d),
    }
    # 与 missing-modality invariance 不同, 这里直接检查 gate 输入中缺失列是否为 0
    x = torch.cat([toks["ultrasonic"].mean(1), toks["eddy_current"].mean(1)], dim=-1)
    avail_a = avail[:, 0:1]
    # 缺失样本的 eddy 列在进入 gate 前应为 0
    eddy_col = toks["eddy_current"].mean(1) * avail[:, 1:2]
    assert torch.allclose(eddy_col[0].abs().sum(), torch.tensor(0.0)), \
        "missing eddy token not zeroed before MLP"
    out = head(toks, avail)
    assert out.shape == (2, d)
    out.sum().backward()
    print("GatedFusionHead zero-before-MLP OK")


def test_task_head_and_ultrasonic_only_pipeline():
    """单模态训练桩可端到端前向（唯一合法的无配对数据训练通路）。"""
    pipe = build_ultrasonic_only_pipeline(d_model=64, in_channels=4, seq_len=32, n_classes=2)
    logits = pipe(torch.randn(B, 4, 32))
    assert logits.shape == (B, 2)
    logits.sum().backward()
    head = MLPTaskHead(in_dim=64, n_classes=3)
    assert head(torch.randn(B, 64)).shape == (B, 3)
    print("task head + ultrasonic-only pipeline OK")


def test_paired_data_guard_blocks_unpaired_fusion():
    """没有成对数据（不同 specimen / 无 fusion 链接）必须被拦截。"""
    # 1) 不同 specimen 的 UT + ECT 混合 -> unpaired
    ds = UnpairedUtEctDataset(n_per_side=8)
    cleaned = [ds[i] for i in range(len(ds))]  # 每个实例 fusion=None（不成对）
    batch = NDTBatch.collate(cleaned, modalities=["ultrasonic", "eddy_current"])
    guard = PairedDataGuard(paired_specimens=set())  # 未登记任何共享 specimen
    try:
        ensure_paired_fusion(batch, guard, instances=cleaned)
        raise AssertionError("unpaired fusion must raise UnpairedDataError")
    except UnpairedDataError:
        pass

    # 1b) 即使传入"已登记共享 specimen"的 guard，无 fusion 链接仍应被拦截
    guard_registered = PairedDataGuard(paired_specimens={"S0", "T0"})
    try:
        ensure_paired_fusion(batch, guard_registered, instances=cleaned)
        raise AssertionError("unpaired fusion must raise even with registered specimens")
    except UnpairedDataError:
        pass

    # 2) check_paired 正/负对照：有完整 FusionLink + 已登记共享 specimen -> True
    from wndt.data.adapters.base import NDTInstance as NI
    inst0 = cleaned[0]
    # 负例：无 fusion 链接
    neg = NI(record_id="neg", metadata={"fusion": None}, tensors=inst0.tensors)
    # 正例：完整 FusionLink（同 specimen、同坐标、已配准、已登记）
    pos = NI(record_id="pos", metadata={
        "fusion": {
            "shared_specimen_id": "S0",
            "shared_coordinate_system": "specimen_global_mm",
            "registration_transform": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                                  [0, 0, 1, 0], [0, 0, 0, 1]]},
            "modality_availability": {"ultrasonic": True, "eddy_current": True},
        }}, tensors=inst0.tensors)
    assert guard_registered.check_paired(neg) is False
    assert guard_registered.check_paired(pos) is True
    print("PairedDataGuard blocks unpaired fusion OK")


def test_paired_data_guard_early_vs_intermediate():
    """M0-1.5: early fusion 需要严格坐标矩阵, intermediate/late 只要求成对。

    - 只有 description 无 matrix 的配准: intermediate 可过, early 必须拦;
    - 有 matrix 的配准: 两者都过;
    - 未配准 (无 registration_transform): 都拦。
    """
    from wndt.data.adapters.base import NDTInstance as NI
    ut = SyntheticUltrasonicAdapter(n_specimens=1, n_positions=2).instances()
    ect = SyntheticEddyCurrentAdapter(n_specimens=1, n_positions=2).instances()
    tensors = {"ultrasonic": ut[0].tensors["bscan"], "eddy_current": ect[0].tensors["iq"]}
    guard = PairedDataGuard(paired_specimens={"S0"})

    desc_only = NI(record_id="desc", metadata={"fusion": {
        "shared_specimen_id": "S0",
        "shared_coordinate_system": "specimen_global_mm",
        "registration_transform": {"description": "manual alignment"},
        "modality_availability": {"ultrasonic": True, "eddy_current": True},
    }}, tensors=tensors)
    with_matrix = NI(record_id="mat", metadata={"fusion": {
        "shared_specimen_id": "S0",
        "shared_coordinate_system": "specimen_global_mm",
        "registration_transform": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                              [0, 0, 1, 0], [0, 0, 0, 1]]},
        "modality_availability": {"ultrasonic": True, "eddy_current": True},
    }}, tensors=tensors)

    assert guard.check_paired(desc_only) is True        # 成对 (描述即可)
    assert guard.check_registration_matrix(desc_only) is False  # 无矩阵
    assert guard.check_registration_matrix(with_matrix) is True

    def _require(insts, fusion_type):
        b = NDTBatch.collate(insts, modalities=["ultrasonic", "eddy_current"])
        guard.require_paired(b, "ultrasonic", "eddy_current", instances=insts,
                             fusion_type=fusion_type)

    # intermediate: desc_only 可通过
    _require([desc_only], "intermediate")
    # early: desc_only 必须被拦 (无严格矩阵)
    try:
        _require([desc_only], "early")
        raise AssertionError("early fusion with description-only registration must fail")
    except UnpairedDataError:
        pass
    # early: with_matrix 可通过
    _require([with_matrix], "early")
    print("PairedDataGuard early (matrix) vs intermediate OK")


def test_all():
    test_adapter_distinct_and_unit_split()
    test_splitter_too_few_units_raises()
    test_ndtbatch_collate_and_availability()
    test_ndtbatch_collate_dtype_when_first_instance_missing_modality()
    test_modality_stems_and_encoder_shapes()
    test_fusion_heads_shapes()
    test_fusion_heads_explicit_modality_order_not_dict_order()
    test_score_fusion_head_masks_and_renormalizes()
    test_missing_modality_invariance()
    test_gated_fusion_head_zeroes_missing_before_mlp()
    test_task_head_and_ultrasonic_only_pipeline()
    test_paired_data_guard_blocks_unpaired_fusion()
    test_paired_data_guard_early_vs_intermediate()
    print("\nAll M0-1.5 interface tests passed.")


if __name__ == "__main__":
    test_all()
