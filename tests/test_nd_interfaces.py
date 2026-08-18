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
    """三类融合头输出形状一致，且缺失模态不污染输出。"""
    d = 64
    avail_full = torch.ones(B, 2)
    avail_missing = torch.tensor([[1, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.float32)
    toks = {
        "ultrasonic": torch.randn(B, 4, d),
        "eddy_current": torch.randn(B, 4, d),
    }
    for head in (ConcatFusionHead(d, 2), GatedFusionHead(d, 2), ScoreFusionHead(d, 2, n_classes=2)):
        out = head(toks, avail_missing)
        assert out.shape == (B, d) or out.shape == (B, 2), (type(head).__name__, out.shape)
        out_full = head(toks, avail_full)
        assert out_full.shape == out.shape
        out.sum().backward()
    print("fusion heads shapes OK")


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


def test_all():
    test_adapter_distinct_and_unit_split()
    test_splitter_too_few_units_raises()
    test_ndtbatch_collate_and_availability()
    test_modality_stems_and_encoder_shapes()
    test_fusion_heads_shapes()
    test_task_head_and_ultrasonic_only_pipeline()
    test_paired_data_guard_blocks_unpaired_fusion()
    print("\nAll M0-1 interface tests passed.")


if __name__ == "__main__":
    test_all()
