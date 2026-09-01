"""general_ndt 最小基础设施 smoke/unit 测试 (General NDT Foundation 主线)。

覆盖 (Phase 1 代码部分):
1. 数据集 registry 注册了 penelope_paut / eddycus;
2. PENELOPE loader: 字段完整 / specimen 覆盖 / 无跨试件泄漏;
3. EddyCus loader: 字段完整 / 8 通道 (4 freq × I/Q);
4. 数据审计统计量 (样本数 / specimen 数 / shape / 标签 / ID 重复);
5. 统一 collate: padding + valid mask 正确;
6. 物理感知掩码: random / time_segment / sensor_channel / freq_band / spatial_region;
7. 自监督目标: masked_recon / tf_consistency / cross_sensor 可计算;
8. 严格划分: leave_one_specimen_split 保证同试件不跨 train/test。

运行:  pytest tests/test_general_ndt.py  (数据缺失的用例自动 skip, 不失败)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pytest
import torch

from general_ndt.datasets.audit import (
    audit_samples,
    check_split_disjointness,
    check_specimen_leak,
)
from general_ndt.datasets.collate import collate_general_ndt
from general_ndt.datasets.registry import DATASETS, build_dataset
from general_ndt.evaluation.probe import leave_one_specimen_split
from general_ndt.ssl.masking import (
    MaskController,
    mask_freq_band,
    mask_random,
    mask_sensor_channel,
    mask_spatial_region,
    mask_time_segment,
)
from general_ndt.ssl.objectives import (
    cross_sensor_invariance_loss,
    masked_recon_loss,
    tf_consistency_loss,
)

PENELOPE_ROOT = Path("data/processed/paut")
EDDYCUS_DIR = Path("data/raw/EddyCus-HDF5/output")


def _has_penelope() -> bool:
    return (PENELOPE_ROOT / "ascans.npy").exists()


def _has_eddycus() -> bool:
    return EDDYCUS_DIR.exists() and len(list(EDDYCUS_DIR.glob("*.h5"))) > 0


@pytest.mark.skipif(not _has_penelope(), reason="PENELOPE processed 数据未就绪")
class TestPenelope:
    def test_registry(self):
        assert "penelope_paut" in DATASETS

    def test_loader_fields(self):
        samples = build_dataset("penelope_paut", {"sample_limit": 50})
        assert len(samples) > 0
        s = samples[0]
        assert s.modality == "ultrasonic"
        assert s.shape_kind == "1d"
        assert s.signal.ndim == 2 and s.signal.shape[0] == 49
        assert s.specimen_id in ("PP3", "PP4", "PP5", "PP6", "PP7")
        assert s.label in (0, 1)

    def test_sample_limit_preserves_specimens(self):
        samples = build_dataset("penelope_paut", {"sample_limit": 30})
        sps = {s.specimen_id for s in samples}
        assert len(sps) >= 2  # 小 limit 也要覆盖多个 coupon

    def test_no_specimen_leak(self):
        samples = build_dataset("penelope_paut", {"sample_limit": 200})
        folds = leave_one_specimen_split(samples)
        assert len(folds) >= 2
        for train_idx, val_idx, test_idx in folds:
            train_sp = {samples[i].specimen_id for i in train_idx}
            test_sp = {samples[i].specimen_id for i in test_idx}
            val_sp = {samples[i].specimen_id for i in val_idx}
            assert not (train_sp & test_sp)
            assert not (val_sp & test_sp)


@pytest.mark.skipif(not _has_eddycus(), reason="EddyCus h5 数据未就绪")
class TestEddycus:
    def test_registry(self):
        assert "eddycus" in DATASETS

    def test_loader_fields(self):
        samples = build_dataset("eddycus", {"sample_limit": 10, "max_points": 1024})
        assert len(samples) > 0
        s = samples[0]
        assert s.modality == "eddy_current"
        assert s.shape_kind == "1d"
        assert s.signal.shape[0] == 8  # 4 freq × I/Q
        assert s.signal.shape[1] <= 1024
        assert s.label in (0, 1)
        assert s.defect_type is not None


def test_audit_stats_synthetic():
    from general_ndt.datasets.schema import GeneralNDTSample

    rng = np.random.default_rng(0)
    samples = [
        GeneralNDTSample(
            sample_id=f"syn:{i}",
            signal=rng.normal(size=(3, 128)).astype(np.float32),
            shape_kind="1d",
            modality="ultrasonic",
            specimen_id=f"sp{i % 4}",
            label=i % 2,
            label_type="binary",
            defect_type="crack" if i % 2 else "clean",
        )
        for i in range(16)
    ]
    stats = audit_samples(samples)
    assert stats["n_samples"] == 16
    assert stats["n_specimens"] == 4
    assert stats["duplicate_sample_ids"] == 0
    assert stats["shape_distribution"].get("((3, 128), '1d')") == 16
    assert stats["label_distribution"] == {"0": 8, "1": 8}


def test_collate_padding_and_valid_mask():
    from general_ndt.datasets.schema import GeneralNDTSample

    samples = [
        GeneralNDTSample(
            sample_id=f"c{i}",
            signal=np.ones((2, 64), dtype=np.float32) * i,
            shape_kind="1d",
            modality="eddy_current",
            specimen_id="sp0",
            label=i % 2,
            label_type="binary",
        )
        for i in range(3)
    ]
    # 长度不一: 64 / 40 / 20
    samples[1].signal = np.ones((2, 40), dtype=np.float32)
    samples[2].signal = np.ones((2, 20), dtype=np.float32)
    batch = collate_general_ndt(samples)
    assert batch.padded_signal.shape == (3, 2, 64)
    assert batch.valid_mask.shape == (3, 64)
    assert batch.valid_mask[0].sum() == 64
    assert batch.valid_mask[1].sum() == 40
    assert batch.valid_mask[2].sum() == 20
    # padding 区域数值为 0
    assert batch.padded_signal[2, 0, 20] == 0.0


class TestMasking:
    def test_random_ratio(self):
        m = mask_random((4, 100), 0.3, seed=0)
        assert m.shape == (4, 100) and m.dtype == bool
        assert 0.25 <= m.mean() <= 0.35

    def test_time_segment_masks_full_columns(self):
        m = mask_time_segment((5, 100), 0.2, seed=0)
        assert m.shape == (5, 100)
        # 整列掩码: 每列要么全掩要么全不掩
        col_sums = m.sum(axis=0)
        assert set(col_sums.tolist()) <= {0, 5}
        assert 0.15 <= m.mean() <= 0.25

    def test_sensor_channel_masks_full_rows(self):
        m = mask_sensor_channel((8, 50), 0.25, seed=0)
        row_sums = m.sum(axis=1)
        assert set(row_sums.tolist()) <= {0, 50}
        assert 0.15 <= m.mean() <= 0.35

    def test_freq_band_masks_contiguous_rows(self):
        m = mask_freq_band((16, 40), 0.2, seed=0)
        rows = np.where(m.sum(axis=1) > 0)[0]
        assert len(rows) >= 1
        # 环状连续: 排序后相邻差 ≤1 (或首尾环)
        dr = np.diff(rows)
        assert all(np.minimum(dr, len(rows) - dr) <= 1)

    def test_spatial_region(self):
        m = mask_spatial_region((10, 10), 0.3, seed=0)
        assert m.shape == (10, 10)
        assert 0.2 <= m.mean() <= 0.45

    def test_controller_mixture(self):
        ctl = MaskController(
            mode={"random": 0.5, "time_segment": 0.5}, ratio=0.3
        )
        for _ in range(20):
            m = ctl((4, 100), seed=1)
            assert m.shape == (4, 100)
            assert 0.15 <= m.mean() <= 0.45

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            MaskController("nonsense")


class TestObjectives:
    def test_masked_recon_loss(self):
        pred = torch.randn(2, 8, 16)
        target = torch.randn(2, 8, 16)
        mask = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.bool)
        valid = torch.ones(2, 8, dtype=torch.bool)
        loss = masked_recon_loss(pred, target, mask, valid)
        assert loss.ndim == 0 and torch.isfinite(loss)
        # 只有 masked∩valid 计算 → 8 个元素
        assert torch.isclose(loss * 8, ((pred - target) ** 2)[mask].sum(), atol=1e-5)

    def test_tf_consistency_loss(self):
        z_raw = torch.randn(4, 32)
        z_tf = torch.randn(4, 32)
        loss = tf_consistency_loss(z_raw, z_tf, temperature=0.1)
        assert loss.ndim == 0 and loss > 0 and torch.isfinite(loss)

    def test_cross_sensor_invariance(self):
        pairs = [(torch.randn(8), torch.randn(8)) for _ in range(3)]
        loss = cross_sensor_invariance_loss(pairs)
        assert loss.ndim == 0 and loss > 0
        assert torch.isclose(cross_sensor_invariance_loss([]), torch.tensor(0.0))


class TestTorchLayers:
    """模态适配器 + 共享骨干 + 目标函数前向 (CPU 小批量, 不训练)。"""

    def test_adapter_1d_and_backbone(self):
        from general_ndt.adapters import ModalAdapter
        from general_ndt.models import PatchTransformer

        torch.manual_seed(0)
        ad = ModalAdapter(d_model=64, patch_len=16, patch2d=16)
        bb = PatchTransformer(d_model=64, n_layers=2, n_heads=2)
        x = torch.randn(2, 49, 512)
        z, grid = ad(x, "1d", ["ultrasonic", "ultrasonic"], [1.0, 1.0])
        assert grid == (49, 32)
        assert z.shape == (2, 49 * 32, 64)
        h = bb(z)
        assert h.shape == (2, 49 * 32 + 1, 64)
        assert bb.pooled(z).shape == (2, 64)

    def test_adapter_2d_and_backbone(self):
        from general_ndt.adapters import ModalAdapter
        from general_ndt.models import PatchTransformer

        torch.manual_seed(0)
        ad = ModalAdapter(d_model=64, patch_len=16, patch2d=16)
        bb = PatchTransformer(d_model=64, n_layers=2, n_heads=2)
        x = torch.randn(2, 16, 32)
        z, grid = ad(x, "2d", ["eddy_current", "eddy_current"], [1.0, 1.0])
        assert z.shape[0] == 2 and z.shape[-1] == 64
        assert bb.pooled(z).shape == (2, 64)

    def test_end_to_end_losses(self):
        from general_ndt.adapters import ModalAdapter
        from general_ndt.models import PatchTransformer
        from general_ndt.ssl.masking import MaskController

        torch.manual_seed(0)
        ad = ModalAdapter(d_model=64, patch_len=16, patch2d=16)
        bb = PatchTransformer(d_model=64, n_layers=2, n_heads=2)
        x = torch.randn(2, 49, 512)
        z, grid = ad(x, "1d", ["ultrasonic", "ultrasonic"], [1.0, 1.0])
        m = MaskController({"random": 0.5, "time_segment": 0.5}, 0.3)(grid, seed=1)
        mask = torch.from_numpy(m).reshape(1, -1).expand(2, -1).bool()
        valid = torch.ones(2, z.shape[1], dtype=torch.bool)
        loss_r = masked_recon_loss(torch.randn_like(z), z, mask, valid)
        loss_t = tf_consistency_loss(bb.pooled(z), bb.pooled(z + 0.1 * torch.randn_like(z)))
        assert loss_r.ndim == 0 and torch.isfinite(loss_r)
        assert loss_t.ndim == 0 and loss_t > 0


def test_leave_one_specimen_split_disjoint():
    from general_ndt.datasets.schema import GeneralNDTSample

    samples = [
        GeneralNDTSample(
            sample_id=f"x{i}",
            signal=np.zeros((1, 32), dtype=np.float32),
            shape_kind="1d",
            modality="ultrasonic",
            specimen_id=f"sp{i % 3}",
            label=0,
            label_type="binary",
        )
        for i in range(9)
    ]
    folds = leave_one_specimen_split(samples)
    assert len(folds) == 3
    for train_idx, val_idx, test_idx in folds:
        assert check_specimen_leak(samples, train_idx, test_idx) == []
        split_of = {}
        for i in train_idx:
            split_of[samples[i].sample_id] = "train"
        for i in val_idx:
            split_of[samples[i].sample_id] = "val"
        for i in test_idx:
            split_of[samples[i].sample_id] = "test"
        assert check_split_disjointness(samples, split_of) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
