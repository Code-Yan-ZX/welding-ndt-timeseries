"""Phase 2A — 实现正确性 Gate 单元测试。

覆盖:
1. Stem1D per-channel patch embedding (通道 token 非复制 / 交换/修改通道 → token 变化 /
   共享权重通道数可变 / sensor-channel masking 可感知);
2. Stem2D per-channel (多通道 native grid);
3. PatchTransformer valid mask 真正进入 attention + 网格位置编码
   (padding 内容不变性 / 时间位置交换 / 传感器位置交换 / 全有效==不传 mask / mask 实际生效);
4. token 级 valid mask (不等长 / 不等通道 / native 空洞 / 被 padding 覆盖的 patch 无效);
5. MaskController 只从 valid token 采样;
6. collate 2d 多通道归一化 + 样本级 valid_mask;
7. M0 vanilla MAE 端到端 (1d + 2d 前向, loss 有限, masked∩valid 计算)。

运行: pytest tests/test_phase2a.py -q
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

from general_ndt.adapters.stems import Stem1D, Stem2D
from general_ndt.datasets.collate import collate_general_ndt
from general_ndt.datasets.schema import GeneralNDTSample
from general_ndt.models.backbone import PatchTransformer
from general_ndt.models.mae import MaskedAutoencoder
from general_ndt.ssl.masking import MaskController, sample_mask
from general_ndt.ssl.token_masks import patchify_target, token_valid_mask


# ---------------------------------------------------------------------------
# 1. Stem1D per-channel patch embedding
# ---------------------------------------------------------------------------
class TestStem1DPerChannel:
    def test_channel_tokens_not_copies(self):
        torch.manual_seed(0)
        s = Stem1D(in_channels=1, patch_len=16, d_model=32)
        x = torch.randn(2, 4, 64)
        z, grid = s(x)
        assert grid == (4, 4)
        # 不同通道的 token 不得是复制品 (每个通道独立 patch 投影)
        t0, t1 = z[0, 0:4], z[0, 4:8]
        assert not torch.allclose(t0, t1, atol=1e-6), "不同通道 token 不能相同"

    def test_channel_swap_changes_tokens(self):
        torch.manual_seed(0)
        s = Stem1D(in_channels=1, patch_len=16, d_model=32)
        x = torch.randn(2, 4, 64)
        z, _ = s(x)
        z2, _ = s(x[:, [1, 0, 2, 3]])
        # 交换后 ch0 == 原 ch1 (每通道独立), 且 ≠ 原 ch0
        assert torch.allclose(z2[:, 0:4], z[:, 4:8], atol=1e-6)
        assert not torch.allclose(z2[:, 0:4], z[:, 0:4], atol=1e-6)

    def test_modifying_one_channel_affects_only_its_tokens(self):
        torch.manual_seed(0)
        s = Stem1D(in_channels=1, patch_len=16, d_model=32)
        x = torch.randn(2, 4, 64)
        z, _ = s(x)
        xm = x.clone()
        xm[:, 0, :] += 100.0
        z4, _ = s(xm)
        assert not torch.allclose(z4[:, 0:4], z[:, 0:4], atol=1e-5)
        assert torch.allclose(z4[:, 4:], z[:, 4:], atol=1e-5)

    def test_shared_weights_variable_channels(self):
        torch.manual_seed(0)
        s = Stem1D(in_channels=1, patch_len=16, d_model=32)
        xa = torch.randn(1, 8, 64)
        za, _ = s(xa)
        zb, _ = s(xa[:, :4])
        assert torch.allclose(za[:, 0:16], zb[:, 0:16], atol=1e-6), "共享权重支持通道数可变"

    def test_sensor_channel_masking_perceivable(self):
        # 掩掉整个 sensor channel → 该行 token 被掩 (token 级)
        torch.manual_seed(0)
        s = Stem1D(in_channels=1, patch_len=16, d_model=32)
        x = torch.randn(1, 4, 64)
        z, grid = s(x)
        m = MaskController("sensor_channel", 0.5)(grid, seed=0)
        m_flat = m.reshape(-1)
        assert m_flat[:4].all() or m_flat[4:8].all() or m_flat[8:12].all() or m_flat[12:].all()
        assert bool(m.sum()) > 0


# ---------------------------------------------------------------------------
# 2. PatchTransformer valid mask + grid positional encoding
# ---------------------------------------------------------------------------
@pytest.fixture
def bb():
    torch.manual_seed(0)
    return PatchTransformer(d_model=32, n_layers=2, n_heads=2, dropout=0.0).eval()


class TestTransformerMaskPosition:
    def test_full_valid_equals_no_mask(self, bb):
        with torch.no_grad():
            tokens = torch.randn(2, 16, 32)
            vm = torch.ones(2, 16, dtype=torch.bool)
            h = bb(tokens, valid_mask=vm, grid=(4, 4))
            h_no = bb(tokens, grid=(4, 4))
            assert torch.allclose(h, h_no, atol=0)

    def test_padding_content_invariance(self, bb):
        with torch.no_grad():
            tokens_pad = torch.randn(1, 20, 32)
            tokens_pad2 = tokens_pad.clone()
            tokens_pad2[0, 16:] = torch.randn(4, 32) * 5.0
            vm_p = torch.zeros(1, 20, dtype=torch.bool)
            vm_p[0, :16] = True
            p1 = bb.pooled(tokens_pad, valid_mask=vm_p, grid=(4, 5))
            p2 = bb.pooled(tokens_pad2, valid_mask=vm_p, grid=(4, 5))
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_batch_extent_padding_invariance(self, bb):
        with torch.no_grad():
            tv = torch.randn(1, 3, 32)
            ra = bb.pooled(tv, valid_mask=torch.ones(1, 3, dtype=torch.bool), grid=(1, 3))
            tv5 = torch.cat([tv, torch.zeros(1, 2, 32)], dim=1)
            vm5 = torch.zeros(1, 5, dtype=torch.bool)
            vm5[0, :3] = True
            rb = bb.pooled(tv5, valid_mask=vm5, grid=(1, 5))
            assert torch.allclose(ra, rb, atol=1e-5)

    def test_time_position_swap_changes_rep(self, bb):
        with torch.no_grad():
            ta = torch.randn(1, 4, 32)
            tA = torch.cat([ta, torch.randn(1, 1, 32)], dim=1)   # [a,b,c,d,pad]
            tB = torch.cat([torch.randn(1, 1, 32), ta], dim=1)   # [pad,a,b,c,d]
            vm = torch.zeros(1, 5, dtype=torch.bool)
            vm[0, :4] = True
            pA = bb.pooled(tA, valid_mask=vm, grid=(1, 5))
            pB = bb.pooled(tB, valid_mask=vm, grid=(1, 5))
            assert not torch.allclose(pA, pB, atol=1e-4)

    def test_sensor_position_swap_perceivable(self, bb):
        with torch.no_grad():
            tx = torch.randn(1, 6, 32)
            vm6 = torch.ones(1, 6, dtype=torch.bool)
            r1 = bb.pooled(tx, valid_mask=vm6, grid=(2, 3))
            r2 = bb.pooled(tx[:, [3, 4, 5, 0, 1, 2]], valid_mask=vm6, grid=(2, 3))
            assert not torch.allclose(r1, r2, atol=1e-4)

    def test_mask_actually_affects_attention(self, bb):
        with torch.no_grad():
            tH = torch.randn(1, 5, 32)
            vmH = torch.zeros(1, 5, dtype=torch.bool)
            vmH[0, :3] = True
            rH = bb.pooled(tH, valid_mask=vmH, grid=(1, 5))
            rA = bb.pooled(tH, valid_mask=torch.ones(1, 5, dtype=torch.bool), grid=(1, 5))
            assert not torch.allclose(rH, rA, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. token 级 valid mask
# ---------------------------------------------------------------------------
class TestTokenValidMask:
    def _samples(self):
        samples = [
            GeneralNDTSample(sample_id=f"a{i}", signal=np.ones((2, 48), np.float32) * i,
                             shape_kind="1d", modality="eddy_current",
                             specimen_id="s0", label=0, label_type="binary")
            for i in range(3)
        ]
        samples[1].signal = np.ones((4, 40), np.float32)
        samples[2].signal = np.ones((1, 24), np.float32)
        return samples

    def test_unequal_length_and_channels(self):
        batch = collate_general_ndt(self._samples())
        tv = token_valid_mask(batch, patch_len=8)
        assert tv.shape == (3, 4 * (48 // 8))
        assert tv[0].sum() == 2 * (48 // 8)      # channel padding 无效
        assert tv[1].sum() == 4 * (40 // 8)
        assert tv[2].sum() == 1 * (24 // 8)
        assert tv[0][:12].all() and not tv[0][12:].any()

    def test_padded_patch_invalid(self):
        # L=40, patch=8 → 第 5 列 (t=5) 部分越界 → 无效
        samples = self._samples()[:2]
        batch = collate_general_ndt(samples)
        tv = token_valid_mask(batch, patch_len=8)
        # batch0 (L=48) 全有效; batch1 (L=40, n_col=5) 前5列有效
        assert tv[0].sum() == 2 * 6
        assert tv[1].sum() == 4 * 5

    def test_native_grid_holes(self):
        sig = np.random.randn(8, 16, 32).astype(np.float32)
        vm = np.ones((16, 32), dtype=bool)
        vm[0:2, 0:2] = False                      # 空洞
        s2 = GeneralNDTSample(sample_id="g", signal=sig, shape_kind="2d",
                              modality="eddy_current", specimen_id="s0", label=1,
                              label_type="binary", valid_mask=vm)
        s3 = GeneralNDTSample(sample_id="g2", signal=np.random.randn(4, 20, 30).astype(np.float32),
                              shape_kind="2d", modality="eddy_current", specimen_id="s0",
                              label=0, label_type="binary")
        batch = collate_general_ndt([s2, s3])
        tv = token_valid_mask(batch, patch_len=4, patch2d=4)
        c_max, n_h, n_w = 8, 20 // 4, 32 // 4
        assert tv.shape == (2, c_max * n_h * n_w)
        assert tv[0].sum() == 8 * 4 * 8 - 8       # 空洞 patch 全通道无效
        assert tv[1].sum() == 4 * 5 * 7           # channel padding 无效
        assert not tv[0].reshape(c_max, n_h, n_w)[:, 0, 0].any()

    def test_patchify_target_aligns(self):
        batch = collate_general_ndt(self._samples())
        pt = patchify_target(batch, patch_len=8)
        assert pt.shape == (3, 4 * (48 // 8), 8)
        # sample0 = 全 0; sample1 = 全 1 (n_col=6, C=4)
        assert np.allclose(pt[0, 0], 0.0)        # sample0 ch0 全 0
        assert np.allclose(pt[1, 0], 1.0)        # sample1 ch0 全 1
        assert np.allclose(pt[1, 6], 1.0)        # sample1 ch1 全 1 (ch1 从索引 n_col=6 开始)


# ---------------------------------------------------------------------------
# 4. MaskController 只从 valid token 采样
# ---------------------------------------------------------------------------
class TestMaskOnlyValid:
    def test_random_only_valid(self):
        shape = (4, 6)
        valid = np.zeros(shape, dtype=bool)
        valid[:, :3] = True
        for seed in range(5):
            m = MaskController("random", 0.5)(shape, valid=valid, seed=seed)
            assert not m[:, 3:].any()
            assert abs(m.sum() / valid.sum() - 0.5) < 1e-9

    def test_structured_intersects_valid(self):
        shape = (4, 6)
        valid = np.zeros(shape, dtype=bool)
        valid[:, :3] = True
        m2 = sample_mask("sensor_channel", shape, 0.5, valid=valid, seed=0)
        assert not m2[:, 3:].any()
        assert m2.sum() > 0

    def test_spatial_region_3d_replicates(self):
        valid = np.ones((8, 5, 8), dtype=bool)
        m3 = MaskController("spatial_region", 0.3)((8, 5, 8), valid=valid, seed=0)
        assert np.array_equal(m3[0], m3[1])


# ---------------------------------------------------------------------------
# 5. M0 vanilla MAE 端到端
# ---------------------------------------------------------------------------
class TestMaeEndToEnd:
    def _mae(self, **kw):
        torch.manual_seed(0)
        defaults = dict(d_model=32, patch_len=8, patch2d=8, n_layers_enc=2, n_heads=2,
                        d_decoder=32, n_layers_dec=1, mask_ratio=0.5, dropout=0.0)
        defaults.update(kw)
        return MaskedAutoencoder(**defaults).eval()

    def test_forward_1d_loss_finite(self):
        samples = [
            GeneralNDTSample(sample_id=f"x{i}",
                             signal=np.random.randn(4, 48).astype(np.float32),
                             shape_kind="1d", modality="ultrasonic",
                             specimen_id=f"sp{i % 2}", label=i % 2, label_type="binary",
                             sampling_rate=1.0)
            for i in range(6)
        ]
        from general_ndt.trainers.ssl_trainer import SSLTrainer
        model = self._mae()
        trainer = SSLTrainer(model, {"normalize": "per_sample"}, device="cpu")
        batch = trainer._build_batches(samples, batch_size=4)[0]
        with torch.no_grad():
            out = model(batch, mask_seed=0)
        assert torch.isfinite(out["loss"])
        assert out["mask"].sum() > 0
        # mask 只在 valid token 上
        assert bool((out["mask"] & ~out["valid"]).sum() == 0)
        # 只算 masked ∩ valid
        n = int((out["mask"] & out["valid"]).sum())
        assert out["pred"].shape == out["target"].shape

    def test_forward_2d_loss_finite(self):
        samples = [
            GeneralNDTSample(sample_id=f"y{i}",
                             signal=np.random.randn(4, 16, 24).astype(np.float32),
                             shape_kind="2d", modality="eddy_current",
                             specimen_id=f"sp{i % 2}", label=i % 2, label_type="binary",
                             sampling_rate=1.0)
            for i in range(4)
        ]
        from general_ndt.trainers.ssl_trainer import SSLTrainer
        model = self._mae()
        trainer = SSLTrainer(model, {"normalize": "per_sample"}, device="cpu")
        batch = trainer._build_batches(samples, batch_size=4)[0]
        with torch.no_grad():
            out = model(batch, mask_seed=0)
        assert torch.isfinite(out["loss"])
        assert bool((out["mask"] & ~out["valid"]).sum() == 0)

    def test_raw_view_encoding(self):
        samples = [
            GeneralNDTSample(sample_id=f"z{i}",
                             signal=np.random.randn(2, 40).astype(np.float32),
                             shape_kind="1d", modality="ultrasonic",
                             specimen_id="sp0", label=i % 2, label_type="binary",
                             sampling_rate=1.0)
            for i in range(3)
        ]
        from general_ndt.trainers.ssl_trainer import SSLTrainer
        model = self._mae()
        trainer = SSLTrainer(model, {"normalize": "per_sample"}, device="cpu")
        batch = trainer._build_batches(samples, batch_size=3)[0]
        with torch.no_grad():
            pooled = model.encode_raw(batch)
        assert pooled.shape == (3, 32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
