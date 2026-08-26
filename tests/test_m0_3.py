"""M0-3 测试：adapter / group 独立性 / 变长输入 / valid-mask loss / 模型加载。

覆盖（M0-3 §四.9 与 Stage A）：
- 合成 .mat FMC（Tx×Rx×T）adapter 解析：view 数 = Tx，全部继承同一 group_id
  （禁止 Tx×Rx 当独立试件）；
- 变长输入 bucket + valid mask；
- ``ExternalUTMaskedAE`` forward/backward、recon_loss 只算 masked∩valid
  （padding/缺失点不进 loss）、无 NaN/Inf；
- 阶段 1 -> 阶段 2 encoder 加载：只迁移 encoder，decoder 新建不迁移；
- mask / 采样计划确定性（P-long 与 W→P 完全一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from wndt.data.adapters.external_weld_ut import (  # noqa: E402
    ExternalWeldUTAdapter, build_view_index, detect_fmc_arrays, group_id_for,
)
from wndt.data.external_weld_ut_pretrain import (  # noqa: E402
    build_ext_views, ext_bucket_plan, paut_ssl_sample_plan,
)
from wndt.models.ssl_ae import ExternalUTMaskedAE, FlexDecoder, MAEEncoder  # noqa: E402


@pytest.fixture(scope="module")
def fmc_tmp(tmp_path_factory):
    """合成 FMC .mat：2 试件文件，每个 (Tx=8, Rx=16, T=512) + time 向量。

    文件名必须与 ``SOURCES`` 中真实数据源的文件名一致（adapter 按名查找）。
    """
    from scipy.io import savemat
    from wndt.data.adapters.external_weld_ut import SOURCES
    root = tmp_path_factory.mktemp("m03_data")
    rng = np.random.default_rng(0)
    for src, n_tx in (("A", 8), ("B", 6)):
        d = root / src
        d.mkdir()
        fmc = rng.normal(size=(n_tx, 16, 512))
        # 时间轴（axis -1）做移动平均：模拟 A-scan 时间连续性，保证
        # canonical_fmc 的时间轴平滑度启发式（真实数据即如此）能识别时间轴。
        kern = np.ones(9) / 9.0
        fmc = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"),
                                  -1, fmc).astype(np.float64)
        tvec = np.linspace(0, 5e-5, 512)
        savemat(d / SOURCES[src]["mat"],
                {"FMC": fmc, "time": tvec, "probe": {"freq": 5e6}})
    return root


def _mk_views(root):
    from wndt.data.adapters.external_weld_ut import SOURCES
    return build_view_index(root, sources=("A", "B"))


def test_adapter_views_and_group(fmc_tmp):
    """view 数 = Σ Tx；同一文件所有 view 共享 group_id（禁止当独立试件）。"""
    views = _mk_views(fmc_tmp)
    assert len(views) == 8 + 6
    ga = {v.group_id for v in views if v.source == "A"}
    gb = {v.group_id for v in views if v.source == "B"}
    assert len(ga) == 1 and len(gb) == 1, "同一 .mat 必须共享一个 group_id"
    assert ga != gb
    ad = ExternalWeldUTAdapter(data_root=fmc_tmp)
    assert len(ad) == 14
    assert len(ad.unit_keys("specimen")) == 14


def test_detect_fmc_arrays(fmc_tmp):
    from wndt.data.adapters.external_weld_ut import SOURCES, load_mat
    mat = load_mat(fmc_tmp / "A" / SOURCES["A"]["mat"])
    fmc, tv = detect_fmc_arrays(mat)
    assert fmc.shape == (8, 16, 512)
    assert tv is not None and tv.shape == (512,)


def test_bucket_plan_deterministic(fmc_tmp):
    """bucket 计划只由 data_seed 决定（P-long/W→P 一致）。"""
    views = build_ext_views(fmc_tmp, sources=("A", "B"))
    p1 = ext_bucket_plan(42, views, 50, 8)
    p2 = ext_bucket_plan(42, views, 50, 8)
    p3 = ext_bucket_plan(7, views, 50, 8)
    assert p1 == p2
    assert p1 != p3
    # 每 batch 同尺寸（bucket 语义）
    for (h, w), idx in p1:
        assert all(views[i].ds_rx == h and views[i].ds_t == w for i in idx)


def test_paut_sample_plan_deterministic():
    a = paut_ssl_sample_plan(42, 100, 30, 8)
    b = paut_ssl_sample_plan(42, 100, 30, 8)
    c = paut_ssl_sample_plan(43, 100, 30, 8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_masked_ae_forward_backward_variable():
    """变长输入 forward/backward + 无 NaN/Inf + valid-mask loss。"""
    torch.manual_seed(0)
    model = ExternalUTMaskedAE(d_model=128, mask_ratio=0.3, block=16)
    x = torch.randn(4, 1, 32, 128)
    valid = torch.ones(4, 32, 128, dtype=torch.bool)
    valid[:, 5:9, :] = False                      # 模拟缺失块
    mask = torch.ones(4, 1, 32, 128)
    mask[:, :, :8, :64] = 0.0                     # 掩码块
    recon, target, mask_t, valid_t = model(x, valid, mask)
    loss = model.recon_loss(recon, target, mask_t, valid_t)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(recon).all()


def test_recon_loss_only_masked_valid():
    """loss 只统计 masked∩valid：padding 位置（valid=False）不贡献。"""
    torch.manual_seed(1)
    model = ExternalUTMaskedAE(d_model=64, mask_ratio=0.3, block=8)
    x = torch.randn(2, 1, 16, 64)
    # 全部 masked，但 valid 一个全 True 一个全 False
    mask = torch.zeros(2, 1, 16, 64)
    valid1 = torch.ones(2, 16, 64, dtype=torch.bool)
    valid2 = valid1.clone(); valid2[1] = False
    model.eval()                              # 关 dropout/BN 随机性，recon 确定
    recon = model(x, valid1, mask)[0]
    l1 = model.recon_loss(recon, x, mask, valid1)
    l2 = model.recon_loss(recon, x, mask, valid2)
    # 仅含样本 0 的单样本 loss：valid2 下样本 1 不贡献 -> l2 == l_single
    l_single = model.recon_loss(recon[0:1], x[0:1], mask[0:1], valid1[0:1])
    assert l1.item() > 0
    assert l2.item() == pytest.approx(l_single.item(), rel=1e-5)
    # 全 valid 时两样本都贡献 -> l1 != l_single
    assert l1.item() != pytest.approx(l_single.item(), abs=1e-3)


def test_phase2_loads_only_encoder(tmp_path):
    """阶段 2 新建 decoder；只加载 encoder（decoder 不迁移）。"""
    torch.manual_seed(2)
    m1 = ExternalUTMaskedAE(d_model=64, mask_ratio=0.3, block=8)
    m1.decoder.refine[2].weight.data.fill_(0.123)   # 阶段1 decoder 特殊权重
    ck = {"state_dict": {"encoder": m1.encoder.state_dict(),
                         "decoder": m1.decoder.state_dict()}}
    p = tmp_path / "phase1.pt"
    torch.save(ck, p)
    torch.manual_seed(3)
    m2 = ExternalUTMaskedAE(d_model=64, mask_ratio=0.3, block=8)
    sd2_dec_before = {k: v.clone() for k, v in m2.decoder.state_dict().items()}
    loaded = m2.encoder.load_state_dict(ck["state_dict"]["encoder"])
    assert loaded.missing_keys == [] and loaded.unexpected_keys == []
    # decoder 保持新初始化（未被阶段1污染）
    assert not torch.allclose(m2.decoder.refine[2].weight, torch.full_like(
        m2.decoder.refine[2].weight, 0.123))


def test_encoder_identical_after_load(tmp_path):
    """encoder 加载后前向输出与阶段 1 完全一致。"""
    torch.manual_seed(4)
    m1 = ExternalUTMaskedAE(d_model=64, mask_ratio=0.3, block=8)
    ck = {"state_dict": {"encoder": m1.encoder.state_dict(),
                         "decoder": m1.decoder.state_dict()}}
    p = tmp_path / "phase1.pt"
    torch.save(ck, p)
    torch.manual_seed(5)
    m2 = ExternalUTMaskedAE(d_model=64, mask_ratio=0.3, block=8)
    m2.encoder.load_state_dict(ck["state_dict"]["encoder"])
    x = torch.randn(2, 1, 16, 64)
    m1.eval(); m2.eval()                    # eval：关 dropout/BN 训练随机性
    with torch.no_grad():
        z1 = m1.encoder(x)
        z2 = m2.encoder(x)
    assert torch.allclose(z1, z2, atol=1e-6)


def test_mask_plan_deterministic():
    from wndt.data.eddycus_pretrain import sample_block_masks
    a = sample_block_masks(32, 128, 4, 42, 5, 0.3, 16)
    b = sample_block_masks(32, 128, 4, 42, 5, 0.3, 16)
    assert torch.equal(a, b)
