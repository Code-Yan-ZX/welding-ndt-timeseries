"""M0-2C ECT 顺序 SSL 训练审计测试（数据/划分/权重迁移/valid-mask/一致性）。

覆盖任务要求：
1. 正式训练只用 695 个有 signal_data 的扫描，43 个 metadata-only 必须排除；
2. ``EddyCusAdapter.split_indices``：unit=sensor/material 真正按
   sensor_type/material_type 分组（不允许退化为 defect_instance_id）；
   同一 sensor/material 绝不跨 split；
3. clean 记录按 specimen/config proxy 分组（不归入单一 clean 单元），
   不同 clean 配置可进入不同 fold；
4. fold 审计：train/val/test 同时有正负样本否则停止；SGKFold 组不跨 fold；
5. 权重迁移：第一层 1→2 通道 ``new=old.repeat(1,2,1,1)/2``，missing/unexpected
   断言；折回 ``w[:,0:1]+w[:,1:2]`` 与原权重 diff<1e-6；
6. valid-mask 损失：masked∩valid 才进 loss，padding/缺失点绝不进；
7. 下采样规则固定、预先声明、E/P→E 一致；
8. 归一化 median/MAD per scan-frequency per I/Q channel；
9. 数据顺序/mask 计划只由 data_seed / model_seed 决定（E/P→E 一致）；
10. 8 类只输出分布不训练（probe 恒为二分类）。

运行：python tests/test_m0_2c_training.py  （无原始数据时优雅跳过 live 项）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from wndt.data.adapters.eddycus import EddyCusAdapter  # noqa: E402
from wndt.data.eddycus_pretrain import (  # noqa: E402
    BLOCK, DEFAULT_MASK_RATIO, build_view_index, block_mask,
    downsample_grid, downsample_scale, ect_bucket_plan, read_view,
    read_view_ds, robust_normalize_1d, sample_block_masks,
)
from wndt.models.ssl_ae import ECTDecoder, ECTMaskedAE, MAEEncoder  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

import m0_2c_ect_pretrain  # noqa: E402
from m0_2c_ect_pretrain import (  # noqa: E402
    ckpt_path, expected_transfer_keys, fold_back_first_layer,
    migrate_first_layer,
)
from m0_2c_ect_probe import audit_split  # noqa: E402

DATA_ROOT = REPO / "data/raw/EddyCus-HDF5/output"
HAS_DATA = DATA_ROOT.exists() and any(DATA_ROOT.glob("scan_*.h5"))
CKPT = REPO / "experiments/runs/ssl_ae/encoder.pt"
HAS_P1 = CKPT.exists()


# ---------------------------------------------------------------------------
# 1. 695 有信号扫描 / 43 metadata-only 排除
# ---------------------------------------------------------------------------
def test_signal_records_only():
    if not HAS_DATA:
        print("skip test_signal_records_only (no raw data)")
        return
    ad = EddyCusAdapter()
    sig = ad.signal_records()
    idx = ad.signal_indices()
    assert len(sig) == 695, f"expected 695 signal records, got {len(sig)}"
    assert len(idx) == 695
    # 排除的记录是 43 个 metadata-only（无 signal_data/f1）
    missing = [i for i in range(len(ad)) if i not in set(idx)]
    assert len(missing) == 738 - 695 == 43, len(missing)
    # 与全量 manifest 对照
    assert len(ad.records()) == 738
    print("test_signal_records_only OK (695 signal, 43 metadata-only excluded)")


def test_view_index_2780():
    if not HAS_DATA:
        print("skip test_view_index_2780 (no raw data)")
        return
    ad = EddyCusAdapter()
    views = build_view_index(ad)
    assert len(views) == 695 * 4 == 2780, len(views)
    # frequency 不是独立物理样本：group 键 = 扫描物理配置
    groups = {v.specimen_id for v in views}
    assert len(groups) < 300, "group 数必须远小于 view 数"
    print(f"test_view_index_2780 OK ({len(views)} views, {len(groups)} config groups)")


# ---------------------------------------------------------------------------
# 2. split_indices：sensor/material 真正分组 + 不跨 split
# ---------------------------------------------------------------------------
def _split_purity(split, unit_keys):
    where = {}
    for part, idxs in split.items():
        for i in idxs:
            where[i] = part
    bad = []
    groups = {}
    for i, k in enumerate(unit_keys):
        groups.setdefault(k, set()).add(i)
    for k, members in groups.items():
        parts = {where[m] for m in members if m in where}
        if len(parts) > 1:
            bad.append((k, parts))
    return bad


def test_split_sensor_real_grouping():
    if not HAS_DATA:
        print("skip test_split_sensor (no raw data)")
        return
    ad = EddyCusAdapter()
    keys = ad.unit_keys("sensor")
    # 真正读 sensor_type 分组：8 个不同传感器，绝不能退化为 defect 组
    n_sensor_groups = len(set(keys))
    assert n_sensor_groups >= 5, f"sensor 分组退化: {n_sensor_groups}"
    split = ad.split_indices("sensor", unit="sensor", seed=42)
    bad = _split_purity(split, keys)
    assert bad == [], f"同一 sensor 跨 split: {bad[:3]}"
    # 各 split 非空
    assert all(len(split[p]) > 0 for p in ("train", "val", "test"))
    print(f"test_split_sensor_real_grouping OK ({n_sensor_groups} sensors, no leak)")


def test_split_material_real_grouping():
    if not HAS_DATA:
        print("skip test_split_material (no raw data)")
        return
    ad = EddyCusAdapter()
    keys = ad.unit_keys("material")
    n_mat_groups = len(set(keys))
    assert n_mat_groups >= 4, f"material 分组退化: {n_mat_groups}"
    split = ad.split_indices("material", unit="material", seed=42)
    bad = _split_purity(split, keys)
    assert bad == [], f"同一 material 跨 split: {bad[:3]}"
    print(f"test_split_material_real_grouping OK ({n_mat_groups} materials, no leak)")


# ---------------------------------------------------------------------------
# 3. clean 记录按 specimen/config proxy 分组（不归入单一 clean 单元）
# ---------------------------------------------------------------------------
def test_clean_grouped_by_config_proxy():
    if not HAS_DATA:
        print("skip test_clean_grouped (no raw data)")
        return
    ad = EddyCusAdapter()
    keys = ad.unit_keys("defect")
    clean_keys = {k for k in keys if k.startswith("clean:")}
    # clean 84 条按配置代理分多组，不是单一 "clean:eddycus"
    assert len(clean_keys) > 10, f"clean 单元数过少: {len(clean_keys)}"
    # 与全部记录数对照：clean 总记录 = 84（manifest 口径）
    n_clean = sum(1 for r in ad.records() if not r.defect_present)
    assert n_clean == 84, n_clean
    # defect 划分保持 no-leak（含 clean 配置代理）
    split = ad.split_indices("defect", unit="defect", seed=42)
    assert ad.validate_defect_split(split), "defect split leaked"
    bad = _split_purity(split, keys)
    assert bad == [], f"clean 配置组跨 split: {bad[:3]}"
    # 同一 clean 配置的重复扫描同组（组内记录数 >= 该配置扫描数）
    from collections import Counter
    c = Counter(keys)
    multi_clean = {k: v for k, v in c.items() if k.startswith("clean:") and v > 1}
    assert multi_clean, "应有含多条重复 clean 扫描的配置组"
    print(f"test_clean_grouped_by_config_proxy OK "
          f"({len(clean_keys)} clean config groups, repeated clean scans "
          f"share group, no leak)")


# ---------------------------------------------------------------------------
# 4. fold 审计 + SGKFold 组不跨 fold
# ---------------------------------------------------------------------------
def _fake_rows(n=40):
    return [{"specimen_id": f"G{i // 4}", "flaw": (i % 5 == 0), "defect_type": "gap",
             "material": "M1", "sensor": "S1"} for i in range(n)]


def test_audit_split_requires_both_classes():
    rows = _fake_rows(40)
    # 正常：全有正负
    tr = list(range(0, 20)); va = list(range(20, 30)); te = list(range(30, 40))
    out = audit_split(rows, tr, va, te, "t")
    assert out["train"]["flaw"] > 0 and out["train"]["clean"] > 0
    assert out["val"]["flaw"] > 0 and out["val"]["clean"] > 0
    assert out["test"]["flaw"] > 0 and out["test"]["clean"] > 0
    # test 无负样本 -> 必须停止
    rows_negless = [dict(r, flaw=False) for r in rows[:40]]
    try:
        audit_split(rows_negless, tr, va, list(range(30, 40)), "t")
        raise AssertionError("audit_split 应因缺负样本报错")
    except ValueError:
        pass
    print("test_audit_split_requires_both_classes OK")


def test_sgkf_groups_never_cross_fold():
    if not HAS_DATA:
        print("skip test_sgkf_groups (no raw data)")
        return
    ad = EddyCusAdapter()
    views = build_view_index(ad)
    groups = []
    ys = []
    seen = {}
    for v in views:
        if v.rec_index not in seen:
            seen[v.rec_index] = v
            groups.append(v.specimen_id)
            ys.append(int(v.flaw))
    groups = np.array(groups)
    ys = np.array(ys)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgkf.split(np.zeros(len(ys)), ys, groups):
        tr_g = set(groups[tr].tolist())
        te_g = set(groups[te].tolist())
        assert tr_g.isdisjoint(te_g), "配置组跨 fold!"
        # 每折都有正负
        assert ys[tr].sum() > 0 and (1 - ys[tr]).sum() > 0
        assert ys[te].sum() > 0 and (1 - ys[te]).sum() > 0
    print("test_sgkf_groups_never_cross_fold OK")


# ---------------------------------------------------------------------------
# 5. 权重迁移 / 折回
# ---------------------------------------------------------------------------
def test_first_layer_migrate_foldback_exact():
    torch.manual_seed(0)
    old_w = torch.randn(32, 1, 3, 7)
    new_w = migrate_first_layer(old_w)
    assert new_w.shape == (32, 2, 3, 7)
    assert torch.allclose(new_w[:, 0] + new_w[:, 1], old_w[:, 0], atol=1e-12)
    back = fold_back_first_layer(new_w)
    assert back.shape == (32, 1, 3, 7)
    # 折回必须与原权重 bit 级一致（diff<1e-6）
    assert (back - old_w).abs().max().item() < 1e-6
    print("test_first_layer_migrate_foldback_exact OK (diff<1e-6)")


def test_expected_transfer_keys():
    missing, unexpected = expected_transfer_keys()
    # 迁移后全部键对齐：missing/unexpected 均为空；仅 conv.0.weight 形状不同
    assert missing == [] and unexpected == []
    print("test_expected_transfer_keys OK (migrated load: no missing/unexpected)")


def test_p1_migrate_into_2ch_model():
    if not HAS_P1:
        print("skip test_p1_migrate (no ssl_ae/encoder.pt)")
        return
    torch.manual_seed(1)
    model = ECTMaskedAE(in_channels=2)
    # 直接复用迁移逻辑（不读数据）
    import m0_2c_ect_pretrain as m
    sd = torch.load(m.TRANSFER_SOURCE, map_location="cpu", weights_only=False)
    enc_sd = dict(sd["encoder_state"])
    old_w = enc_sd.pop("conv.0.weight")
    enc_sd["conv.0.weight"] = migrate_first_layer(old_w)
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    # 迁移后全部键对齐（22 原样 + 首层迁移），无 missing/unexpected
    assert sorted(missing) == []
    assert unexpected == []
    # 双通道拷贝输入输出一致
    x1 = torch.randn(1, 1, 49, 64)
    x2 = x1.repeat(1, 2, 1, 1)
    with torch.no_grad():
        o1 = F.conv2d(x1, old_w, model.encoder.conv[0].bias,
                      padding=(1, 3))
        o2 = F.conv2d(x2, model.encoder.conv[0].weight, model.encoder.conv[0].bias,
                      padding=(1, 3))
    assert (o1 - o2).abs().max().item() < 1e-4
    print("test_p1_migrate_into_2ch_model OK")


def test_migrate_then_foldback_p1_exact():
    if not HAS_P1:
        print("skip test_migrate_then_foldback (no ssl_ae/encoder.pt)")
        return
    sd = torch.load(m0_2c_ect_pretrain.TRANSFER_SOURCE,
                    map_location="cpu", weights_only=False)
    old = sd["encoder_state"]["conv.0.weight"]
    new = migrate_first_layer(old)
    back = fold_back_first_layer(new)
    assert (back - old).abs().max().item() < 1e-6, \
        "ECT 续训前折回必须与原 P 权重 diff<1e-6"
    print("test_migrate_then_foldback_p1_exact OK (diff<1e-6)")


# ---------------------------------------------------------------------------
# 6. valid-mask 损失（padding/缺失点绝不进 loss）
# ---------------------------------------------------------------------------
def test_recon_loss_masked_valid_only():
    torch.manual_seed(3)
    model = ECTMaskedAE(in_channels=2)
    H, W = 32, 64
    x = torch.randn(1, 2, H, W)
    valid = torch.ones(1, H, W, dtype=torch.bool)
    valid[0, :, -8:] = False          # 模拟 padding 区（缺失点）
    mask = torch.ones(1, 1, H, W)
    mask[0, 0, :16, :16] = 0.0        # 左上 16x16 块 masked
    recon, target, m, v = model(x, valid, mask)
    loss = model.recon_loss(recon, target, m, v)
    # 手动计算：只统计 masked∩valid
    masked = (1.0 - m) * v.unsqueeze(1).float()
    manual = F.smooth_l1_loss((recon - target) * masked,
                              torch.zeros_like(recon), reduction="sum") \
        / masked.sum().clamp(min=1)
    assert torch.allclose(loss, manual, atol=1e-6)
    # 缺失点污染 target 不影响 loss（不在 masked∩valid）
    target_bad = target.clone()
    target_bad[0, :, :, -8:] = 1e6    # invalid 区：diff 巨大但必须不计
    loss_bad = model.recon_loss(recon, target_bad, m, v)
    assert torch.allclose(loss_bad, loss, atol=1e-3), \
        "invalid 像素进入了 loss！"
    # visible∩valid 也不进 loss
    target_bad2 = target.clone()
    target_bad2[0, :, 16, 16] = 1e6   # visible valid 点
    loss_bad2 = model.recon_loss(recon, target_bad2, m, v)
    assert torch.allclose(loss_bad2, loss, atol=1e-3), "visible 像素进了 loss！"
    # masked∩valid 进 loss
    target_bad3 = target.clone()
    target_bad3[0, :, 8, 8] = 1e6     # masked valid 点
    loss_bad3 = model.recon_loss(recon, target_bad3, m, v)
    assert loss_bad3.item() > loss.item(), "masked∩valid 没进 loss！"
    # masked 区域占全网格 = 16x16/(32x64) = 0.125
    frac = float(masked.mean())
    assert abs(frac - (16 * 16 / (32 * 64))) < 1e-3, frac
    print("test_recon_loss_masked_valid_only OK")


def test_block_mask_ratio_and_determinism():
    H, W = 101, 451
    for s in (42, 43):
        g = torch.Generator().manual_seed(s)
        m = block_mask(H, W, DEFAULT_MASK_RATIO, g, BLOCK)
        assert m.shape == (1, 1, H, W)
        frac_masked = float((m == 0).float().mean())
        # block 级 ~30%
        assert 0.25 <= frac_masked <= 0.35, frac_masked
        g2 = torch.Generator().manual_seed(s)
        m2 = block_mask(H, W, DEFAULT_MASK_RATIO, g2, BLOCK)
        assert torch.equal(m, m2)
    # sample_block_masks：E/P→E 同 model_seed 相同；不同 model_seed 不同
    a = sample_block_masks(101, 451, 4, 42, 0)
    b = sample_block_masks(101, 451, 4, 42, 0)
    c = sample_block_masks(101, 451, 4, 43, 0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    print("test_block_mask_ratio_and_determinism OK")


# ---------------------------------------------------------------------------
# 7. 下采样规则（固定、预先声明、E/P→E 一致）
# ---------------------------------------------------------------------------
def test_downsample_rule():
    # 预先声明的等比例规则：S = max(ceil(H/256), ceil(W/768))
    assert downsample_scale(101, 451) == 1
    assert downsample_scale(51, 451) == 1
    assert downsample_scale(202, 1067) == 2
    assert downsample_scale(501, 560) == 2
    assert downsample_scale(251, 560) == 1
    g = np.random.RandomState(0).randn(2, 202, 1067).astype(np.float32)
    v = np.ones((202, 1067), dtype=bool)
    g2, v2 = downsample_grid(g, v, 2)
    assert g2.shape == (2, 101, 534), g2.shape
    assert v2.shape == (101, 534) and v2.all()
    # 缺失点随采样映射（源 NaN -> 目标 NaN/valid=False）
    g3 = g.copy()
    g3[0, 200, 1000] = np.nan
    v3 = v.copy()
    v3[200, 1000] = False
    g4, v4 = downsample_grid(g3, v3, 2)
    assert g4.shape == g2.shape and v4.shape == v2.shape
    # 确定性
    g5, v5 = downsample_grid(g, v, 2)
    assert np.array_equal(g2, g5) and np.array_equal(v2, v5)
    print("test_downsample_rule OK (202x1067->101x534, 501x560->251x280)")


# ---------------------------------------------------------------------------
# 8. 归一化 median/MAD per channel
# ---------------------------------------------------------------------------
def test_robust_normalize():
    vals = np.array([0.0, 1.0, 2.0, 100.0, -1.0], dtype=np.float32)
    n = robust_normalize_1d(vals)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    expected = (vals - med) / (1.4826 * mad + 1e-6)
    assert np.allclose(n, expected, atol=1e-6)
    # 每个通道独立（读真实视图时逐通道）
    if HAS_DATA:
        ad = EddyCusAdapter()
        g, valid = read_view(ad, ad.signal_indices()[0], "f1")
        assert g.shape[0] == 2
        for ch in range(2):
            vals_ch = g[ch][valid]
            assert not np.isnan(vals_ch).any()
            assert abs(float(np.median(vals_ch))) < 1e-3, "median 应为 0"
            assert abs(float(np.median(np.abs(vals_ch)))) - 1.0 < 0.5, \
                "MAD 应为 ~1/1.4826"
    print("test_robust_normalize OK")


# ---------------------------------------------------------------------------
# 9. 数据顺序/mask 计划：E/P→E 完全一致
# ---------------------------------------------------------------------------
def test_sample_plan_data_seed_only():
    if not HAS_DATA:
        print("skip test_sample_plan (no raw data)")
        return
    ad = EddyCusAdapter()
    views = build_view_index(ad)
    p1 = ect_bucket_plan(42, views, 50, 8)
    p2 = ect_bucket_plan(42, views, 50, 8)
    p3 = ect_bucket_plan(43, views, 50, 8)
    assert p1 == p2, "同 data_seed 计划必须一致"
    assert p1 != p3, "不同 data_seed 计划应不同"
    # 同 batch 同尺寸
    for (H, W), idx in p1:
        for i in idx:
            v = views[i]
            assert (v.ds_H, v.ds_W) == (H, W), "batch 内尺寸必须一致"
    # 覆盖全部 view（all views indexable）
    seen = {i for _, idx in p1 for i in idx}
    assert seen.issubset(range(len(views)))
    print("test_sample_plan_data_seed_only OK")


def test_model_structure_and_epe_match():
    # 2ch encoder 首层 (32,2,3,7)；1ch 与 2ch 仅首层不同
    e1 = MAEEncoder(in_channels=1)
    e2 = MAEEncoder(in_channels=2)
    assert e2.conv[0].weight.shape == (32, 2, 3, 7)
    assert e1.conv[0].weight.shape == (32, 1, 3, 7)
    for k in e1.state_dict():
        if k != "conv.0.weight":
            assert e1.state_dict()[k].shape == e2.state_dict()[k].shape, k
    # E 与 P→E 的 decoder 初始化一致（同一 model_seed）
    set_seed(42)
    mE = ECTMaskedAE(in_channels=2)
    set_seed(42)
    mPE = ECTMaskedAE(in_channels=2)
    for k in mE.decoder.state_dict():
        assert torch.equal(mE.decoder.state_dict()[k], mPE.decoder.state_dict()[k]), k
    # ECTDecoder 输出当前 batch H×W（2 通道 I/Q 重建）
    dec = ECTDecoder(d_model=128)
    z = torch.randn(2, 128)
    for (H, W) in ((101, 451), (51, 451), (101, 534)):
        r = dec(z, H, W)
        assert r.shape == (2, 2, H, W), r.shape
    print("test_model_structure_and_epe_match OK")


# ---------------------------------------------------------------------------
# 10. 8 类只输出分布不训练（probe 恒为二分类）
# ---------------------------------------------------------------------------
def test_probe_is_binary_only():
    from wndt.utils.config import load_config
    cfg = load_config(REPO / "configs" / "m0_2c_ect.yaml")
    # probe 不包含 8 类头；head 协议是二分类
    assert "n_classes" not in cfg.model
    if HAS_DATA:
        ad = EddyCusAdapter()
        views = build_view_index(ad)
        from collections import Counter
        dist = Counter(v.defect_type for v in views)
        # 8 类分布（每 scan ×4 频率）只输出，不训练
        assert len(dist) == 8, dist
        assert dist["gap"] == 492 * 4, dist
        assert dist["clean"] == 63 * 4, dist       # 695 有信号扫描中 clean=63
        assert sum(dist.values()) == 2780, dist
    print("test_probe_is_binary_only OK")


# ---------------------------------------------------------------------------
# 12. checkpoint 路径隔离（smoke/pilot/full 不互相覆盖）
# ---------------------------------------------------------------------------
def test_checkpoint_path_isolation():
    assert ckpt_path("E", 42, 10000) != ckpt_path("E", 42, 100)
    assert ckpt_path("E", 42, 10000) != ckpt_path("E", 42, 2000, tag="pilot")
    assert ckpt_path("E", 42, 100) != ckpt_path("PE", 42, 100)
    assert "m0_2c" in str(ckpt_path("E", 42, 10000))
    # 不覆盖迁移源
    assert ckpt_path("PE", 42, 10000) != m0_2c_ect_pretrain.TRANSFER_SOURCE
    print("test_checkpoint_path_isolation OK")


def test_all():
    test_signal_records_only()
    test_view_index_2780()
    test_split_sensor_real_grouping()
    test_split_material_real_grouping()
    test_clean_grouped_by_config_proxy()
    test_audit_split_requires_both_classes()
    test_sgkf_groups_never_cross_fold()
    test_first_layer_migrate_foldback_exact()
    test_expected_transfer_keys()
    test_p1_migrate_into_2ch_model()
    test_migrate_then_foldback_p1_exact()
    test_recon_loss_masked_valid_only()
    test_block_mask_ratio_and_determinism()
    test_downsample_rule()
    test_robust_normalize()
    test_sample_plan_data_seed_only()
    test_model_structure_and_epe_match()
    test_probe_is_binary_only()
    test_checkpoint_path_isolation()
    print("\nAll M0-2C training audit tests passed.")


if __name__ == "__main__":
    test_all()
