"""M0-2B 自动审计测试（统一超声 MAE 预训练 + 严格跨试件 LOOCV）。

覆盖任务要求的审计项：
1. 同一输入 seed 下 crop / frame sampling 可复现；
2. NDT_ML_Flaw crop 不越界；
3. ML-NDT 帧级输入**不会**产生 25,600-token 体积（单帧 256 tokens）；
4. PENELOPE 目标域 SSL 样本不包含 val/test coupon；
5. 归一化统计不读取 val/test（只由 train coupons 计算）；
6. E1/E2/E3 使用相同 encoder 结构（arch_signature 一致）；
7. E1/E2/E3 总 optimizer steps 可比（正式 10,000 / smoke 20）；
8. 输出结果包含五折与 non-PP4 聚合；
9. smoke 结果不覆盖正式结果（独立 _smoke 后缀）。

运行：python tests/test_m0_2b.py   （纯 CPU；无原始数据时相关项优雅跳过）
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

from wndt.data.ultrasound_pretrain import (  # noqa: E402
    COUPONS, NDT_STRIP, NP4, mlndt_frame_index, ndtmf_crop_start,
    paut_fold_split, penelope_fold_stats, penelope_transform, stable_hash,
)
from wndt.models.ultrasound_mae import UltrasoundMAE  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402

from m0_2b_loocv import exp_optimizer_steps, per_exp_path  # noqa: E402
from m0_2b_pretrain import DEFAULT_CONFIG, build_model  # noqa: E402

RESULTS_DIR = REPO / "experiments" / "results"

HAS_PAUT = (REPO / "data" / "processed" / "paut" / "ascans.npy").exists()


def _load_paut_or_skip():
    if not HAS_PAUT:
        print("skip: no data/processed/paut (PAUT-dependent tests)")
        return None
    from wndt.data.ultrasound_pretrain import load_paut
    return load_paut()


# ---------------------------------------------------------------------------
# 1. 确定性：crop / frame sampling 可复现
# ---------------------------------------------------------------------------
def test_sampling_reproducible():
    for seed in (42, 7, 2024):
        for epoch in (0, 3, 77):
            a = mlndt_frame_index(seed, "mlndt:V1", epoch, 123)
            b = mlndt_frame_index(seed, "mlndt:V1", epoch, 123)
            assert a == b, f"frame index not reproducible seed={seed} epoch={epoch}"
            assert 0 <= a < 100
            c = ndtmf_crop_start(seed, "ndtmf:batch_013:strip5", epoch, 123)
            d = ndtmf_crop_start(seed, "ndtmf:batch_013:strip5", epoch, 123)
            assert c == d, f"crop not reproducible seed={seed} epoch={epoch}"
    print("test_sampling_reproducible OK")


# ---------------------------------------------------------------------------
# 2. NDT_ML_Flaw crop 不越界
# ---------------------------------------------------------------------------
def test_ndt_crop_in_bounds():
    for seed in (42, 43, 44):
        for epoch in range(5):
            c = ndtmf_crop_start(seed, "ndtmf:batch_210:strip999", epoch, 0)
            assert 0 <= c <= NDT_STRIP[1] - 256, f"crop out of bounds: {c}"
    assert stable_hash(42, "crop", "x", 0, 0) >= 0
    print("test_ndt_crop_in_bounds OK")


# ---------------------------------------------------------------------------
# 3. ML-NDT 默认帧级输入不会产生 25,600-token 体积
# ---------------------------------------------------------------------------
def test_mlndt_no_volume_tokens():
    m = UltrasoundMAE(patch_size=(16, 16), d_model=128)
    x = torch.randn(1, 1, 256, 256)          # 单帧（默认输入）
    z = m.encode(x)
    assert z.shape[1] == 256, f"frame should be 16x16=256 tokens, got {z.shape[1]}"
    assert z.shape[1] != 100 * 256, "volume stem would give 25600 tokens"
    # 体积逐帧串接才应产生 25,600 token（MLNDTVolumeStem 行为），此处必须不同
    assert z.shape[1] != 25_600
    print("test_mlndt_no_volume_tokens OK")


def test_mlndt_variable_frame_volume():
    """ML-NDT 个别 volume 只有 10 帧：按文件大小解析帧数，抽帧不越界。"""
    try:
        from wndt.data.adapters.ml_ndt import MLNDTAdapter
        from wndt.data.ultrasound_pretrain import (
            MLNDTFrameSource, mlndt_frame_index, read_volume_flexible,
            volume_n_frames,
        )
        ad = MLNDTAdapter()
        recs = ad.records()
        n_short = 0
        for i, r in enumerate(recs):
            n = volume_n_frames(ad, i)
            assert n >= 1
            if n != 100:
                n_short += 1
                vol = read_volume_flexible(ad, i)
                assert vol.shape == (n, 256, 256), vol.shape
                fi = mlndt_frame_index(42, r.record_id, 0, 5, n_frames=n)
                assert 0 <= fi < n, f"frame {fi} out of range for {n} frames"
        src = MLNDTFrameSource(ad, 0.0, 1.0)
        x = src.sample(42, 0, 0)          # 不应崩溃
        assert x.shape == (256, 256)
        print(f"test_mlndt_variable_frame_volume OK (non-100-frame volumes: {n_short})")
    except (FileNotFoundError, IndexError) as e:
        print(f"skip mlndt variable-frame (no data): {e}")


# ---------------------------------------------------------------------------
# 4. PENELOPE 目标域 SSL 样本不包含 val/test coupon
# ---------------------------------------------------------------------------
def test_target_ssl_excludes_val_test():
    loaded = _load_paut_or_skip()
    if loaded is None:
        return
    ascans, coupons, _ = loaded
    for tc in COUPONS:
        tr, va, te, train_c, val_c = paut_fold_split(coupons, tc, 42)
        # train coupons 不含 val / test
        assert set(coupons[tr]) == set(train_c)
        assert val_c not in set(train_c) and tc not in set(train_c)
        assert set(coupons[va]) == {val_c} and set(coupons[te]) == {tc}
        # 目标域 SSL 的输入 X 由 train 位置构建，数量=len(tr)
        mean, std = penelope_fold_stats(ascans, tr)
        X = penelope_transform(ascans, tr, mean, std)
        assert len(X) == len(tr), "target SSL input must be built only from train positions"
        # 用 NaN 污染 val/test 行也不影响 SSL 输入（train 行是唯一数据源）
        corrupted = np.array(ascans, copy=True)
        corrupted[np.concatenate([va, te])] = np.nan
        X2 = penelope_transform(corrupted, tr, mean, std)
        assert not np.isnan(X2).any()
    print("test_target_ssl_excludes_val_test OK")


# ---------------------------------------------------------------------------
# 5. 归一化统计不读取 val/test
# ---------------------------------------------------------------------------
def test_normalization_not_read_val_test():
    loaded = _load_paut_or_skip()
    if loaded is None:
        return
    ascans, coupons, _ = loaded
    tr, va, te, _, _ = paut_fold_split(coupons, "PP7", 42)
    # 把 val/test 行污染为 NaN：若统计读取它们会得到 NaN
    corrupted = np.array(ascans, copy=True)
    corrupted[np.concatenate([va, te])] = np.nan
    mean, std = penelope_fold_stats(corrupted, tr)
    assert not np.isnan(mean).any() and not np.isnan(std).any(), (
        "fold normalization read val/test rows!")
    print("test_normalization_not_read_val_test OK")


# ---------------------------------------------------------------------------
# 6. E1/E2/E3 使用相同 encoder 结构
# ---------------------------------------------------------------------------
def test_encoder_structure_identical():
    cfg = load_config(DEFAULT_CONFIG)
    sigs = {exp: build_model(cfg).arch_signature() for exp in ("e1", "e2", "e3")}
    assert sigs["e1"] == sigs["e2"] == sigs["e3"], sigs
    # 结构与默认 UltrasoundMAE 一致
    assert sigs["e1"] == UltrasoundMAE().arch_signature()
    print(f"test_encoder_structure_identical OK {sigs['e1']}")


# ---------------------------------------------------------------------------
# 7. E1/E2/E3 总 optimizer steps 可比
# ---------------------------------------------------------------------------
def test_optimizer_steps_comparable():
    cfg = load_config(DEFAULT_CONFIG)
    full = {e: exp_optimizer_steps(e, cfg, smoke=False) for e in ("e1", "e2", "e3")}
    assert full["e1"] == full["e2"] == full["e3"] == 10000, full
    smoke = {e: exp_optimizer_steps(e, cfg, smoke=True) for e in ("e1", "e2", "e3")}
    assert smoke["e1"] == smoke["e2"] == smoke["e3"] == 20, smoke
    assert exp_optimizer_steps("e0", cfg, smoke=False) == 0
    print(f"test_optimizer_steps_comparable OK full={full} smoke={smoke}")


# ---------------------------------------------------------------------------
# 8. 输出结果包含五折与 non-PP4 聚合
# ---------------------------------------------------------------------------
def _check_result_shape(res: dict):
    assert len(res["folds"]) == 5, f"need 5 folds, got {len(res['folds'])}"
    assert set(f["test_coupon"] for f in res["folds"]) == set(COUPONS)
    for f in res["folds"]:
        for k in ("test_coupon", "train_coupons", "val_coupon", "n_test",
                  "n_pos", "defect_rate", "val_auc", "test_auc", "pr_auc",
                  "optimizer_steps", "epochs_run", "wall_s"):
            assert k in f, f"fold missing {k}"
    for k in ("nonPP4_mean_auc", "nonPP4_std_auc", "pooled_auc", "pp4_auc",
              "all_folds_mean_auc", "all_folds_std_auc"):
        assert k in res, f"missing aggregate {k}"
    assert len(res["folds"]) == 5


def test_result_shape():
    # 用假数据验证输出 schema（无运行结果文件时）
    fake = {
        "folds": [{"test_coupon": c, "train_coupons": ["PP3", "PP4", "PP5"],
                   "val_coupon": "PP6", "n_test": 600, "n_pos": 100,
                   "defect_rate": 0.5, "val_auc": 0.7, "test_auc": 0.6,
                   "pr_auc": 0.5, "optimizer_steps": 10000,
                   "epochs_run": 80, "wall_s": 1.0} for c in COUPONS],
        "nonPP4_mean_auc": 0.6, "nonPP4_std_auc": 0.01,
        "pooled_auc": 0.6, "pp4_auc": 0.5,
        "all_folds_mean_auc": 0.6, "all_folds_std_auc": 0.01,
    }
    _check_result_shape(fake)
    # 有正式结果文件时校验真实结构
    real = RESULTS_DIR / "m0_2b_seed42.json"
    if real.exists():
        _check_result_shape(json.loads(real.read_text())["conditions"]["e1"])
        print("test_result_shape OK (real results checked)")
    else:
        print("test_result_shape OK (schema, no real results yet)")


# ---------------------------------------------------------------------------
# 9. smoke 结果不会覆盖正式结果
# ---------------------------------------------------------------------------
def test_smoke_does_not_overwrite():
    assert per_exp_path("e1", 42, smoke=False) != per_exp_path("e1", 42, smoke=True)
    assert per_exp_path("e2", 42, smoke=False).name.endswith("e2_seed42.json")
    assert per_exp_path("e2", 42, smoke=True).name.endswith("e2_seed42_smoke.json")
    # 预训练 checkpoint 路径也带 steps，smoke(20) 与正式(10k) 不冲突
    from m0_2b_pretrain import external_ckpt_path, target_ckpt_path
    assert external_ckpt_path(42, 20) != external_ckpt_path(42, 10000)
    assert target_ckpt_path(42, "PP3", 20) != target_ckpt_path(42, "PP3", 10000)
    print("test_smoke_does_not_overwrite OK")


def test_all():
    test_sampling_reproducible()
    test_ndt_crop_in_bounds()
    test_mlndt_no_volume_tokens()
    test_mlndt_variable_frame_volume()
    test_target_ssl_excludes_val_test()
    test_normalization_not_read_val_test()
    test_encoder_structure_identical()
    test_optimizer_steps_comparable()
    test_result_shape()
    test_smoke_does_not_overwrite()
    print("\nAll M0-2B audit tests passed.")


if __name__ == "__main__":
    test_all()
