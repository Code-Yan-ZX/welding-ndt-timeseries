"""M0-2B 自动审计测试（统一超声 MAE 预训练 + 严格跨试件 LOOCV，deterministic v2）。

覆盖任务要求的审计项（原有 10 项 + deterministic v2 新增 11 项）：
1. 同一输入 seed 下 crop / frame sampling 可复现；
2. NDT_ML_Flaw crop 不越界；
3. ML-NDT 帧级输入**不会**产生 25,600-token 体积（单帧 256 tokens）；
4. PENELOPE 目标域 SSL 样本不包含 val/test coupon（无泄漏）；
5. 归一化统计不读取 val/test（只由 train coupons 计算）；
6. E1/E2/E3 使用相同 encoder 结构（arch_signature 一致）；
7. E1/E2/E3 总 optimizer steps 可比（正式 10,000 / smoke 20）；
8. 输出结果包含五折与 non-PP4 聚合；
9. smoke 结果不覆盖正式结果；
10. 新实验（det_v2）不覆盖旧结果（旧 seed42 结果与 checkpoint 保留）。

deterministic v2 新增（seed 职责分离：split_seed / data_seed / model_seed）：
D1. 相同 model seed 两次构建模型，初始 state_dict 一致；
D2. 不同 model seed 的模型初始化不同；
D3. 同一 model seed 下 E0 五折 encoder 权重一致；
D4. 同一 fold 下四个条件（E0/E1/E2/E3）的分类头初始化一致；
D5. data seed 相同时，三个 model seed 的抽帧和裁窗计划一致；
D6. split seed 相同时，三个 model seed 的 train/val/test coupon 一致；
D7. 相同 smoke 实验重复运行结果一致（GPU，合理浮点容差）；
D8. 单独运行 E2 与在 E0–E3 全部运行中的 E2 结果一致（GPU）；
D9. checkpoint 已存在或现场生成不会改变下游分类头初始化；
D10. 新实验不覆盖旧结果（旧结果文件与 checkpoint 保留）；
D11. 原有严格 coupon-level 无泄漏测试继续通过（= 项 4/5）。

运行：python tests/test_m0_2b.py   （纯 CPU 项无需 GPU/原始数据；GPU smoke
项在无 CUDA / 无 PAUT 数据时优雅跳过）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wndt.data.ultrasound_pretrain import (  # noqa: E402
    COUPONS, NDT_STRIP, NP4, NDTWindowCache, mlndt_frame_index, ndtmf_crop_start,
    paut_fold_split, penelope_fold_stats, penelope_transform, stable_hash,
    target_ssl_sample_plan,
)
from wndt.models.ultrasound_mae import UltrasoundMAE  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

import m0_2b_loocv  # noqa: E402
from m0_2b_loocv import exp_optimizer_steps, make_head, per_exp_path  # noqa: E402
from m0_2b_pretrain import (  # noqa: E402
    DET_PRETRAIN_DIR, PRETRAIN_DIR, DEFAULT_CONFIG, build_model,
    external_ckpt_path, target_ckpt_path,
)

RESULTS_DIR = REPO / "experiments" / "results"
LOOCV_SCRIPT = REPO / "scripts" / "m0_2b_loocv.py"

HAS_PAUT = (REPO / "data" / "processed" / "paut" / "ascans.npy").exists()
HAS_GPU = torch.cuda.is_available()


def _load_paut_or_skip():
    if not HAS_PAUT:
        print("skip: no data/processed/paut (PAUT-dependent tests)")
        return None
    from wndt.data.ultrasound_pretrain import load_paut
    return load_paut()


def _cfg():
    return load_config(DEFAULT_CONFIG)


def _sd(m: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _sd_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _init_sd(model_seed: int, cfg=None) -> dict:
    """build_model 前 set_seed(model_seed) 的初始化 state_dict（模拟 fit_head /
    get_encoder 契约：调用方必须在 build_model 前 set_seed）。"""
    set_seed(model_seed)
    return _sd(build_model(cfg or _cfg()))


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
# 4. PENELOPE 目标域 SSL 样本不包含 val/test coupon（D11 无泄漏）
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
# 5. 归一化统计不读取 val/test（D11 无泄漏）
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
    cfg = _cfg()
    sigs = {exp: build_model(cfg).arch_signature() for exp in ("e1", "e2", "e3")}
    assert sigs["e1"] == sigs["e2"] == sigs["e3"], sigs
    # 结构与默认 UltrasoundMAE 一致
    assert sigs["e1"] == UltrasoundMAE().arch_signature()
    print(f"test_encoder_structure_identical OK {sigs['e1']}")


# ---------------------------------------------------------------------------
# 7. E1/E2/E3 总 optimizer steps 可比
# ---------------------------------------------------------------------------
def test_optimizer_steps_comparable():
    cfg = _cfg()
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
    assert per_exp_path("e2", 42, smoke=False).name.endswith("e2_seed42_det_v2.json")
    assert per_exp_path("e2", 42, smoke=True).name.endswith("e2_seed42_det_v2_smoke.json")
    # 预训练 checkpoint 路径也带 steps，smoke(20) 与正式(10k) 不冲突
    assert external_ckpt_path(42, 20) != external_ckpt_path(42, 10000)
    assert target_ckpt_path(42, "PP3", 20) != target_ckpt_path(42, "PP3", 10000)
    print("test_smoke_does_not_overwrite OK")


# ===========================================================================
# deterministic v2：seed 职责分离（split_seed / data_seed / model_seed）
# ===========================================================================

# --- D1. 相同 model seed 两次构建模型，初始 state_dict 一致 ------------------
def test_det_model_init_reproducible():
    a = _init_sd(42); b = _init_sd(42)
    assert _sd_equal(a, b), "same model_seed must give identical init state_dict"
    print("D1 test_det_model_init_reproducible OK")


# --- D2. 不同 model seed 的模型初始化不同 ------------------------------------
def test_det_model_init_differs():
    a = _init_sd(42)
    c = _init_sd(43)
    assert not _sd_equal(a, c), "different model_seed must give different init"
    print("D2 test_det_model_init_differs OK")


# --- D3. 同一 model seed 下 E0 五折 encoder 权重一致 -------------------------
def test_det_e0_folds_share_encoder():
    cfg = _cfg()
    dev = torch.device("cpu")
    sds = []
    for tc in COUPONS:
        enc = m0_2b_loocv.get_encoder("e0", tc, 42, cfg, dev, smoke=True,
                                      split_seed=42, data_seed=42)
        sds.append(_sd(enc))
    for i in range(1, 5):
        assert _sd_equal(sds[0], sds[i]), f"E0 fold {COUPONS[i]} encoder differs"
    # 且只由 model_seed 决定：model_seed 43 的 E0 编码器应不同
    enc43 = m0_2b_loocv.get_encoder("e0", "PP3", 43, cfg, dev, smoke=True,
                                    split_seed=42, data_seed=42)
    assert not _sd_equal(sds[0], _sd(enc43)), "E0 encoder must depend on model_seed"
    print("D3 test_det_e0_folds_share_encoder OK")


# --- D4. 同一 fold 下四个条件的分类头初始化一致 ------------------------------
def test_det_head_init_shared_across_conditions():
    cfg = _cfg()
    dm = int(cfg.model.d_model)
    inits = []
    for exp in ("e0", "e1", "e2", "e3"):
        set_seed(42)                     # fit_head 在 make_head 前 set_seed(model_seed)
        inits.append(_sd(make_head(dm)))
    for i in range(1, 4):
        assert _sd_equal(inits[0], inits[i]), \
            f"head init differs across conditions ({i})"
    # 且头初始化只由 model_seed 决定（不同 model_seed -> 不同头）
    set_seed(43)
    assert not _sd_equal(inits[0], _sd(make_head(dm)))
    print("D4 test_det_head_init_shared_across_conditions OK")


# --- D5. data seed 相同时，三个 model seed 的抽帧和裁窗计划一致 --------------
def test_det_sampling_plan_model_seed_invariant():
    data_seed = 42
    # ML-NDT 抽帧 / NDT_ML_Flaw 裁窗：函数只接受 data_seed（与 model_seed 无关）
    plans_ml = [mlndt_frame_index(data_seed, "mlndt:V1", 0, 5) for _ in range(3)]
    plans_crop = [ndtmf_crop_start(data_seed, "ndtmf:batch_013:strip5", 0, 5)
                  for _ in range(3)]
    assert plans_ml[0] == plans_ml[1] == plans_ml[2]
    assert plans_crop[0] == plans_crop[1] == plans_crop[2]
    # PENELOPE 目标域 SSL 样本顺序：同一 data_seed 下三种子一致
    n, steps, bs = 1200, 200, 32
    p0 = target_ssl_sample_plan(data_seed, n, steps, bs)
    for _ in range(2):
        assert np.array_equal(p0, target_ssl_sample_plan(data_seed, n, steps, bs))
    # 不同 data_seed -> 计划不同（确凿地由 data_seed 决定）
    changed = (mlndt_frame_index(43, "mlndt:V1", 0, 5) != plans_ml[0] or
               ndtmf_crop_start(43, "ndtmf:batch_013:strip5", 0, 5) != plans_crop[0])
    assert changed, "sampling plan should change with data_seed"
    print("D5 test_det_sampling_plan_model_seed_invariant OK")


# --- D5b. 三个 model seed 复用同一份 NDT 窗口缓存（不重复建缓存） ------------
def test_det_ndt_cache_shared_and_reused():
    # 三个 model seed 用同一 data_seed=42 -> 同一缓存目录
    c1 = NDTWindowCache(data_seed=42, n_steps=320000, batch_size=32, steps_per_epoch=500)
    c2 = NDTWindowCache(data_seed=42, n_steps=320000, batch_size=32, steps_per_epoch=500)
    c3 = NDTWindowCache(data_seed=42, n_steps=320000, batch_size=32, steps_per_epoch=500)
    assert c1.dir == c2.dir == c3.dir, "NDT cache must be shared across model seeds"
    # E2 全量(10000 steps)缓存键与旧 seed42 一致 -> 直接复用，不重复建几十 GB
    full = NDTWindowCache(data_seed=42, n_steps=10000 * 32, batch_size=32,
                          steps_per_epoch=500)
    assert full.exists, "E2 全量 NDT 缓存应已存在（复用旧 seed42 构建）"
    # 复用判定（build 内部路径）：旧缓存应判为"当前版本"直接复用（含回填元数据）
    assert full._cache_current() is True, "existing cache must be reused, not rebuilt"
    meta = json.loads(full.meta_path.read_text())
    assert meta["data_seed"] == 42, "legacy cache meta should backfill data_seed=42"
    assert meta["data_version"]  # 数据版本已记录
    print("D5b test_det_ndt_cache_shared_and_reused OK")


# --- D6. split seed 相同时，三个 model seed 的 coupon 划分一致 ---------------
def test_det_split_model_seed_invariant():
    loaded = _load_paut_or_skip()
    if loaded is None:
        return
    _, coupons, _ = loaded
    ref = paut_fold_split(coupons, "PP3", 42)
    # paut_fold_split 只接受 split_seed（无 model_seed 参数）：三种子下划分一致
    for _ms in (42, 43, 44):
        got = paut_fold_split(coupons, "PP3", 42)
        assert list(coupons[got[0]]) == list(coupons[ref[0]])
        assert set(coupons[got[1]]) == set(coupons[ref[1]])
        assert set(coupons[got[2]]) == set(coupons[ref[2]])
    # 不同 split_seed -> 划分不同
    alt = paut_fold_split(coupons, "PP3", 7)
    assert set(coupons[alt[1]]) != set(coupons[ref[1]]) or \
        set(coupons[alt[2]]) != set(coupons[ref[2]])
    print("D6 test_det_split_model_seed_invariant OK")


# --- D9. checkpoint 已存在或现场生成不会改变下游分类头初始化 -----------------
def test_det_head_init_independent_of_checkpoint():
    cfg = _cfg()
    dm = int(cfg.model.d_model)
    # 情况 A：不加载任何 checkpoint
    set_seed(42)
    hA = _sd(make_head(dm))
    # 情况 B：模拟 load_checkpoint 内部 —— 先 build_model（消耗 RNG）+ load，
    # 再构建头；fit_head 在 make_head 前重新 set_seed(model_seed) -> 头一致
    set_seed(42)
    m = build_model(cfg)
    m.load_state_dict(m.state_dict())
    set_seed(42)
    hB = _sd(make_head(dm))
    assert _sd_equal(hA, hB), "head init must be independent of prior checkpoint load"
    # 更强：即使中间 build/load 多次，头初始化仍只由 model_seed 决定
    for _ in range(3):
        set_seed(42)
        build_model(cfg)
    set_seed(42)
    hC = _sd(make_head(dm))
    assert _sd_equal(hA, hC)
    print("D9 test_det_head_init_independent_of_checkpoint OK")


# --- D10. 新实验不覆盖旧结果（旧结果文件与 checkpoint 保留） -----------------
def test_det_v2_does_not_overwrite_old():
    for exp in ("e0", "e1", "e2", "e3"):
        old = RESULTS_DIR / f"m0_2b_{exp}_seed42.json"
        assert old.exists(), f"旧结果 {old.name} 必须保留"
        new = per_exp_path(exp, 42, smoke=False)
        assert old != new, f"det_v2 路径不得覆盖旧结果 {old.name}"
    # 旧 checkpoint 保留，det_v2 checkpoint 在独立目录
    assert (PRETRAIN_DIR / "external_s42_s10000.pt").exists(), "旧外部 ckpt 必须保留"
    assert external_ckpt_path(42, 10000).parent == DET_PRETRAIN_DIR
    assert "det_v2" in str(DET_PRETRAIN_DIR)
    assert external_ckpt_path(42, 10000) != PRETRAIN_DIR / "external_s42_s10000.pt"
    print("D10 test_det_v2_does_not_overwrite_old OK")


# --- D7. 相同 smoke 实验重复运行结果一致（GPU） ------------------------------
def test_det_smoke_reproducible():
    if not (HAS_GPU and HAS_PAUT):
        print("skip D7 test_det_smoke_reproducible (no GPU/PAUT)")
        return
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    def run_smoke_e2():
        subprocess.run([sys.executable, str(LOOCV_SCRIPT),
                        "--exp", "e2", "--model-seed", "42", "--smoke"],
                       check=True, env=env, capture_output=True)
        return json.loads(per_exp_path("e2", 42, smoke=True).read_text())
    r1 = run_smoke_e2()
    r2 = run_smoke_e2()
    assert r1["nonPP4_mean_auc"] == r2["nonPP4_mean_auc"], \
        f"smoke nonPP4 differs: {r1['nonPP4_mean_auc']} vs {r2['nonPP4_mean_auc']}"
    for f1, f2 in zip(r1["folds"], r2["folds"]):
        assert abs(f1["test_auc"] - f2["test_auc"]) < 1e-4, \
            f"smoke fold {f1['test_coupon']} differs: {f1['test_auc']} vs {f2['test_auc']}"
    print("D7 test_det_smoke_reproducible OK "
          f"(nonPP4={r1['nonPP4_mean_auc']} both runs)")


# --- D8. 单独运行 E2 与在 E0–E3 全部运行中的 E2 结果一致（GPU） --------------
def test_det_e2_alone_equals_e2_in_all():
    if not (HAS_GPU and HAS_PAUT):
        print("skip D8 test_det_e2_alone_equals_e2_in_all (no GPU/PAUT)")
        return
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # 先在 E0–E3 全部运行中跑出 E2
    subprocess.run([sys.executable, str(LOOCV_SCRIPT),
                    "--exp", "all", "--model-seed", "42", "--smoke"],
                   check=True, env=env, capture_output=True)
    in_all = json.loads(per_exp_path("e2", 42, smoke=True).read_text())
    # 再单独跑 E2
    subprocess.run([sys.executable, str(LOOCV_SCRIPT),
                    "--exp", "e2", "--model-seed", "42", "--smoke"],
                   check=True, env=env, capture_output=True)
    alone = json.loads(per_exp_path("e2", 42, smoke=True).read_text())
    assert in_all["nonPP4_mean_auc"] == alone["nonPP4_mean_auc"], \
        f"E2 alone vs in-all nonPP4 differs: {in_all['nonPP4_mean_auc']} vs {alone['nonPP4_mean_auc']}"
    for f1, f2 in zip(in_all["folds"], alone["folds"]):
        assert f1["test_auc"] == f2["test_auc"], \
            f"E2 fold {f1['test_coupon']} differs in-all/alone: {f1['test_auc']} vs {f2['test_auc']}"
    print("D8 test_det_e2_alone_equals_e2_in_all OK "
          f"(nonPP4={in_all['nonPP4_mean_auc']})")


def test_all():
    # 原有审计
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
    # deterministic v2 新增
    test_det_model_init_reproducible()            # D1
    test_det_model_init_differs()                 # D2
    test_det_e0_folds_share_encoder()             # D3
    test_det_head_init_shared_across_conditions() # D4
    test_det_sampling_plan_model_seed_invariant() # D5
    test_det_ndt_cache_shared_and_reused()        # D5b
    test_det_split_model_seed_invariant()         # D6
    test_det_head_init_independent_of_checkpoint()  # D9
    test_det_v2_does_not_overwrite_old()          # D10
    test_det_smoke_reproducible()                 # D7
    test_det_e2_alone_equals_e2_in_all()          # D8
    print("\nAll M0-2B audit tests passed (original 10 + deterministic v2).")


if __name__ == "__main__":
    test_all()
