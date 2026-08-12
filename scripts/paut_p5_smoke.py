#!/usr/bin/env python3
"""P5 冒烟测试：缺陷注入可视化与物理保真度验证。

在 clean A-scan 上注入一个合成缺陷（局部高斯峰），对比与真实 pos/neg 样本的差异。
如注入分布与真实 pos 在 0.5%-10% 异常区域的特征相似（局部峰、跨多波束、
幅度相对本底较小），则 P5 注入设计物理保真。

Usage:
  python scripts/paut_p5_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def inject_defect(x: np.ndarray, rng: np.random.Generator,
                  amp_range=(800, 3000), sigma_b=(0.8, 2.0),
                  sigma_t=(2.0, 6.0), edge=(8, 20)):
    """在 clean A-scan (49, 512) 上注入一个局部高斯峰。

    参数选择依据 (data/processed/paut 实测):
      - 真实 pos 相对 neg 的峰值差异: 200-3700 (PP4/PP6 实测)
      - 缺陷空间范围: 2-6 beams, ~10-30 time samples
      - 位置均匀随机, 避开边沿 (避免几何边界)
    """
    H, W = x.shape
    b0 = rng.integers(edge[0], H - edge[0])
    t0 = rng.integers(edge[1], W - edge[1])
    A = rng.uniform(*amp_range)
    sb = rng.uniform(*sigma_b)
    st = rng.uniform(*sigma_t)
    b_grid = np.arange(H)[:, None]
    t_grid = np.arange(W)[None, :]
    peak = A * np.exp(-((b_grid - b0) ** 2) / (2 * sb ** 2)) \
              * np.exp(-((t_grid - t0) ** 2) / (2 * st ** 2))
    return x + peak.astype(x.dtype), (b0, t0, A, sb, st)


def main():
    processed = REPO / "data/processed/paut"
    asc = np.load(processed / "ascans.npy")
    env = np.load(processed / "env.npy")
    lbl = np.load(processed / "meta_label.npy")
    cp = np.load(processed / "meta_coupon.npy")

    print("=" * 60)
    print("P5 注入冒烟测试")
    print("=" * 60)
    print(f"ascans shape: {asc.shape}, range: [{asc.min():.0f}, {asc.max():.0f}]")
    print(f"defect rate per coupon:")
    for c in np.unique(cp):
        m = (cp == c)
        print(f"  {c}: n={m.sum()}, pos={lbl[m].sum()}, rate={lbl[m].mean():.3f}")

    rng = np.random.default_rng(42)

    # Pick a clean (label=0) sample
    neg_idx = np.where((cp == "PP3") & (lbl == 0))[0]
    sample = asc[neg_idx[10]]
    print(f"\n选取 PP3 neg sample (idx={neg_idx[10]})")
    print(f"  原图 max: {sample.max():.0f}, argmax(beam,time): "
          f"{np.unravel_index(sample.argmax(), sample.shape)}")

    # Inject
    injected, params = inject_defect(sample, rng)
    b0, t0, A, sb, st = params
    print(f"  注入: beam={b0}, time={t0}, A={A:.0f}, sigma_b={sb:.2f}, sigma_t={st:.2f}")
    print(f"  注入后 max: {injected.max():.0f}, 增量: {(injected-sample).max():.0f}")

    # Compare to real positives
    print(f"\n=== 注入与真实 pos 样本的分布对比 (PP3) ===")
    pos_idx = np.where((cp == "PP3") & (lbl == 1))[0][:50]
    pos_peaks = []
    for i in pos_idx:
        a = asc[i]
        # Find localized peak: max of (a - median) within small window
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(a, size=(7, 15))
        peak_val = (a * (a == local_max)).max()
        pos_peaks.append(peak_val)
    print(f"  PP3 pos 局部峰值: q25/50/75 = "
          f"{np.percentile(pos_peaks, [25, 50, 75]).astype(int).tolist()}")

    # Simulate many injections and check distribution
    sim_peaks = []
    for _ in range(200):
        i = neg_idx[rng.integers(0, len(neg_idx))]
        a = asc[i]
        inj, _ = inject_defect(a, rng)
        # Find the new peak
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(inj, size=(7, 15))
        peak_val = (inj * (inj == local_max)).max()
        sim_peaks.append(peak_val)
    print(f"  注入 200 次 局部峰值: q25/50/75 = "
          f"{np.percentile(sim_peaks, [25, 50, 75]).astype(int).tolist()}")

    # Verify injection is visible but not dominant
    print(f"\n=== 注入相对本底显著性 ===")
    n_trials = 100
    snrs = []
    for _ in range(n_trials):
        i = neg_idx[rng.integers(0, len(neg_idx))]
        a = asc[i]
        inj, params = inject_defect(a, rng)
        b0, t0, A, sb, st = params
        # SNR: peak increment / local background std
        b0, t0 = int(b0), int(t0)
        bg_window = a[max(0, b0-15):b0+15, max(0, t0-30):t0+30]
        bg_std = bg_window.std()
        increment = A  # the Gaussian peak amplitude
        snr = increment / (bg_std + 1e-6)
        snrs.append(snr)
    snrs = np.array(snrs)
    print(f"  SNR (peak/local std): q25/50/75 = "
          f"{np.percentile(snrs, [25, 50, 75]).tolist()}")
    print(f"  SNR range: [{snrs.min():.2f}, {snrs.max():.2f}]")

    # Save a sample for visual inspection
    out_dir = REPO / "experiments/runs/p5_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sample_clean.npy", sample)
    np.save(out_dir / "sample_injected.npy", injected)
    print(f"\n样本已保存到 {out_dir}/")
    print("下一步: python scripts/paut_p5_inject_pretrain.py --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
