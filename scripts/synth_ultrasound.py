#!/usr/bin/env python3
"""物理参数化超声 B-scan 合成器（焊接 PAUT 样式）—— 大模型预训练数据扩充 (Synth-UT)。

动机 (2026-08-18, 新方向探索): PAUT 真实数据只有 5 试件 / 3000 位置(多视角 11980),
缺陷率 0.5%-76% 与试件身份强耦合, P0-P6 全杠杆证伪, 天花板在表征级 0.58。
P4/P6 报告点名翻盘路径 = "物理保真合成数据教缺陷回波物理"。P5 已证伪"2D 高斯峰注入"
太假不迁移 (inj_acc 0.998 但下游 0.545<0.579)。本合成器模拟真实超声传播要素:

  1. 结构噪声散斑: 带通噪声(窄带中心频率 f0), 幅度随深度指数衰减(晶粒散射)
  2. 缺陷回波: 高斯包络正弦峰(反射回波), 幅度随深度衰减(声程衰减)
  3. 声束横向展宽: 缺陷回波幅度随 |beam-center| 高斯衰减 (声束宽度决定)
  4. 走时几何: 相邻波束回波深度有线性偏移 (线扫/扇扫投影)
  5. 底面回波: 板厚深度处的强回波 (无损试件唯一稳定特征)
  6. 试件级参数: 增益 / 衰减系数 / 散斑能量 / 缺陷幅度 / 缺陷率 —— 每试件不同
      → 复现真实数据 "缺陷率-试件耦合" 挑战 (PP4 0.5% / PP6 76% / PP7 14%)

输出与 data/processed/paut/ascans.npy 同格式: (N, 49, 512) float32。
标签 = 位置级 0/1 (该位置是否落在缺陷区段), 与 meta_label.npy 同格式。

Usage:
  python scripts/synth_ultrasound.py --n-coupons 12 --n-pos-per-coupon 1000 \
      --out data/processed/synth_ut --seed 42
  python scripts/synth_ultrasound.py --viz  # 出图检查物理合理性

生成后预训练 + 规范头 LOOCV 评估见 paut_p7_synth_ssl.sh / paut_p4_ssl_variants.py。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import numpy as np

RNG_GLOBAL = None  # 由 seed 初始化的全局 RNG


def gauss_env(w, f0, sigma):
    """高斯包络正弦回波 (长度 w): sin(2π f0 t) * exp(-t²/(2σ²))。"""
    t = np.arange(w) - w / 2
    return np.sin(2 * np.pi * f0 * t / w) * np.exp(-(t ** 2) / (2 * sigma ** 2))


def make_coupon(rng: np.random.Generator, n_pos: int, coupon_cfg: dict) -> np.ndarray:
    """生成一个合成试件 (焊缝) 的 B-scan 序列。

    coupon_cfg 键:
      gain      : 全局增益 (试件灵敏度)
      atten     : 深度衰减系数 (1/样本), 幅度 *= exp(-atten*depth)
      speckle   : 散斑能量 (结构噪声幅度)
      f0        : 探头中心频率 (周期/512 深度)
      bw        : 回波包络带宽
      beam_w    : 声束宽度 (缺陷跨波束横向展宽 σ, 单位: 波束)
      def_amp   : 缺陷回波幅度 (相对散斑)
      def_depth : 缺陷深度 (0-1, 相对板厚; 底面回波在 1.0)
      def_len   : 缺陷纵向长度 (mm = 位置步进数)
      n_def     : 缺陷区段数
      bottom    : 底面回波幅度
    """
    n_beams, L = 49, 512
    B = np.zeros((n_pos, n_beams, L), dtype=np.float32)
    label = np.zeros(n_pos, dtype=np.int64)

    g = coupon_cfg["gain"]
    atten = coupon_cfg.get("atten", 0.003)
    speckle = coupon_cfg.get("speckle", 1.0)
    f0 = coupon_cfg.get("f0", 8.0)
    bw = coupon_cfg.get("bw", 20.0)
    beam_w = coupon_cfg.get("beam_w", 3.0)
    def_amp = coupon_cfg.get("def_amp", 3.0)
    def_depth = coupon_cfg.get("def_depth", 0.35)
    def_len = coupon_cfg.get("def_len", 40)
    n_def = coupon_cfg.get("n_def", 3)
    bottom = coupon_cfg.get("bottom", 2.0)

    depth = np.arange(L, dtype=np.float32) / L
    depth_atten = np.exp(-atten * np.arange(L, dtype=np.float32))

    # 结构噪声散斑: 窄带带通噪声 (零均值), 幅度随深度衰减
    # 用低维平滑随机场插值出窄带结构 → 比白噪声更像晶粒散射
    speck_field = _narrowband_noise(rng, n_pos, n_beams, L, f0, speckle)
    B += speck_field * depth_atten[None, None, :]

    # 底面回波: 板厚深度处的强峰 (所有位置一致)
    if bottom > 0:
        bottom_w = 48
        k = int(L * 0.95) - bottom_w // 2
        B[:, :, k:k + bottom_w] += (bottom * depth_atten[k] *
                                    gauss_env(bottom_w, f0 * 1.0, bw)[None, None, :])

    # 缺陷区段: 每个区段 = 连续 n_def_len 个位置, 缺陷回波跨 beam_w 波束
    avail = np.linspace(int(L * 0.12), int(L * 0.8), n_beams).astype(int)
    rng.shuffle(avail)
    for seg in range(n_def):
        seg_len = max(8, int(def_len * (0.7 + 0.6 * rng.random())))
        d0 = max(0, rng.integers(0, n_pos - seg_len))
        bc = rng.integers(4, n_beams - 4)
        depth0 = def_depth + 0.15 * rng.uniform(-0.5, 0.5)
        # 缺陷在深度轴的形态: 气孔=尖锐峰 / 裂纹=跨深度展宽
        kind = rng.choice(["blob", "crack"])
        width = rng.uniform(0.5, 1.5) if kind == "blob" else rng.uniform(2.5, 5.0)
        amp = def_amp * (0.7 + 0.6 * rng.random())
        for p in range(d0, min(d0 + seg_len, n_pos)):
            label[p] = 1
            rel = (p - d0) / max(1, seg_len - 1)  # 0..1 沿缺陷纵向
            for b in range(n_beams):
                # 声束横向权重: 距中心越远越弱 (声束宽度 beam_w)
                bw_g = np.exp(-((b - bc) ** 2) / (2 * beam_w ** 2))
                if bw_g < 0.15:
                    continue
                # 走时几何: 缺陷深度随波束线性渐变 (探伤角度投影)
                dd = depth0 + (b - bc) * 0.002 * rng.choice([-1, 1])
                w = max(6, int(width * 8))
                d_pix = min(max(int(dd * L) - w // 2, 4), L - w - 4)
                peak = amp * bw_g * depth_atten[d_pix]
                # 回波 = 高斯包络 × 载波 (超声回波典型形态, 载波周期 6-14 样本)
                cp = rng.uniform(6.0, 14.0)
                t = np.arange(w) - w / 2
                echo = np.sin(2 * np.pi * t / cp) * np.exp(-(t ** 2) / (2 * bw ** 2))
                B[p, b, d_pix:d_pix + w] += peak * echo

    B *= g
    return B.astype(np.float32), label.astype(np.int64)


def _narrowband_noise(rng: np.random.Generator, n_pos, n_beams, L, f0, amp) -> np.ndarray:
    """窄带结构噪声: 低频幅度场 × 快速载波, 幅度由 f0 调制的正弦结构。"""
    # 慢变幅度场 (散斑包络), 平滑插值
    r = rng.standard_normal((n_pos, n_beams, max(8, L // 8))).astype(np.float32)
    # 上采样到 L (平滑)
    import scipy.ndimage as ndi
    r = ndi.zoom(r, (1, 1, L / r.shape[2]), order=1)
    # 深度维微调制
    mod = 0.6 + 0.8 * np.abs(np.sin(np.arange(L) * f0 * 0.4))
    return (amp * r * mod).astype(np.float32)


def default_coupon_cfgs(rng: np.random.Generator, n_coupons: int) -> list[dict]:
    """生成 n_coupons 个试件, 缺陷率在 0.05-0.7 区间分布 (复现 PP4/PP5/PP6/PP7 的悬殊)。"""
    cfgs = []
    # 缺陷率档位: 少量近零缺陷 + 大量中低 + 少量高 (模仿真实)
    rates = np.clip(np.random.default_rng(0).beta(1.5, 2.0, n_coupons), 0.03, 0.75)
    rates[: max(1, n_coupons // 6)] = np.random.default_rng(0).uniform(0.02, 0.05)  # 近零缺陷试件
    rng.shuffle(rates)
    for i in range(n_coupons):
        cfgs.append({
            "gain": float(np.exp(rng.uniform(-0.3, 0.4))),      # 试件灵敏度差异
            "atten": float(rng.uniform(0.0015, 0.0035)),         # 衰减差异 (减半以保留深处信号)
            "speckle": float(rng.uniform(0.25, 0.7)),            # 散斑能量 (压低以突出缺陷)
            "f0": float(rng.uniform(6.0, 10.0)),                 # 探头频率差异
            "bw": float(rng.uniform(14.0, 26.0)),                # 带宽差异
            "beam_w": float(rng.uniform(2.0, 4.5)),              # 声束宽度差异
            "def_amp": float(rng.uniform(5.0, 10.0)),            # 缺陷回波幅度 (大幅提高, 让回波压过散斑)
            "def_depth": float(rng.uniform(0.25, 0.45)),         # 缺陷深度差异
            "def_len": float(rng.uniform(25, 60)),               # 缺陷纵向长度差异
            "n_def": int(rng.integers(2, 5)),                    # 缺陷区段数
            "bottom": float(rng.uniform(1.5, 3.0)),              # 底面回波幅度
            "def_rate": float(rates[i]),
        })
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-coupons", type=int, default=12)
    ap.add_argument("--n-pos-per-coupon", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/synth_ut")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--viz", action="store_true", help="只出图检查物理合理性")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfgs = default_coupon_cfgs(rng, args.n_coupons)

    if args.viz:
        _viz(cfgs, rng)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_x, all_y, coupons, meta_per_coupon = [], [], [], []
    for i, cfg in enumerate(cfgs):
        n_pos = args.n_pos_per_coupon
        X, y = make_coupon(rng, n_pos, cfg)
        all_x.append(X); all_y.append(y)
        coupons.append(np.full(n_pos, i, dtype=np.int64))
        meta_per_coupon.append({**{k: v for k, v in cfg.items() if not isinstance(v, np.ndarray)},
                                "n_pos": n_pos, "n_pos_pos": int(y.sum())})
    X = np.concatenate(all_x, axis=0)            # (N,49,512)
    y = np.concatenate(all_y, axis=0)            # (N,)
    c = np.concatenate(coupons, axis=0)          # (N,)
    np.save(args.out / "ascans.npy", X)
    np.save(args.out / "labels.npy", y)
    np.save(args.out / "coupons.npy", c)
    with open(args.out / "meta.json", "w") as fh:
        json.dump({
            "n_samples": int(X.shape[0]), "n_beams": 49, "target_len": 512,
            "defect_rate": float(y.mean()),
            "n_coupons": args.n_coupons, "seed": args.seed,
            "per_coupon": meta_per_coupon,
            "defect_rate_range": [float(min(m["def_rate"] for m in meta_per_coupon)),
                                  float(max(m["def_rate"] for m in meta_per_coupon))],
        }, fh, indent=2, ensure_ascii=False)
    print(f"Synth-UT 生成完成 | {X.shape} | 缺陷率 {y.mean():.3f} | "
          f"{time.time()-t0:.1f}s | -> {args.out}")
    print("  per-coupon 缺陷率:", [round(m['def_rate'], 3) for m in meta_per_coupon])


def _viz(cfgs, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(3, len(cfgs))
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.4 * n))
    axes = axes.reshape(n, 2)
    # 用全局统一 vlim 以便跨图对比
    samp_X, _ = make_coupon(rng, 50, cfgs[0])
    absmax = float(np.percentile(np.abs(samp_X), 99.5))
    for i in range(n):
        X, y = make_coupon(rng, 300, cfgs[i])
        pos_i = np.where(y == 1)[0]
        neg_i = np.where(y == 0)[0]
        for col, idx, title in [(0, int(pos_i[len(pos_i) // 2]), "defect"),
                                (1, int(neg_i[len(neg_i) // 2]), "clean")]:
            ax = axes[i, col]
            ax.imshow(X[idx].T, aspect="auto", cmap="seismic", origin="lower",
                      vmin=-absmax, vmax=absmax)
            ax.set_title(f"coupon{i} {title} (def_rate={cfgs[i]['def_rate']:.2f})")
            ax.set_xlabel("beam"); ax.set_ylabel("depth")
    fig.suptitle(f"Synth-UT B-scan samples | global vlim ±{absmax:.1f}", y=1.001)
    fig.tight_layout()
    out = REPO / "experiments/results/synth_ut_viz.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("viz ->", out)


if __name__ == "__main__":
    main()
