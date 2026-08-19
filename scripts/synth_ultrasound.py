#!/usr/bin/env python3
"""物理启发的程序化超声 B-scan 合成器（焊接 PAUT 样式）—— Synth-UT。

术语约定（Protocol V2 / M0-1.5）：
  本工具生成的是 **physics-inspired procedural synthetic data**
  （物理启发的程序化合成数据）。它模拟真实超声传播的若干要素（散斑/回波/
  走时/底面），但**不是 CIVA 级或物理保真仿真**——不要宣称其与真实超声
  传播的定量一致。任何报告/README 中凡提到本工具输出，一律使用该措辞。

动机 (2026-08-18, 新方向探索): PAUT 真实数据只有 5 试件 / 3000 位置(多视角 11980),
缺陷率 0.5%-76% 与试件身份强耦合, P0-P6 全杠杆证伪, 天花板在表征级 0.58。
P4/P6 报告点名翻盘路径 = "物理启发的合成数据教缺陷回波物理"。P5 已证伪"2D 高斯峰
注入"太假不迁移 (inj_acc 0.998 但下游 0.545<0.579)。本合成器模拟:

  1. 结构噪声散斑: 带通噪声(窄带中心频率 f0), 幅度随深度指数衰减(晶粒散射)
  2. 缺陷回波: 高斯包络正弦峰(反射回波), 幅度随深度衰减(声程衰减)
  3. 声束横向展宽: 缺陷回波幅度随 |beam-center| 高斯衰减 (声束宽度决定)
  4. 走时几何: 相邻波束回波深度有线性偏移 (线扫/扇扫投影)
  5. 底面回波: 板厚深度处的强回波 (无损试件唯一稳定特征)
  6. 试件级参数: 增益 / 衰减系数 / 散斑能量 / 缺陷幅度 / 缺陷率 —— 每试件不同

两种抽样模式 (M0-1.5 新增):
  --mode orthogonal (默认): coupon style 参数 (gain/atten/speckle/f0/bw/beam_w/
      bottom) 与缺陷参数 (def_rate/def_type/def_depth/def_amp) **独立采样**,
      缺陷属性不绑定 style —— 用于生成"风格多样但标签统计与风格无关"的数据。
  --mode confounded: 缺陷率与 style 耦合 (如高增益/低散斑试件同时配高缺陷率),
      仅用于**诊断复现**真实数据"缺陷率-试件耦合"挑战, 不得作为无偏预训练数据。

def_rate 语义 (M0-1.5 修复): coupon_cfg.def_rate **实际控制标签** ——
make_coupon 会迭代生成缺陷区段直到正样本占比达到 def_rate (容差见
--tol-def-rate), 并在 meta.json 同时保存 target_defect_rate 与
actual_defect_rate。每种缺陷类型/深度/幅值会分布在多个 coupon style 中
(见 _assign_defect_profiles)。

输出与 data/processed/paut/ascans.npy 同格式: (N, 49, 512) float32。
标签 = 位置级 0/1 (该位置是否落在缺陷区段), 与 meta_label.npy 同格式。

Usage:
  python scripts/synth_ultrasound.py --n-coupons 12 --n-pos-per-coupon 1000 \
      --out data/processed/synth_ut --seed 42
  python scripts/synth_ultrasound.py --mode confounded --n-coupons 12 --seed 42
  python scripts/synth_ultrasound.py --viz  # 出图检查物理合理性
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import numpy as np

RNG_GLOBAL = None  # 由 seed 初始化的全局 RNG

DEFECT_TYPES = ["porosity", "lack_of_fusion", "slag_inclusion", "crack", "edm_notch"]


def gauss_env(w, f0, sigma):
    """高斯包络正弦回波 (长度 w): sin(2π f0 t) * exp(-t²/(2σ²))。"""
    t = np.arange(w) - w / 2
    return np.sin(2 * np.pi * f0 * t / w) * np.exp(-(t ** 2) / (2 * sigma ** 2))


def make_coupon(rng: np.random.Generator, n_pos: int, coupon_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """生成一个合成试件 (焊缝) 的 B-scan 序列 + 位置级标签。

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
      n_def     : 缺陷区段数上限
      def_type  : 缺陷类型 (porosity / lack_of_fusion / ...)
      bottom    : 底面回波幅度
      def_rate  : **目标缺陷率** (0-1)。M0-1.5 修复: 实际控制标签。
                  迭代生成区段直到正样本占比达到目标。
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
    def_type = coupon_cfg.get("def_type", "porosity")
    bottom = coupon_cfg.get("bottom", 2.0)
    target_rate = float(coupon_cfg.get("def_rate", 0.2))

    depth = np.arange(L, dtype=np.float32) / L
    depth_atten = np.exp(-atten * np.arange(L, dtype=np.float32))

    # 结构噪声散斑: 窄带带通噪声 (零均值), 幅度随深度衰减
    speck_field = _narrowband_noise(rng, n_pos, n_beams, L, f0, speckle)
    B += speck_field * depth_atten[None, None, :]

    # 底面回波: 板厚深度处的强峰 (所有位置一致)
    if bottom > 0:
        bottom_w = 48
        k = int(L * 0.95) - bottom_w // 2
        B[:, :, k:k + bottom_w] += (bottom * depth_atten[k] *
                                    gauss_env(bottom_w, f0 * 1.0, bw)[None, None, :])

    # 缺陷区段: 每个区段 = 连续 len 个位置, 缺陷回波跨 beam_w 波束。
    # M0-1.5 修复: 以 target_rate 为迭代目标 —— 生成区段直到正样本占比达标,
    # 而不是由 n_def/def_len 隐式决定标签。
    # 精确控制: 先算目标正样本数, 逐段放置, 最后一段截断到目标, 避免过冲。
    n_target = int(round(n_pos * target_rate))
    if n_target <= 0:
        B *= g
        return B.astype(np.float32), label.astype(np.int64)

    avail = np.linspace(int(L * 0.12), int(L * 0.8), n_beams).astype(int)
    rng.shuffle(avail)
    seg = 0
    placed = 0
    # 安全上限: 每段至少 8 个位置, 迭代次数有界; 不设 n_seg_max 硬上限,
    # 否则高 target_rate (如 0.59) 在少段数下永远无法达标。
    max_iters = n_pos // 8 + 1
    while placed < n_target and seg < max_iters:
        seg_len = max(8, int(def_len * (0.7 + 0.6 * rng.random())))
        seg_len = min(seg_len, n_target - placed)
        if seg_len < 8:
            break
        d0 = max(0, rng.integers(0, n_pos - seg_len))
        bc = rng.integers(4, n_beams - 4)
        depth0 = def_depth + 0.15 * rng.uniform(-0.5, 0.5)
        # 缺陷在深度轴的形态: 气孔=尖锐峰 / 裂纹/未熔合=跨深度展宽
        if def_type in ("crack", "lack_of_fusion"):
            kind = "crack"
        elif def_type == "edm_notch":
            kind = "blob"
        else:
            kind = "blob"
        width = rng.uniform(0.5, 1.5) if kind == "blob" else rng.uniform(2.5, 5.0)
        amp = def_amp * (0.7 + 0.6 * rng.random())
        for p in range(d0, min(d0 + seg_len, n_pos)):
            if label[p]:
                continue
            label[p] = 1
            placed += 1
            rel = (p - d0) / max(1, seg_len - 1)  # 0..1 沿缺陷纵向
            for b in range(n_beams):
                bw_g = np.exp(-((b - bc) ** 2) / (2 * beam_w ** 2))
                if bw_g < 0.15:
                    continue
                dd = depth0 + (b - bc) * 0.002 * rng.choice([-1, 1])
                w = max(6, int(width * 8))
                d_pix = min(max(int(dd * L) - w // 2, 4), L - w - 4)
                peak = amp * bw_g * depth_atten[d_pix]
                cp = rng.uniform(6.0, 14.0)
                t = np.arange(w) - w / 2
                echo = np.sin(2 * np.pi * t / cp) * np.exp(-(t ** 2) / (2 * bw ** 2))
                B[p, b, d_pix:d_pix + w] += peak * echo
        seg += 1

    B *= g
    return B.astype(np.float32), label.astype(np.int64)


def _narrowband_noise(rng: np.random.Generator, n_pos, n_beams, L, f0, amp) -> np.ndarray:
    """窄带结构噪声: 低频幅度场 × 快速载波, 幅度由 f0 调制的正弦结构。"""
    r = rng.standard_normal((n_pos, n_beams, max(8, L // 8))).astype(np.float32)
    import scipy.ndimage as ndi
    r = ndi.zoom(r, (1, 1, L / r.shape[2]), order=1)
    mod = 0.6 + 0.8 * np.abs(np.sin(np.arange(L) * f0 * 0.4))
    return (amp * r * mod).astype(np.float32)


# ---------------------------------------------------------------------------
# 缺陷属性分配: 每种缺陷类型/深度/幅值分布在多个 coupon style 上。
# ---------------------------------------------------------------------------
def _assign_defect_profiles(rng: np.random.Generator, cfgs: list[dict], mode: str) -> None:
    """给每个 coupon 分配 (def_type, def_depth, def_amp, def_rate)。

    orthogonal: 缺陷属性从全局池里独立抽样, 与 style 参数无绑定;
                同一缺陷属性 (类型/深度/幅值) 会出现在多个 style 不同的
                coupon 上 (通过轮转分配 + 独立抽样保证)。
    confounded: 缺陷率与 style 耦合 —— 高增益/低散斑 coupon 配高缺陷率,
                复现真实数据的"缺陷率-试件身份"耦合 (仅诊断用)。
    """
    n = len(cfgs)
    # 每个 coupon 先独立抽样缺陷属性 (类型/深度/幅值) —— 两种 mode 都做,
    # 保证缺陷属性分布覆盖且不依赖 style。
    for i, c in enumerate(cfgs):
        c["def_type"] = DEFECT_TYPES[i % len(DEFECT_TYPES)]          # 轮转: 类型跨 style
        c["def_depth"] = float(rng.uniform(0.25, 0.45))
        c["def_amp"] = float(rng.uniform(5.0, 10.0))

    if mode == "confounded":
        # 故意把 style 与缺陷率绑起来: 复现真实耦合挑战 (仅诊断用)。
        # "脏"度 = 高增益 × 高幅值 / 低散斑 —— 这类 coupon 配高缺陷率。
        styles = [c["gain"] * c["def_amp"] / max(1e-6, c["speckle"]) for c in cfgs]
        order = np.argsort(styles)                       # 从"干净"到"脏"
        rates = np.linspace(0.03, 0.75, n)
        for rank, idx in enumerate(order):
            cfgs[idx]["def_rate"] = float(rates[rank])
        return

    # orthogonal: 缺陷率独立抽样, 与 style 无绑定
    for c in cfgs:
        c["def_rate"] = float(rng.uniform(0.03, 0.75))


def default_coupon_cfgs(rng: np.random.Generator, n_coupons: int, mode: str) -> list[dict]:
    """生成 n_coupons 个试件的 style 参数。

    缺陷率分布不再由这里硬编码 —— 由 ``_assign_defect_profiles`` 按 mode
    决定。style 参数 (gain/atten/speckle/f0/bw/beam_w/bottom) 与缺陷属性
    相互独立。
    """
    cfgs = []
    for i in range(n_coupons):
        cfgs.append({
            "gain": float(np.exp(rng.uniform(-0.3, 0.4))),      # 试件灵敏度差异
            "atten": float(rng.uniform(0.0015, 0.0035)),         # 衰减差异 (减半以保留深处信号)
            "speckle": float(rng.uniform(0.25, 0.7)),            # 散斑能量 (压低以突出缺陷)
            "f0": float(rng.uniform(6.0, 10.0)),                 # 探头频率差异
            "bw": float(rng.uniform(14.0, 26.0)),                # 带宽差异
            "beam_w": float(rng.uniform(2.0, 4.5)),              # 声束宽度差异
            "def_len": float(rng.uniform(25, 60)),               # 缺陷纵向长度差异
            "n_def": int(rng.integers(2, 5)),                    # 缺陷区段数上限
            "bottom": float(rng.uniform(1.5, 3.0)),              # 底面回波幅度
            # 缺陷属性由 _assign_defect_profiles 填充
        })
    _assign_defect_profiles(rng, cfgs, mode)
    return cfgs


def assert_defect_rate(actual: float, target: float, tol: float, label: str) -> None:
    """对每个 coupon 的实际缺陷率与目标值做容差断言。"""
    if abs(actual - target) > tol:
        raise AssertionError(
            f"{label} actual defect rate {actual:.4f} deviates from target "
            f"{target:.4f} by {abs(actual-target):.4f} > tol {tol}; "
            "def_rate did not control the label as expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-coupons", type=int, default=12)
    ap.add_argument("--n-pos-per-coupon", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/synth_ut")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["orthogonal", "confounded"], default="orthogonal",
                    help="orthogonal=默认, style 与缺陷属性独立; "
                         "confounded=仅诊断复现缺陷率-试件耦合")
    ap.add_argument("--tol-def-rate", type=float, default=0.05,
                    help="actual vs target defect rate 容差 (绝对值)")
    ap.add_argument("--viz", action="store_true", help="只出图检查物理合理性")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    global RNG_GLOBAL
    RNG_GLOBAL = rng
    cfgs = default_coupon_cfgs(rng, args.n_coupons, args.mode)

    if args.viz:
        _viz(cfgs, rng)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_x, all_y, coupons, meta_per_coupon = [], [], [], []
    for i, cfg in enumerate(cfgs):
        n_pos = args.n_pos_per_coupon
        X, y = make_coupon(rng, n_pos, cfg)
        # 容差断言: def_rate 必须实际控制标签
        actual = float(y.mean())
        assert_defect_rate(actual, cfg["def_rate"], args.tol_def_rate,
                           f"coupon {i}")
        all_x.append(X); all_y.append(y)
        coupons.append(np.full(n_pos, i, dtype=np.int64))
        meta_per_coupon.append({
            **{k: v for k, v in cfg.items() if not isinstance(v, np.ndarray)},
            "n_pos": n_pos,
            "target_defect_rate": float(cfg["def_rate"]),
            "actual_defect_rate": actual,
        })
    X = np.concatenate(all_x, axis=0)            # (N,49,512)
    y = np.concatenate(all_y, axis=0)            # (N,)
    c = np.concatenate(coupons, axis=0)          # (N,)
    np.save(args.out / "ascans.npy", X)
    np.save(args.out / "labels.npy", y)
    np.save(args.out / "coupons.npy", c)
    with open(args.out / "meta.json", "w") as fh:
        json.dump({
            "kind": "physics-inspired procedural synthetic data",  # M0-1.5 术语
            "mode": args.mode,
            "n_samples": int(X.shape[0]), "n_beams": 49, "target_len": 512,
            "defect_rate": float(y.mean()),
            "n_coupons": args.n_coupons, "seed": args.seed,
            "tol_def_rate": args.tol_def_rate,
            "per_coupon": meta_per_coupon,
            "target_defect_rate_range": [float(min(m["target_defect_rate"] for m in meta_per_coupon)),
                                         float(max(m["target_defect_rate"] for m in meta_per_coupon))],
            "actual_defect_rate_range": [float(min(m["actual_defect_rate"] for m in meta_per_coupon)),
                                         float(max(m["actual_defect_rate"] for m in meta_per_coupon))],
        }, fh, indent=2, ensure_ascii=False)
    print(f"Synth-UT 生成完成 | {X.shape} | 实际缺陷率 {y.mean():.3f} | "
          f"mode={args.mode} | {time.time()-t0:.1f}s | -> {args.out}")
    print("  per-coupon (target/actual):",
          [(round(m['target_defect_rate'], 3), round(m['actual_defect_rate'], 3))
           for m in meta_per_coupon])


def _viz(cfgs, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(3, len(cfgs))
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.4 * n))
    axes = axes.reshape(n, 2)
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
            ax.set_title(f"coupon{i} {title} (target_rate={cfgs[i]['def_rate']:.2f})")
            ax.set_xlabel("beam"); ax.set_ylabel("depth")
    fig.suptitle(f"Synth-UT B-scan samples | global vlim ±{absmax:.1f}", y=1.001)
    fig.tight_layout()
    out = REPO / "experiments/results/synth_ut_viz.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("viz ->", out)


if __name__ == "__main__":
    main()
