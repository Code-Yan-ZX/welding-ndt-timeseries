#!/usr/bin/env python3
"""M1-2: amplitude-only shortcut baseline（Shortcut Decomposition 第 2 步）。

**不使用任何空间纹理**。每个扫描只提取全局标量统计（native grid + valid mask，
不做 resize/不逐像素输入），Real/Imag/Magnitude/Phase 分别计算：

  mean / std / median / IQR / min / max / ptp / RMS / q05 q25 q75 q95 /
  dynamic-range(q95-q05)

Phase 用圆统计（circular mean/std via exp(iφ)）。另加少量全局频谱统计
（log-magnitude 2D FFT 的 mean/std/max，per 频率）作为对照 arm——注意 FFT 统计
已含弱空间信息，因此与纯幅值 arm 分开报告。

模型 LR / RF；协议 P0 / P1 / P3::HP-U300 / P3::Kohlegelege。

目的：仅凭"每扫描一个幅值直方图摘要"能走多远。若 P1 接近 pilot CNN 而 P3 仍差，
说明问题不只是幅值 scale；若 amplitude-only P1≈CNN-P1，则 pilot 的 CNN 性能
大部分可由幅值 shortcut 解释。

输出: experiments/results/eddycus_m1/amplitude_shortcut.csv + _summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.eddycus import _load_native_grid, BlockingError  # noqa: E402

OUT = REPO / "experiments" / "results" / "eddycus_m1"
SCAN_TABLE = REPO / "experiments" / "results" / "ndt_pilot" / "eddycus_scan_table.csv"
SPLIT_DIR = REPO / "data" / "manifests" / "eddycus_pilot_splits"
H5_DIR = REPO / "data" / "raw" / "EddyCus-HDF5" / "output"
SEED = 42
PROTOCOLS = [
    ("P0", "P0.json"),
    ("P1", "P1.json"),
    ("P3::HP-U300", "P3__HP-U300_122C.json"),
    ("P3::Kohlegelege", "P3__KohlegelegeST50g.json"),
]
QUANTILES = (0.05, 0.25, 0.75, 0.95)


def circular_stats(phi: np.ndarray) -> tuple[float, float]:
    """phi in [-pi, pi] → (circular mean, circular resultant-based std)。"""
    c = np.exp(1j * phi)
    r = np.abs(c.mean())
    mu = float(np.angle(c.mean())) if r > 1e-12 else 0.0
    std = float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12)))))
    return mu, std


def amplitude_features(g: np.ndarray, with_fft: bool) -> dict[str, float]:
    """g: (8,H,W) 4 频率 × (I,Q)，含空洞（NaN→0 已由 loader 处理则按原样）。

    只算逐扫描全局标量——**无任何逐像素空间模式输入模型**。
    """
    feats: dict[str, float] = {}
    for f in range(4):
        i_ch, q_ch = g[2 * f], g[2 * f + 1]
        mag = np.hypot(i_ch, q_ch)
        phi = np.arctan2(q_ch, i_ch)
        for name, arr in (("I", i_ch), ("Q", q_ch), ("mag", mag)):
            a = arr.ravel()
            q05, q25, q75, q95 = np.quantile(a, QUANTILES)
            feats[f"f{f + 1}_{name}_mean"] = float(a.mean())
            feats[f"f{f + 1}_{name}_std"] = float(a.std())
            feats[f"f{f + 1}_{name}_median"] = float(np.median(a))
            feats[f"f{f + 1}_{name}_iqr"] = float(q75 - q25)
            feats[f"f{f + 1}_{name}_min"] = float(a.min())
            feats[f"f{f + 1}_{name}_max"] = float(a.max())
            feats[f"f{f + 1}_{name}_ptp"] = float(a.max() - a.min())
            feats[f"f{f + 1}_{name}_rms"] = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
            feats[f"f{f + 1}_{name}_q05"] = float(q05)
            feats[f"f{f + 1}_{name}_q25"] = float(q25)
            feats[f"f{f + 1}_{name}_q75"] = float(q75)
            feats[f"f{f + 1}_{name}_q95"] = float(q95)
            feats[f"f{f + 1}_{name}_dynrange"] = float(q95 - q05)
        mu, sd = circular_stats(phi.ravel())
        feats[f"f{f + 1}_phase_circmean"] = mu
        feats[f"f{f + 1}_phase_circstd"] = sd
        if with_fft:
            logmag = np.log1p(np.clip(mag, 0, None))
            fft = np.abs(np.fft.fft2(logmag))
            feats[f"f{f + 1}_fft_mean"] = float(fft.mean())
            feats[f"f{f + 1}_fft_std"] = float(fft.std())
            feats[f"f{f + 1}_fft_max"] = float(fft.max())
    return feats


def build(with_fft: bool) -> tuple[pd.DataFrame, pd.Index]:
    df = pd.read_csv(SCAN_TABLE)
    sig = df[df.has_signal].reset_index(drop=True)
    rows, ok_idx = [], []
    for k, row in sig.iterrows():
        try:
            g, valid, _ = _load_native_grid(H5_DIR / row.file)
        except BlockingError:
            continue
        rows.append({"scan_id": row.scan_id, **amplitude_features(g, with_fft)})
        ok_idx.append(k)
    return pd.DataFrame(rows), sig.loc[ok_idx, "scan_id"]


def eval_protocol(X: pd.DataFrame, y: np.ndarray, split: np.ndarray) -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, va, te = ((split == p) for p in ("train", "val", "test"))
    rows = []
    models = {
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RF": RandomForestClassifier(n_estimators=500, random_state=SEED,
                                     class_weight="balanced", n_jobs=-1),
    }
    for name, m in models.items():
        m.fit(X[tr], y[tr].astype(int))
        s = m.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            rows.append({"model": name, "auc": float(roc_auc_score(y[te], s)),
                         "n_test": int(te.sum()), "n_def_test": int(y[te].sum()),
                         "n_clean_test": int((~y[te]).sum())})
        else:
            rows.append({"model": name, "auc": None, "one_class": True,
                         "n_test": int(te.sum()), "n_def_test": int(y[te].sum())})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splits: dict[str, np.ndarray] = {}
    for pname, fname in PROTOCOLS:
        sp = json.loads((SPLIT_DIR / fname).read_text())["splits"]
        splits[pname] = sp

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for tag, with_fft in (("amp_only", False), ("amp_plus_fft", True)):
        feats, ids = build(with_fft)
        y = pd.read_csv(SCAN_TABLE).set_index("scan_id").loc[ids, "defect_present"].to_numpy()
        for pname in splits:
            split = np.array([splits[pname][s] for s in ids])
            for r in eval_protocol(feats.drop(columns=["scan_id"]), y, split):
                all_rows.append({"arm": tag, "protocol": pname, **r})
            print(f"[{tag}] {pname} done", flush=True)
        summaries[tag] = {"n_features": feats.shape[1] - 1}

    res = {"seed": SEED, "runtime_s": round(time.time() - t0, 1),
           "feature_note": "全局标量统计（valid native grid，无 resize，无逐像素输入）；"
                           "amp_plus_fft 额外含 log-mag 2D-FFT 全局统计（弱空间信息，仅对照）",
           **summaries, "results": all_rows}
    pd.DataFrame(all_rows).to_csv(OUT / "amplitude_shortcut.csv", index=False)
    (OUT / "amplitude_shortcut_summary.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string())


if __name__ == "__main__":
    main()
