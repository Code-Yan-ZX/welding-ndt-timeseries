#!/usr/bin/env python3
"""M0-3 聚合 + GO 判据（P-long vs W→P，非PP4 逐折 mean ROC-AUC）。

读入 ``m0_3_loocv_{P-long,WP}_seed{seed}_e{ext}_t{tgt}*.json``。

pilot GO（seed 42，全 folds）：
  1. W→P − P-long 非PP4 mean ROC-AUC >= +0.01；
  2. PP3/PP5/PP6/PP7 至少 3 折不下降（W→P >= P-long）；
  3. 不出现单折下降超过 0.05；
  4. 结果不是由 PP4 或 pooled 指标驱动（pooled 仅参考）。

正式 GO（seeds 42/43/44）：
  1. mean[W→P − P-long] >= +0.01；
  2. >=2/3 seed 为正；
  3. 非PP4 多数折没有系统性退化。

输出 ``experiments/results/m0_3_aggregate.json`` + ``.md``。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from m0_3_loocv import per_exp_path  # noqa: E402

RESULTS_DIR = REPO / "experiments" / "results"
NP4 = ["PP3", "PP5", "PP6", "PP7"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _load(p: Path) -> dict | None:
    if p.exists():
        return json.loads(p.read_text())
    return None


def _fold_map(res: dict) -> dict[str, dict]:
    return {f["test_coupon"]: f for f in res.get("folds", [])}


def aggregate(ext_steps: int, tgt_steps: int, seeds: list[int], tag: str | None,
              smoke: bool) -> dict:
    per_seed = {}
    fold_deltas: dict[str, list[float]] = {c: [] for c in NP4}
    for s in seeds:
        pl = _load(per_exp_path("P-long", s, ext_steps, tgt_steps, tag, smoke))
        wp = _load(per_exp_path("WP", s, ext_steps, tgt_steps, tag, smoke))
        row = {"seed": s}
        if pl and wp:
            fpl, fwp = _fold_map(pl), _fold_map(wp)
            row["P-long"] = pl["nonPP4_mean_auc"]
            row["WP"] = wp["nonPP4_mean_auc"]
            row["d_nonPP4"] = round(wp["nonPP4_mean_auc"] - pl["nonPP4_mean_auc"], 4)
            row["pooled"] = {"P-long": pl["pooled_auc"], "WP": wp["pooled_auc"],
                             "d": round(wp["pooled_auc"] - pl["pooled_auc"], 4)}
            row["pp4"] = {"P-long": pl["pp4_auc"], "WP": wp["pp4_auc"],
                          "d": round(wp["pp4_auc"] - pl["pp4_auc"], 4)}
            row["per_fold_d"] = {}
            for c in NP4:
                d = round(fwp[c]["test_auc"] - fpl[c]["test_auc"], 4)
                row["per_fold_d"][c] = d
                fold_deltas[c].append(d)
            row["n_folds_no_drop"] = sum(1 for d in row["per_fold_d"].values() if d >= 0)
            row["max_fold_drop"] = min(row["per_fold_d"].values())
        per_seed[s] = row

    d_vals = [r["d_nonPP4"] for r in per_seed.values() if "d_nonPP4" in r]
    mean_d = float(np.mean(d_vals)) if d_vals else float("nan")
    n_pos = sum(1 for d in d_vals if d > 0)
    n_fold_no_drop = sum(1 for r in per_seed.values()
                         if r.get("n_folds_no_drop", 0) >= 3)
    fold_means = {c: round(float(np.mean(v)), 4) for c, v in fold_deltas.items()}
    max_single_fold_drop = min(min(v) for v in fold_deltas.values()
                               if v) if any(fold_deltas.values()) else 0.0

    is_pilot = len(seeds) == 1
    if is_pilot:
        go1 = mean_d >= 0.01
        go2 = (per_seed.get(seeds[0]) or {}).get("n_folds_no_drop", 0) >= 3
        go3 = max_single_fold_drop > -0.05
        go4 = True                     # 见说明：pooled/PP4 只参考，逐折非PP4 为主
        pilot_go = bool(go1 and go2 and go3 and go4)
        formal_go = None
        verdict = ("pilot GO 判据通过：W→P−P-long 非PP4 mean=%.4f ≥ +0.01，"
                   "%d/4 非PP4 折未下降，最大单折下降 %.4f > −0.05"
                   % (mean_d, per_seed.get(seeds[0], {}).get("n_folds_no_drop", 0),
                      max_single_fold_drop)) if pilot_go else \
            ("pilot GO 判据不通过：W→P−P-long 非PP4 mean=%.4f（需 ≥+0.01），"
             "未下降折 %d/4（需 ≥3），最大单折下降 %.4f（需 >−0.05）。"
             "立即停止；不调参；不跑 3 seeds；结论写为'少量外部真实焊缝 FMC "
             "未带来稳定迁移'。" % (mean_d,
                                   per_seed.get(seeds[0], {}).get("n_folds_no_drop", 0),
                                   max_single_fold_drop))
    else:
        formal_go = bool(mean_d >= 0.01 and n_pos >= max(1, len(seeds) * 2 // 3)
                         and n_fold_no_drop >= max(1, len(seeds) // 2))
        pilot_go = None
        verdict = ("正式 GO 判据通过：mean[W→P−P-long]=%.4f ≥ +0.01，"
                   "%d/%d seed 为正，非PP4 多数折无系统性退化"
                   % (mean_d, n_pos, len(seeds))) if formal_go else \
            ("正式 GO 判据不通过：mean[W→P−P-long]=%.4f（需 ≥+0.01），"
             "%d/%d seed 为正（需 ≥2/3），多数折系统性退化检查失败。"
             % (mean_d, n_pos, len(seeds)))

    out = {
        "exp": "m0_3_aggregate", "run_type": "smoke" if smoke else "full",
        "ext_steps": ext_steps, "tgt_steps": tgt_steps,
        "model_seeds": list(seeds), "is_pilot": is_pilot,
        "main_metric": "nonPP4_fold_mean ROC-AUC (PP3/PP5/PP6/PP7)",
        "per_seed": per_seed,
        "mean_d_nonPP4": round(mean_d, 4),
        "n_seeds_wp_gt_plong": n_pos,
        "fold_mean_d": fold_means,
        "max_single_fold_drop": round(max_single_fold_drop, 4),
        "n_seeds_with_ge3_folds_no_drop": n_fold_no_drop,
        "pilot_go": pilot_go,
        "formal_go": formal_go,
        "verdict": verdict,
        "notes": [
            "pooled 与 PP4 仅参考，不驱动判据（主指标 = 非PP4 逐折均值）；",
            "判据失败时立即停止，不调参、不跑 3 seeds、不做补救实验；",
            "真实独立焊缝试件 < 10 时结论必须标注 exploratory。",
        ],
        "code_commit": git_commit(),
    }
    json_path = RESULTS_DIR / f"m0_3_aggregate{'_smoke' if smoke else ''}.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_markdown(out, json_path.with_suffix(".md"))
    print(f"-> {json_path}")
    print(verdict)
    return out


def write_markdown(agg: dict, path: Path) -> None:
    seeds = agg["model_seeds"]
    L = ["# M0-3 真实焊缝多源超声 SSL 聚合（P-long vs W→P）",
         "",
         f"- model seeds: {seeds}；data_seed=42；ext_steps={agg['ext_steps']} "
         f"tgt_steps={agg['tgt_steps']}（总 {agg['ext_steps'] + agg['tgt_steps']}）",
         f"- 主指标: {agg['main_metric']}",
         "",
         "## 逐 seed 结果",
         "",
         "| seed | P-long | W→P | Δ非PP4 | PP3 Δ | PP5 Δ | PP6 Δ | PP7 Δ | 未降折数 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        r = agg["per_seed"][s]
        if "d_nonPP4" not in r:
            continue
        pf = r["per_fold_d"]
        L.append(f"| {s} | {r['P-long']:.4f} | {r['WP']:.4f} | {r['d_nonPP4']:+.4f} "
                 f"| {pf['PP3']:+.4f} | {pf['PP5']:+.4f} | {pf['PP6']:+.4f} "
                 f"| {pf['PP7']:+.4f} | {r['n_folds_no_drop']}/4 |")
    L += ["",
          f"- mean Δ非PP4 = **{agg['mean_d_nonPP4']:+.4f}**"
          f"（正 seed {agg['n_seeds_wp_gt_plong']}/{len(seeds)}）",
          f"- 逐折平均 Δ: " + "，".join(f"{k}={v:+.4f}" for k, v in agg['fold_mean_d'].items()),
          f"- 最大单折下降 = {agg['max_single_fold_drop']:+.4f}",
          "",
          "## 判据",
          ""]
    if agg["is_pilot"]:
        L += [f"- pilot GO（seed42）：**{'通过' if agg['pilot_go'] else '不通过'}**",
              "  - W→P−P-long 非PP4 mean ≥ +0.01",
              "  - PP3/PP5/PP6/PP7 至少 3 折不下降",
              "  - 无单折下降 > 0.05",
              "  - 结果不由 PP4 / pooled 驱动"]
    else:
        L += [f"- 正式 GO（3 seeds）：**{'通过' if agg['formal_go'] else '不通过'}**",
              "  - mean[W→P−P-long] ≥ +0.01",
              "  - ≥2/3 seed 为正",
              "  - 非PP4 多数折无系统性退化"]
    L += ["",
          "## 结论",
          "",
          agg["verdict"],
          "",
          "> 措辞纪律：独立试件 < 10 时必须标注 exploratory external pretraining "
          "source；判据失败直接停止，不调参、不跑 3 seeds；不用 pooled/PP4 替代"
          "逐折主指标。"]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-steps", type=int, default=None)
    ap.add_argument("--tgt-steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", type=Path, default=REPO / "configs/m0_3_weld_ut.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from wndt.utils.config import load_config
    cfg = load_config(args.config)
    if args.smoke:
        args.ext_steps = args.ext_steps or 20
        args.tgt_steps = args.tgt_steps or 20
    if args.ext_steps is None or args.tgt_steps is None:
        args.ext_steps = int(cfg.pretrain.pilot_external_steps)
        args.tgt_steps = int(cfg.pretrain.pilot_target_steps)
    seeds = args.seeds or [42]
    aggregate(args.ext_steps, args.tgt_steps, seeds, args.tag, args.smoke)


if __name__ == "__main__":
    main()
