#!/usr/bin/env python3
"""M0-2C 三种子聚合 + 最终判据（ECT 迁移 + PAUT 保持）。

读入（full，steps=10000）：
- ECT probe   : ``m0_2c_ect_probe_{E,PE}_seed{42,43,44}_s10000.json``
  （fold mean ROC-AUC / PR-AUC / balanced accuracy；transductive probe）
- PAUT 回测   : ``m0_2c_paut_retention_{P,PE}_seed{42,43,44}_s10000.json``
  （非PP4 逐折均值）

判据：
- ECT 迁移：mean[P→E − E] >= +0.01 **且** >=2/3 seed 为正；
- PAUT 保持：mean[P→E − P] >= −0.01（否则 = 灾难性遗忘）；
- 两者都过 -> 顺序训练得到更通用的 NDT 编码器；任一失败直接停止，
  不调参、不做 replay/freeze 补救。

输出 ``experiments/results/m0_2c_aggregate.json`` + ``.md``。
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

from m0_2c_ect_probe import per_exp_path as probe_path  # noqa: E402
from m0_2c_paut_retention import per_exp_path as retention_path  # noqa: E402

RESULTS_DIR = REPO / "experiments" / "results"
SEEDS = [42, 43, 44]
STEPS = 10000


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


def aggregate(steps: int = STEPS, seeds: list[int] | None = None,
              smoke: bool = False) -> dict:
    seeds = seeds or SEEDS
    # ---------------- ECT probe ----------------
    ect = {}
    for s in seeds:
        e = _load(probe_path("E", s, steps, None, smoke))
        pe = _load(probe_path("PE", s, steps, None, smoke))
        ect[s] = {"E": e, "PE": pe}
    d_auc = []
    d_pr = []
    d_bacc = []
    per_seed = {}
    for s in seeds:
        e, pe = ect[s]["E"], ect[s]["PE"]
        row = {"seed": s}
        if e and pe:
            row["E"] = {"roc_auc": e["fold_mean_roc_auc"],
                        "pr_auc": e["fold_mean_pr_auc"],
                        "balanced_acc": e["fold_mean_balanced_acc"]}
            row["PE"] = {"roc_auc": pe["fold_mean_roc_auc"],
                         "pr_auc": pe["fold_mean_pr_auc"],
                         "balanced_acc": pe["fold_mean_balanced_acc"]}
            row["d_roc_auc"] = round(pe["fold_mean_roc_auc"] - e["fold_mean_roc_auc"], 4)
            row["d_pr_auc"] = round(pe["fold_mean_pr_auc"] - e["fold_mean_pr_auc"], 4)
            row["d_balanced_acc"] = round(
                pe["fold_mean_balanced_acc"] - e["fold_mean_balanced_acc"], 4)
            d_auc.append(row["d_roc_auc"])
            d_pr.append(row["d_pr_auc"])
            d_bacc.append(row["d_balanced_acc"])
        per_seed[s] = row
    n_pos = sum(1 for x in d_auc if x > 0)
    avg_d_auc = float(np.mean(d_auc)) if d_auc else float("nan")
    avg_d_pr = float(np.mean(d_pr)) if d_pr else float("nan")
    avg_d_bacc = float(np.mean(d_bacc)) if d_bacc else float("nan")
    ect_criterion = {"min_delta": 0.01, "min_positive_seeds": max(1, len(seeds) * 2 // 3)}
    ect_met = (avg_d_auc >= 0.01) and (n_pos >= ect_criterion["min_positive_seeds"])

    # ---------------- PAUT retention ----------------
    ret = {}
    for s in seeds:
        p = _load(retention_path("P", s, steps, None, smoke))
        pe = _load(retention_path("PE", s, steps, None, smoke))
        ret[s] = {"P": p, "PE": pe}
    d_ret = []
    ret_per_seed = {}
    for s in seeds:
        p, pe = ret[s]["P"], ret[s]["PE"]
        row = {"seed": s}
        if p and pe and p.get("nonPP4_mean_auc") is not None and \
                pe.get("nonPP4_mean_auc") is not None:
            row["P"] = p["nonPP4_mean_auc"]
            row["PE"] = pe["nonPP4_mean_auc"]
            row["d_nonPP4"] = round(pe["nonPP4_mean_auc"] - p["nonPP4_mean_auc"], 4)
            d_ret.append(row["d_nonPP4"])
        ret_per_seed[s] = row
    avg_d_ret = float(np.mean(d_ret)) if d_ret else float("nan")
    ret_met = avg_d_ret >= -0.01

    # ---------------- 结论 ----------------
    if ect_met and ret_met:
        verdict = ("顺序训练（PAUT SSL -> ECT SSL）得到更通用的 NDT 编码器："
                   f"ECT 迁移判据通过（平均 P→E−E={avg_d_auc:+.4f} ≥ +0.01，"
                   f"{n_pos}/{len(seeds)} seed 为正），PAUT 保持判据通过"
                   f"（平均 P→E−P={avg_d_ret:+.4f} ≥ −0.01）。")
    elif not ect_met:
        verdict = ("ECT 迁移判据不通过"
                   f"（平均 P→E−E={avg_d_auc:+.4f}，正 seed {n_pos}/{len(seeds)}）："
                   "直接停止，不调参、不做 replay/freeze 补救实验。")
    else:
        verdict = ("PAUT 保持判据不通过"
                   f"（平均 P→E−P={avg_d_ret:+.4f} < −0.01）：结论 = 灾难性遗忘，"
                   "直接停止，不调参、不做 replay/freeze 补救实验。")

    out = {
        "exp": "m0_2c_aggregate", "run_type": "smoke" if smoke else "full",
        "steps": steps, "model_seeds": list(seeds),
        "ect_probe": {
            "run_type_note": "transductive_unlabeled representation probe；"
                             "SSL 使用全部 ECT 无标注视图后冻结；group 5 折按 "
                             "config/specimen proxy，同一配置组绝不跨 fold；"
                             "不得写成严格 cross-group 泛化。",
            "per_seed": per_seed,
            "avg_d_roc_auc": round(avg_d_auc, 4),
            "avg_d_pr_auc": round(avg_d_pr, 4),
            "avg_d_balanced_acc": round(avg_d_bacc, 4),
            "n_seeds_pe_gt_e": n_pos,
            "criterion": ect_criterion,
            "met": ect_met,
        },
        "paut_retention": {
            "metric": "nonPP4_fold_mean (PP3/PP5/PP6/PP7)",
            "per_seed": ret_per_seed,
            "avg_d_nonPP4": round(avg_d_ret, 4),
            "criterion": {"min_delta": -0.01},
            "met": ret_met,
            "catastrophic_forgetting": avg_d_ret < -0.01,
        },
        "verdict": verdict,
        "code_commit": git_commit(),
    }
    json_path = RESULTS_DIR / f"m0_2c_aggregate{'_smoke' if smoke else ''}.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_markdown(out, json_path.with_suffix(".md"))
    print(f"-> {json_path}")
    print(verdict)
    return out


def write_markdown(agg: dict, path: Path) -> None:
    e = agg["ect_probe"]
    r = agg["paut_retention"]
    seeds = agg["model_seeds"]
    L = ["# M0-2C ECT 顺序 SSL 聚合（E vs P→E，三种子）",
         "",
         f"- model seeds: {seeds}；data_seed=42；steps={agg['steps']}",
         f"- ECT probe: {e['run_type_note']}",
         f"- PAUT 回测指标: {r['metric']}",
         "",
         "## 1. ECT probe（fold mean）",
         "",
         "| seed | E ROC-AUC | P→E ROC-AUC | Δ ROC-AUC | Δ PR-AUC | Δ bAcc |",
         "|---|---|---|---|---|---|"]
    for s in seeds:
        row = e["per_seed"][s]
        if "E" not in row:
            continue
        L.append(f"| {s} | {row['E']['roc_auc']:.4f} | {row['PE']['roc_auc']:.4f} "
                 f"| {row['d_roc_auc']:+.4f} | {row['d_pr_auc']:+.4f} "
                 f"| {row['d_balanced_acc']:+.4f} |")
    L += ["",
          f"- 平均 Δ ROC-AUC = **{e['avg_d_roc_auc']:+.4f}**"
          f"（Δ PR-AUC = {e['avg_d_pr_auc']:+.4f}，Δ bAcc = {e['avg_d_balanced_acc']:+.4f}）",
          f"- P→E > E 的 seed 数：{e['n_seeds_pe_gt_e']}/{len(seeds)}",
          f"- 判据（mean ≥ +0.01 且 ≥2/3 seed 正）：{'通过' if e['met'] else '不通过'}",
          "",
          "## 2. PAUT 回测（非PP4 逐折均值）",
          "",
          "| seed | P | P→E | Δ |",
          "|---|---|---|---|"]
    for s in seeds:
        row = r["per_seed"][s]
        if "P" not in row:
            continue
        L.append(f"| {s} | {row['P']:.4f} | {row['PE']:.4f} | {row['d_nonPP4']:+.4f} |")
    L += ["",
          f"- 平均 Δ = **{r['avg_d_nonPP4']:+.4f}**",
          f"- 判据（mean ≥ −0.01）：{'通过（保持）' if r['met'] else '不通过'}",
          f"- 灾难性遗忘：{'是' if r.get('catastrophic_forgetting') else '否'}",
          "",
          "## 3. 结论",
          "",
          f"{agg['verdict']}",
          "",
          "> 措辞纪律：transductive probe 不得写成严格 cross-group 泛化；"
          "若判据失败直接停止，不做 replay/freeze 补救；不用 pooled 替代逐折主指标。"]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    aggregate(args.steps, args.seeds, args.smoke)


if __name__ == "__main__":
    main()
