#!/usr/bin/env python
"""聚合 P0 全部 LOOCV 结果为对比表 experiments/results/paut_loocv_table.md。

读取 experiments/results/paut_loocv_seed42*.json (基线 ssf/encoder/rf、无增强对照、
4 单增强、all、dann), 按 (模型, 增强) 分行, 报告 AUC mean±std、每折 AUC、相对裸
SSF 的 delta。裸 SSF 基线 = --tag control 的 ssf (与基线同 AUC, 但带 val_scores)。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/results"
COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


def model_label(name: str, augment: dict | None) -> str:
    base = {"ssf": "SSF", "encoder_only": "encoder", "classic_rf": "RF",
            "dann": "DANN"}.get(name, name)
    if augment:
        return base + "+" + "+".join(sorted(augment.keys()))
    return base


def load_all():
    rows = []
    for jp in sorted(glob.glob(str(RES / "paut_loocv_seed42*.json"))):
        if "_smoke" in jp:
            continue
        data = json.loads(Path(jp).read_text())
        augment = data.get("augment")
        for mname, mdata in data["models"].items():
            agg = mdata["agg"]
            pf = agg["per_fold_auc"]
            # 非 PP4 AUC: 剔除退化折 (PP4 仅 3 正样本, AUC 纯噪声 ±0.03)
            non_pp4 = [pf[c] for c in ["PP3", "PP5", "PP6", "PP7"] if c in pf]
            non_pp4_mean = float(np.mean(non_pp4)) if non_pp4 else float("nan")
            rows.append({
                "file": Path(jp).name,
                "model": mname,
                "augment": augment,
                "label": model_label(mname, augment),
                "auc_mean": agg["auc_mean"],
                "auc_std": agg["auc_std"],
                "non_pp4": non_pp4_mean,
                "f1m": agg["f1_macro_mean"],
                "per_fold": pf,
            })
    return rows


def main():
    rows = load_all()
    # 裸 SSF 基线 AUC (优先 control, 否则基线 json 的 ssf)
    bare = next((r for r in rows if r["model"] == "ssf" and not r["augment"]
                 and "control" in r["file"]), None)
    if bare is None:
        bare = next((r for r in rows if r["model"] == "ssf" and not r["augment"]), None)
    base_auc = bare["auc_mean"] if bare else None

    lines = ["# PAUT LOOCV P0 汇总对比表", "",
             "5 折留一试件交叉验证 (PP3-PP7 轮流 test), 其余 4 试件 85/15 分层 train/val,",
             "per-fold 归一化 (无泄漏, 单次归一化), val 调阈值+早停。seed=42。", ""]
    if base_auc is not None:
        lines.append(f"**裸 SSF LOOCV AUC = {base_auc:.4f}** (跨试件泛化基线, "
                     f"远低于单点 PP7=0.626, 揭示跨试件过拟合)。")
        lines.append(f"成功门槛: SSF+增强+DANN 较裸 SSF 提升 ≥0.03 (即 AUC ≥ {base_auc+0.03:.4f})。")
        lines.append("")
    lines += ["## 总表", "",
              "| 模型 | AUC mean±std | 非PP4 AUC | Δ vs 裸SSF(非PP4) | F1m mean | PP3 | PP4 | PP5 | PP6 | PP7 | 来源 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    # 排序: 裸SSF, encoder, RF, 然后SSF增强变体, DANN
    def order(r):
        key = {"ssf": 0, "encoder_only": 1, "classic_rf": 2, "dann": 5}.get(r["model"], 9)
        if r["model"] == "ssf" and r["augment"]:
            key = 3 + (0 if "all" in r["augment"] or len(r["augment"]) > 1 else 1)
        return (key, r["label"])
    # 裸 SSF 非 PP4 基线
    bare_np4 = bare["non_pp4"] if bare else None
    for r in sorted(rows, key=order):
        delta = r["auc_mean"] - base_auc if base_auc is not None else float("nan")
        dlt = f"{delta:+.3f}" if base_auc is not None else "-"
        dnp4 = (r["non_pp4"] - bare_np4) if bare_np4 is not None else float("nan")
        dnp4_s = f"{dnp4:+.3f}" if bare_np4 is not None else "-"
        pf = r["per_fold"]
        lines.append(f"| {r['label']} | {r['auc_mean']:.3f}±{r['auc_std']:.3f} | {r['non_pp4']:.3f} | "
                     f"{dnp4_s} | {r['f1m']:.3f} | {pf.get('PP3',0):.3f} | {pf.get('PP4',0):.3f} | "
                     f"{pf.get('PP5',0):.3f} | {pf.get('PP6',0):.3f} | {pf.get('PP7',0):.3f} | "
                     f"`{r['file']}` |")
    lines += ["", "## 备注", "",
              "- PP4 仅 3 个局部缺陷位置 (0.5%): 作 test 时 AUC 纯噪声 (±0.03 随机波动), "
              "使含 PP4 的 mean±std 不可靠。**非PP4 AUC** (剔除 PP4) 是更可信的跨试件泛化指标。",
              "- 增强变体: beam_dropout/time_shift/amp_jitter/gaussian_noise (单独) 与 all (四者全开), 仅作用于训练集。",
              "- DANN: SSF 编码器 + 梯度反转层 + 试件域判别器 (域=训练集4试件), 推理只用标签头。",
              "- 每折 val/test scores 已存于各 JSON, 供 temperature scaling 校准分析 (P0-5)。"]
    out = RES / "paut_loocv_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")
    print(f"裸 SSF AUC = {base_auc:.4f}" if base_auc else "未找到裸 SSF")
    print(f"共 {len(rows)} 行")


if __name__ == "__main__":
    main()
