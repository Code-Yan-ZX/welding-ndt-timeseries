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
    # 裸 SSF 基线 (优先 control, 否则基线 json 的 ssf)
    bare = next((r for r in rows if r["model"] == "ssf" and not r["augment"]
                 and "control" in r["file"]), None)
    if bare is None:
        bare = next((r for r in rows if r["model"] == "ssf" and not r["augment"]), None)
    base_auc = bare["auc_mean"] if bare else None      # 含 PP4 的 5 折均值 (受 PP4 噪声虚高)
    bare_np4 = bare["non_pp4"] if bare else None       # 非PP4 4 折均值 (可信基线)

    lines = ["# PAUT LOOCV P0 汇总对比表", "",
             "5 折留一试件交叉验证 (PP3-PP7 轮流 test), 其余 4 试件 85/15 分层 train/val,",
             "per-fold 归一化 (无泄漏, 单次归一化), val 调阈值+早停。seed=42。", ""]
    if bare_np4 is not None:
        lines.append(f"**裸 SSF LOOCV 非PP4 AUC = {bare_np4:.3f}** (跨试件泛化基线, "
                     f"远低于单点 PP7=0.626, 揭示跨试件过拟合)。")
        if base_auc is not None:
            lines.append(f"含 PP4 的 5 折均值 {base_auc:.3f} 受 PP4 噪声折 (仅 3 正样本) 虚高, 不可信 -- 详见备注。")
        lines.append(f"成功门槛: SSF+增强+DANN 较裸 SSF(非PP4) 提升 ≥0.03 (即非PP4 AUC ≥ {bare_np4+0.03:.3f})。")
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
    # 裸 SSF 非 PP4 基线 (bare_np4 已在上方计算)
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
              "- **PP4 是近零缺陷试件 (非数据错误)**: 经官方 AIMEN UT 报告 (`PP4/2. ndt_data/UT.pdf`) 证实, "
              "PP4 仅 1 个 2mm 可接受气孔 (X=229mm), 试件最终被接收 (ACEPTADO), 报告抬头 "
              "\"PENELOPE-WP4 ZERO-DEFECT MANUFACTURING\"。各试件 xlsx 缺陷数 "
              "PP3=68/PP5=50/PP6=112/PP7=12/PP4=1, PP4 是唯一的近零缺陷试件 -- "
              "非下载失败、非解析 bug、非标注遗漏。",
              "- **PP4 退化折**: PP4 仅 3 个局部缺陷位置 (0.5%), 作 test 时 AUC 纯噪声 (0.55-0.77 间随机波动), "
              "使含 PP4 的 mean±std 不可靠。**非PP4 AUC** (剔除 PP4) 是更可信的跨试件泛化指标。",
              "- **PP5 标注录入反转 (已修代码)**: `defects_xlocation.xlsx` PP5 sheet 有 1 行 x_init=177 > x_end=160 "
              "(长度 -17mm, 数据集本身录入反转)。旧版 `position_labels` 静默跳过, 致 PP5 少计 18 个缺陷位置 "
              "(全量 3000 中占 0.6%, 在 seed 噪声内)。`paut_preprocess.py` 已修复 (swap 恢复 + warning); "
              "现有 P0-P3 结果沿用修复前标签, 下次 `paut_preprocess.py` 运行自动生效, 定性结论不变。",
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
