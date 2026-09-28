#!/usr/bin/env python
"""General NDT Foundation — F 系列: 多源 NDT 缺陷语义预训练 → 焊缝 PAUT 跨试件迁移。

矩阵 (用户原 E0–E5, 分支重命名 F0–F5):
  F0 仅目标 PAUT (单域 vanilla MAE)  = E1 锚点 0.5680, 不重跑
  F1 P + external_weld_ut            — 真实外部 UT 对照
  F2 P + SWRD (焊缝 RT, 2D 图像)     — 同缺陷语义、不同物理模态
  F3 P + extUT + SWRD                — 模态互补
  F4 P + extUT + SWRD + EddyCus      — 完整多源 + 缺陷感知掩码 (语义对齐)
  F5 = F4 数据, 全 random masking    — 归因: 数据量(F5−E1) vs 语义(F4−F5)

协议 (严格, 对齐 E1/E2b 纪律):
- **per-fold 严格预训练**: 每折 test coupon 的 PENELOPE 信号不进预训练
  (PENELOPE 组件恒 3000 步, 只用 4 个非 test coupon); 外部源无标签、与
  PENELOPE 无交集, 整循环只构建一次, 外部步数合计 3000 (与 E2b 同总步数 6000)。
- **模态专用 stem** (E2b 已证正迁移的默认多源架构) + 冻结 logistic 探针。
- 主指标 = 非PP4 逐折均值 ± std (4 折 × 3 seed); 负迁移审计
  Δ≥+0.01 且 ≥2/3 seed 正 = pass; Δ≤−0.01 = stop。

用法:
  python scripts/general_ndt_f_series.py --config configs/general_ndt_f4_full.yaml
                                         [--smoke] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from general_ndt.datasets.registry import build_dataset                    # noqa: E402
from general_ndt.evaluation.probe import logistic_probe                    # noqa: E402
from general_ndt.models.mae import MaskedAutoencoder                       # noqa: E402
from general_ndt.trainers.ssl_trainer import SSLTrainer                    # noqa: E402

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]
NON_PP4 = ["PP3", "PP5", "PP6", "PP7"]
TARGET_DATASET = "penelope_paut"


def make_model(cfg, model_seed):
    torch.manual_seed(model_seed)
    m = cfg["model"]
    return MaskedAutoencoder(
        d_model=int(m["d_model"]), patch_len=int(m["patch_len"]), patch2d=int(m["patch2d"]),
        n_layers_enc=int(m["n_layers_enc"]), n_heads=int(m["n_heads"]),
        d_decoder=int(m["d_decoder"]), n_layers_dec=int(m["n_layers_dec"]),
        mask_ratio=float(m["mask_ratio"]), n_modalities=int(m.get("n_modalities", 9)),
        n_sensors=int(m.get("n_sensors", 32)), dropout=float(m.get("dropout", 0.0)),
        per_modality_stem=bool(m.get("per_modality_stem", True)),
    )


def normalize_policy(policy) -> dict | None:
    """config 的 mask_policy → trainer/model 约定: None/'random' → None;
    {'mode':'region_bias','region_frac':f} → 原样。"""
    if policy in (None, "random", {"mode": "random"}):
        return None
    if isinstance(policy, dict) and policy.get("mode") == "region_bias":
        return policy
    raise ValueError(f"未知 mask_policy: {policy}")


def audit_verdict(delta_per_seed: dict, baseline_mean: float, mean: float) -> str:
    """负迁移审计: Δ≥+0.01 且 ≥2/3 seed 为正 = pass; Δ≤−0.01 = stop; 其余 inconclusive。"""
    delta = mean - baseline_mean
    n_pos = sum(1 for v in delta_per_seed.values() if v > 0)
    if delta >= 0.01 and n_pos >= 2:
        return "pass"
    if delta <= -0.01:
        return "stop"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_name = cfg.get("experiment", args.config.stem)
    sources_cfg = list(cfg["sources"])
    total_steps = int(cfg["ssl"]["n_steps"])

    # 协议断言: PENELOPE 恒为首源、3000 步; 外部合计 = 总步数 - 3000
    assert sources_cfg[0]["dataset"] == TARGET_DATASET, "首源必须是 penelope_paut (per-fold)"
    pen_steps = int(sources_cfg[0]["steps"])
    assert pen_steps == 3000, f"PENELOPE 步数必须为 3000 (E1 同分布), 得到 {pen_steps}"
    assert sum(int(s["steps"]) for s in sources_cfg) == total_steps, "源步数之和 ≠ ssl.n_steps"

    model_seeds = args.seeds or [int(s) for s in cfg["ssl"]["model_seeds"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        # 等比缩到 60 总步 (每源 ≥1), 1 seed, 小样本
        for s in sources_cfg[1:]:
            s["steps"] = max(1, int(s["steps"] * 60 / total_steps))
        sources_cfg[0]["steps"] = max(1, 60 - sum(int(s["steps"]) for s in sources_cfg[1:]))
        total_steps = sum(int(s["steps"]) for s in sources_cfg)
        model_seeds = [0]
        for s in sources_cfg:
            s.setdefault("config", {})["sample_limit"] = 60 if s["dataset"] != TARGET_DATASET else 80

    # 外部源整循环只构建一次 (fold 无关, 无 PENELOPE 泄漏可能)
    external: list[dict] = []
    for s in sources_cfg[1:]:
        samples = build_dataset(s["dataset"], s.get("config", {}))
        if not samples:
            print(f"[{exp_name}] 外部源 '{s['dataset']}' 为空", file=sys.stderr)
            return 2
        kinds = {x.shape_kind for x in samples}
        external.append({
            "name": s["dataset"], "samples": samples,
            "mask_policy": normalize_policy(s.get("mask_policy")),
            "steps": int(s["steps"]), "shape_kinds": sorted(kinds),
        })
        print(f"[{exp_name}] 外部源 '{s['dataset']}': {len(samples)} samples, "
              f"steps={s['steps']}, mask_policy={s.get('mask_policy') or 'random'}, "
              f"shape_kinds={sorted(kinds)}")
    has_2d = any("2d" in e["shape_kinds"] for e in external)

    # PENELOPE 目标域
    pen_cfg = dict(sources_cfg[0].get("config", {}))
    samples = build_dataset(TARGET_DATASET, pen_cfg)
    if not samples:
        print(f"[{exp_name}] 空 PENELOPE", file=sys.stderr)
        return 2
    labels = np.asarray([int(s.label) if s.label is not None else -1 for s in samples])
    print(f"[{exp_name}] PENELOPE {len(samples)} samples (pos_rate={labels.mean():.4f}); "
          f"total_steps={total_steps}, device={device}, seeds={model_seeds}")

    val_frac = float(cfg["probe"].get("val_frac", 0.15))
    data_seed = int(cfg["ssl"]["data_seed"])
    idx_by_coupon = {c: [i for i, s in enumerate(samples) if s.specimen_id == c]
                     for c in COUPONS}
    t0 = time.time()
    all_results = []
    for test_c in NON_PP4:
        te_idx = idx_by_coupon[test_c]
        rest = np.asarray([i for c in COUPONS if c != test_c for i in idx_by_coupon[c]])
        y_rest = labels[rest]
        strat = y_rest if (np.bincount(y_rest, minlength=2) >= 2).all() else None
        tr, va = train_test_split(rest, test_size=val_frac, random_state=data_seed,
                                  shuffle=True, stratify=strat)
        tr_idx, va_idx = sorted(tr.tolist()), sorted(va.tolist())
        pretrain_pen = [samples[i] for i in sorted(set(tr_idx) | set(va_idx))]
        for ms in model_seeds:
            model = make_model(cfg, ms).to(device)
            trainer_cfg = dict(cfg["ssl"])
            trainer_cfg["model_seed"] = ms
            trainer_cfg["data_seed"] = data_seed
            trainer = SSLTrainer(model, trainer_cfg, device=device)
            # 源规格: PENELOPE(per-fold) + 外部源; 按步数配额展开加权轮转表
            source_specs = [{"name": "penelope_paut", "samples": pretrain_pen,
                             "mask_policy": normalize_policy(sources_cfg[0].get("mask_policy"))}]
            source_specs += [{"name": e["name"], "samples": e["samples"],
                              "mask_policy": e["mask_policy"]} for e in external]
            quota = [int(sources_cfg[0]["steps"])] + [e["steps"] for e in external]
            order, rem = [], list(quota)
            while any(r > 0 for r in rem):          # 确定性: 按配额比例轮转展开
                for si, r in enumerate(rem):
                    if r > 0:
                        order.append(si)
                        rem[si] -= 1
            assert len(order) == total_steps
            ckpt = trainer.train_multi(
                source_specs,
                n_steps=total_steps,
                order=order,
                batch_size=int(cfg["ssl"]["batch_size"]),
                log_every=int(cfg["ssl"].get("log_every", 1000)),
                ckpt_every=int(cfg["ssl"].get("ckpt_every", 10**9)),
                output_dir=Path(cfg["output_dir"]) / f"ckpts/{test_c}_seed{ms}",
            )
            eval_samples = [samples[i] for i in tr_idx + va_idx + te_idx]
            feats, ids = trainer.extract_features(eval_samples, batch_size=64)
            feat_by_id = dict(zip(ids, feats))
            F = np.stack([feat_by_id[samples[i].sample_id] for i in tr_idx + te_idx])
            y_sub = np.asarray([labels[i] for i in tr_idx + te_idx])
            tr_pos = list(range(len(tr_idx)))
            te_pos = list(range(len(tr_idx), len(tr_idx) + len(te_idx)))
            r = logistic_probe(F, y_sub, tr_pos, te_pos)
            r["test_coupon"] = test_c
            r["model_seed"] = ms
            r["ckpt"] = str(ckpt)
            all_results.append(r)
            print(f"  seed={ms} test={test_c}: auroc={r['auroc']:.4f} "
                  f"bal_acc={r['balanced_acc']:.4f} f1={r['macro_f1']:.4f} "
                  f"pos_rate={labels[te_idx].mean():.3f}")

    auroc_by_seed = {ms: [r["auroc"] for r in all_results if r["model_seed"] == ms]
                     for ms in model_seeds}
    per_seed_mean = {str(ms): float(np.mean([a for a in v if not np.isnan(a)]))
                     for ms, v in auroc_by_seed.items()}
    all_auroc = [r["auroc"] for r in all_results if not np.isnan(r["auroc"])]
    mean, std = float(np.mean(all_auroc)), float(np.std(all_auroc))

    # 负迁移审计 (vs E1; 若配置了 E1 per-seed 均值则逐 seed 比较)
    baselines = cfg.get("baselines", {})
    audit = {"criterion": "Δ≥+0.01 且 ≥2/3 seed 为正 = pass; Δ≤−0.01 = stop"}
    if "e1_mean" in baselines:
        e1_mean = float(baselines["e1_mean"])
        e1_ps = {str(k): float(v) for k, v in baselines.get("e1_per_seed", {}).items()}
        delta_ps = {ms: per_seed_mean[ms] - e1_ps[ms] for ms in per_seed_mean if ms in e1_ps}
        audit.update({
            "e1_mean": e1_mean, "delta_vs_e1": round(mean - e1_mean, 4),
            "delta_per_seed_vs_e1": {k: round(v, 4) for k, v in delta_ps.items()},
            "seeds_positive_vs_e1": sum(1 for v in delta_ps.values() if v > 0),
            "n_seeds": len(model_seeds),
            "verdict": audit_verdict(delta_ps or {"all": mean - e1_mean}, e1_mean, mean),
        })
    if "e2b_mean" in baselines:
        audit["e2b_mean"] = float(baselines["e2b_mean"])
        audit["delta_vs_e2b"] = round(mean - float(baselines["e2b_mean"]), 4)

    sources_out = [{
        "dataset": s["dataset"], "steps": int(s["steps"]),
        "mask_policy": s.get("mask_policy") or "random",
        "n_samples": len(samples) if s["dataset"] == TARGET_DATASET else
                     len(next(e["samples"] for e in external if e["name"] == s["dataset"])),
    } for s in sources_cfg]
    summary = {
        "experiment": exp_name,
        "config": str(args.config),
        "protocol": "F-series per-fold strict multi-source MAE (per-modality stems) + frozen probe",
        "sources": sources_out,
        "n_steps": total_steps,
        "model_seeds": model_seeds,
        "data_seed": data_seed,
        "per_seed_mean_auroc": per_seed_mean,
        "auroc_mean": mean,
        "auroc_std": std,
        "per_fold_auroc": [r["auroc"] for r in all_results],
        "audit": audit,
        "results": all_results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"\n[{exp_name}] per-seed 非PP4 逐折均值: { {k: round(v, 4) for k, v in per_seed_mean.items()} }")
    print(f"[{exp_name}] AUROC = {mean:.4f} ± {std:.4f} (非PP4 逐折均值, {len(all_auroc)} 折×seed)")
    if audit.get("delta_vs_e1") is not None:
        print(f"[{exp_name}] 审计 vs E1: Δ={audit['delta_vs_e1']:+.4f} → verdict={audit.get('verdict')}")
    out = args.out or (Path(cfg["output_dir"]) / f"{exp_name}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{exp_name}] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
