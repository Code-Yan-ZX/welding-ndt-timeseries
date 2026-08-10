#!/usr/bin/env python
"""PAUT 多模态 LLM 零样本缺陷检测 (P2-②) -- transformers 直推版。

用 transformers 4.57 加载 Qwen3.6-27B (Qwen3.5-VL), device_map=auto 跨 2 GPU。对每位置
B-scan 图(灰度 + VLT 式频谱伪彩色)零样本 QA: 问"是否有焊缝缺陷", 取 prompt 后首 token
的 yes/no logprob 差作连续分数。AUC vs 缺陷标签。

注: vLLM 0.26 需 CUDA13, 但本机 driver 535 仅支持 CUDA12.2, 故回退 transformers 直推
(目标允许自主决断方法)。推理较慢, 默认子采样 600 (分层)。

Usage:
  .venv/bin/python scripts/paut_vlm_zeroshot.py --max-samples 600 --image-type both
  .venv/bin/python scripts/paut_vlm_zeroshot.py --max-samples 0  # 全部 3000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "data/processed/paut/images"
MODEL = "models/Qwen3.6-27B"
PROMPT = ("You are an expert ultrasonic weld NDT inspector. This is a PAUT B-scan image "
          "(beam axis x depth/time axis). Is there a weld defect indication in this B-scan? "
          "Answer with a single word: yes or no.\nThe answer is")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-type", default="both", choices=["bscan", "spec", "both"])
    ap.add_argument("--max-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    labels = np.load(REPO / "data/processed/paut/meta_label.npy")
    coupons = np.load(REPO / "data/processed/paut/meta_coupon.npy")
    N = len(labels)
    idxs = list(range(N))
    if args.max_samples and args.max_samples < N:
        rng = np.random.default_rng(args.seed)
        idxs_arr = np.array(idxs)
        strata = np.array([str(coupons[i]) + str(labels[i]) for i in idxs_arr])
        sel = []
        for s in np.unique(strata):
            s_idx = idxs_arr[strata == s]
            n = max(1, int(round(len(s_idx) * args.max_samples / N)))
            sel.append(rng.choice(s_idx, min(n, len(s_idx)), replace=False))
        idxs = np.concatenate(sel).tolist()
        print(f"子采样 {len(idxs)}/{N} (分层)")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    t0 = time.time()
    print("加载 processor + 模型 (bf16, device_map=auto) ...")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device
    tok = proc.tokenizer
    yes_ids, no_ids = set(), set()
    for w in [" yes", "Yes", "yes"]:
        yes_ids.update(tok.encode(w, add_special_tokens=False))
    for w in [" no", "No", "no"]:
        no_ids.update(tok.encode(w, add_special_tokens=False))
    print(f"模型加载 {time.time()-t0:.0f}s | yes_ids={yes_ids} no_ids={no_ids}")

    @torch.no_grad()
    def score_image(img: Image.Image) -> float:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        out = model(**inputs)
        logits = out.logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)[0]
        lp_y = max((lp[t].item() for t in yes_ids), default=-1e9)
        lp_n = max((lp[t].item() for t in no_ids), default=-1e9)
        return float(lp_y - lp_n)

    img_types = ["bscan", "spec"] if args.image_type == "both" else [args.image_type]
    results = {t: [] for t in img_types}
    from sklearn.metrics import roc_auc_score
    for t in img_types:
        print(f"\n=== 零样本打分: {t} ({len(idxs)} 张) ===")
        ts = time.time()
        for ii, idx in enumerate(idxs):
            img = Image.open(IMG_DIR / f"{idx:05d}_{t}.png").convert("RGB")
            sc = score_image(img)
            results[t].append({"idx": int(idx), "coupon": str(coupons[idx]),
                               "label": int(labels[idx]), "score": sc})
            if (ii + 1) % 50 == 0:
                print(f"  {ii+1}/{len(idxs)} ({(time.time()-ts)/60:.1f}min)")
        y = np.array([r["label"] for r in results[t]])
        s = np.array([r["score"] for r in results[t]])
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, s))
            cps = np.array([r["coupon"] for r in results[t]])
            nonpp4 = cps != "PP4"
            auc_np4 = float(roc_auc_score(y[nonpp4], s[nonpp4])) if len(np.unique(y[nonpp4])) == 2 else float("nan")
            print(f">> {t} 零样本 AUC = {auc:.4f} (非PP4 {auc_np4:.4f})")
            results[t + "_auc"] = {"auc": auc, "auc_nonpp4": auc_np4, "n": len(y)}

    out = REPO / "experiments/results/paut_vlm_zeroshot.json"
    save = {k: v for k, v in results.items() if not k.endswith("_auc")}
    with open(out, "w") as fh:
        json.dump(save, fh, indent=2)
    summary = {k: v for k, v in results.items() if k.endswith("_auc")}
    summary["max_samples"] = args.max_samples
    with open(REPO / "experiments/results/paut_vlm_zeroshot_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n=== 零样本汇总 ===\n{json.dumps(summary, indent=2)}")
    best = max((s.get("auc_nonpp4", 0) for s in summary.values() if isinstance(s, dict)), default=0)
    print(f"最佳非PP4 AUC={best:.4f} {'>0.55 触发微调' if best > 0.55 else '<=0.55 不微调'}")


if __name__ == "__main__":
    main()
