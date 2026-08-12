#!/usr/bin/env python
"""PAUT 物理条件化 LoRA 5 折 LOOCV (P3 组件 C)。

在 P2 视觉端 LoRA(LLM 冻结, 仅微调 model.visual.blocks.*.attn.qkv/proj, r=16)基础上:
  - 用 physics 条件化 prompt(组件 A)
  - 完整 5 折 LOOCV: PP3-PP7 轮流 test, 其余 4 试件训练(采样至 max_train 控成本)
  - 每折报告 AUC, 汇总 per-fold + 非PP4 池化

打分用单 token yes/no logprob 差(与 P2 LoRA 同口径)。物理参数从 meta_summary.json 动态读。

硬约束同 paut_vlm_physics_zeroshot.py: .venv_p2, transformers 直推, CUDA_VISIBLE_DEVICES=1,2。

Usage:
  CUDA_VISIBLE_DEVICES=1,2 .venv_p2/bin/python scripts/paut_vlm_physics_lora.py \
      --max-train 400 --epochs 1 --tag full
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "data/processed/paut/images"
MODEL = "models/Qwen3.6-27B"
PAUT = REPO / "data/processed/paut"

# 复用 zeroshot 脚本的 prompt 构建(单一来源)
from paut_vlm_physics_zeroshot import build_prompt, PHYSICS_PREAMBLE  # noqa: F401

COUPONS = ["PP3", "PP4", "PP5", "PP6", "PP7"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=400, help="每折训练位置上限")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-type", default="bscan")
    ap.add_argument("--mode", default="physics", choices=["bare", "physics"],
                    help="LoRA 训练/评测用哪种 prompt(单 token 打分)")
    ap.add_argument("--folds", default="all", help="all 或 PP7,PP6 逗号分隔子集")
    ap.add_argument("--tag", default="run")
    return ap.parse_args()


def main():
    args = parse_args()
    labels = np.load(PAUT / "meta_label.npy")
    coupons = np.load(PAUT / "meta_coupon.npy")
    with open(PAUT / "meta_summary.json") as fh:
        pc = json.load(fh)["per_coupon"]
    folds = COUPONS if args.folds == "all" else [c.strip() for c in args.folds.split(",")]

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    print("加载 processor + 模型 ...")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tok = proc.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    dev = next(base.parameters()).device
    yes_ids, no_ids = set(), set()
    for w in [" yes", "Yes", "yes"]:
        yes_ids.update(tok.encode(w, add_special_tokens=False))
    for w in [" no", "No", "no"]:
        no_ids.update(tok.encode(w, add_special_tokens=False))
    yes_ids, no_ids = yes_ids - no_ids, no_ids - yes_ids

    # LoRA 配置(与 P2 同: 仅视觉端 attn)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv", "attn.proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    # 单次 wrap; 每折重置 LoRA 权重(A=kaiming, B=0)避免跨折泄漏, 不重新加载 52GB 基座
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    def prompt_for(i):
        c = str(coupons[i])
        return build_prompt(args.mode, c, float(pc[c]["offset_mm"]),
                            int(pc[c]["n_beams"]), float(pc[c]["res_mm"]))

    def reinit_lora():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B" in n:
                torch.nn.init.zeros_(p)

    @torch.no_grad()
    def score_batch(imgs, prompts):
        # 用 peft model(LoRA active), 非 base
        texts = [proc.apply_chat_template([{"role": "user", "content": [
            {"type": "image", "image": im}, {"type": "text", "text": p}]}],
            tokenize=False, add_generation_prompt=True) for p, im in zip(prompts, imgs)]
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        out = model(**inputs)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        scs = []
        for i in range(len(imgs)):
            ly = max((lp[i, t].item() for t in yes_ids), default=-1e9)
            ln = max((lp[i, t].item() for t in no_ids), default=-1e9)
            scs.append(float(ly - ln))
        return scs

    from sklearn.metrics import roc_auc_score
    fold_results = []
    for test_c in folds:
        print(f"\n=== 折 test={test_c} ===")
        train_idx = np.nonzero(coupons != test_c)[0]
        test_idx = np.nonzero(coupons == test_c)[0]
        rng = np.random.default_rng(42)
        if len(train_idx) > args.max_train:
            train_idx = rng.choice(train_idx, args.max_train, replace=False)
        print(f"训练 {len(train_idx)} | 测试 {len(test_idx)} ({test_c})")

        # 每折重置 LoRA 权重 + 新优化器(无跨折泄漏)
        reinit_lora()
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        t0 = time.time()
        for ep in range(args.epochs):
            order = rng.permutation(train_idx)
            for ii, idx in enumerate(order):
                img = Image.open(IMG_DIR / f"{int(idx):05d}_{args.image_type}.png").convert("RGB")
                pr = prompt_for(int(idx))
                ans = " yes" if labels[idx] == 1 else " no"
                msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": pr}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + ans
                inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                labels_in = inputs["input_ids"].clone()
                ans_ids = tok.encode(ans, add_special_tokens=False)
                labels_in[:, :-len(ans_ids)] = -100
                out = model(**inputs, labels=labels_in)
                loss = out.loss
                opt.zero_grad(); loss.backward(); opt.step()
                if (ii + 1) % 50 == 0:
                    print(f"  ep{ep} {ii+1}/{len(order)} loss {loss.item():.4f} ({(time.time()-t0)/60:.1f}min)")
        # 评测
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(test_idx), 8):
                batch = test_idx[i:i + 8]
                imgs = [Image.open(IMG_DIR / f"{int(j):05d}_{args.image_type}.png").convert("RGB") for j in batch]
                prompts = [prompt_for(int(j)) for j in batch]
                scores.extend(score_batch(imgs, prompts))
        y = labels[test_idx]
        auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")
        print(f">> LoRA {test_c} AUC = {auc:.4f} (n={len(y)})")
        fold_results.append({"test_coupon": test_c, "auc": auc, "n_test": int(len(y)),
                             "n_train": int(len(train_idx)), "scores": [float(s) for s in scores]})

    # 汇总
    nonpp4 = [f for f in fold_results if f["test_coupon"] != "PP4"]
    # 正确指标:逐折 AUC 均值(每折是不同 LoRA 模型, 分数尺度不可比, 跨折池化无效)
    mean_auc = float(np.nanmean([f["auc"] for f in fold_results]))
    auc_np4 = float(np.nanmean([f["auc"] for f in nonpp4]))
    # 池化(仅参考, 因每折模型不同而不可比)
    ys, ss = [], []
    for f in nonpp4:
        ti = np.nonzero(coupons == f["test_coupon"])[0]
        ys.extend(labels[ti].tolist()); ss.extend(f["scores"])
    auc_np4_pooled = float(roc_auc_score(ys, ss)) if len(np.unique(ys)) == 2 else float("nan")
    summary = {"mode": args.mode, "tag": args.tag, "max_train": args.max_train,
               "epochs": args.epochs, "per_fold": {f["test_coupon"]: round(f["auc"], 4) for f in fold_results},
               "mean_auc": mean_auc, "auc_nonpp4": auc_np4,
               "auc_nonpp4_pooled_invalid": auc_np4_pooled}
    print(f"\n=== LoRA 汇总 ({args.mode}) ===\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    out = REPO / f"experiments/results/paut_vlm_physics_lora_{args.tag}.json"
    with open(out, "w") as fh:
        json.dump({"summary": summary, "folds": fold_results}, fh, indent=2, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
