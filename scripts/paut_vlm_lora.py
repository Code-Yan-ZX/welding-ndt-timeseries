#!/usr/bin/env python
"""PAUT 多模态 LLM LoRA 微调 (P2-③) -- 仅当零样本 AUC>0.55 时运行。

冻结 LLM 主干, LoRA 微调视觉端 (Qwen3.5-VL 的视觉编码器 + adapter), 在 1798 训练位置
(LOOCV 单折: 4 试件训练, 1 试件测试) 上微调。受限于 27B 模型 + 慢路径 (无 vLLM/causal-conv1d),
本脚本做精简单折微调验证思路 (非完整 5 折 LOOCV)。

Usage: .venv_p2/bin/python scripts/paut_vlm_lora.py --test-coupon PP7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "data/processed/paut/images"
MODEL = "models/Qwen3.6-27B"
PROMPT = ("You are an expert ultrasonic weld NDT inspector. This is a PAUT B-scan image. "
          "Is there a weld defect indication? Answer with a single word: yes or no.\nThe answer is")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-coupon", default="PP7")
    ap.add_argument("--image-type", default="bscan")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=400, help="训练位置上限 (时间约束)")
    return ap.parse_args()


def main():
    args = parse_args()
    labels = np.load(REPO / "data/processed/paut/meta_label.npy")
    coupons = np.load(REPO / "data/processed/paut/meta_coupon.npy")
    train_idx = np.nonzero(coupons != args.test_coupon)[0]
    test_idx = np.nonzero(coupons == args.test_coupon)[0]
    rng = np.random.default_rng(42)
    if len(train_idx) > args.max_train:
        train_idx = rng.choice(train_idx, args.max_train, replace=False)
    print(f"训练 {len(train_idx)} 位置 (4试件) | 测试 {len(test_idx)} ({args.test_coupon})")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    # LoRA: 仅微调视觉端注意力 (model.visual.blocks.*.attn.qkv / attn.proj), LLM 主干完全冻结
    # (LLM 用 q_proj/v_proj, 不含 qkv; "attn.proj" 仅匹配视觉, 不匹配 LLM 的 o_proj)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv", "attn.proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()
    dev = next(model.parameters()).device
    tok = proc.tokenizer
    yes_ids = set(); no_ids = set()
    for w in [" yes","Yes","yes"]: yes_ids.update(tok.encode(w, add_special_tokens=False))
    for w in [" no","No","no"]: no_ids.update(tok.encode(w, add_special_tokens=False))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 训练
    for ep in range(args.epochs):
        for ii, idx in enumerate(train_idx):
            img = Image.open(IMG_DIR / f"{idx:05d}_{args.image_type}.png").convert("RGB")
            ans = " yes" if labels[idx] == 1 else " no"
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + ans
            inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            labels_in = inputs["input_ids"].clone()
            # 只对答案 token 算 loss (简化: 对最后答案 token)
            ans_ids = tok.encode(ans, add_special_tokens=False)
            labels_in[:, :-len(ans_ids)] = -100
            out = model(**inputs, labels=labels_in)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            if (ii+1) % 50 == 0:
                print(f"  ep{ep} {ii+1}/{len(train_idx)} loss {loss.item():.4f}")

    # 评估
    model.eval()
    scores = []
    with torch.no_grad():
        for idx in test_idx:
            img = Image.open(IMG_DIR / f"{idx:05d}_{args.image_type}.png").convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            out = model(**inputs)
            lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0]
            lp_y = max((lp[t].item() for t in yes_ids), default=-1e9)
            lp_n = max((lp[t].item() for t in no_ids), default=-1e9)
            scores.append(float(lp_y - lp_n))
    from sklearn.metrics import roc_auc_score
    y = labels[test_idx]
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")
    print(f"\nLoRA 微调后 {args.test_coupon} AUC = {auc:.4f} (n={len(y)})")
    out = {"test_coupon": args.test_coupon, "auc": auc, "n_train": len(train_idx),
           "n_test": len(test_idx), "image_type": args.image_type}
    json.dump(out, open(REPO / "experiments/results/paut_vlm_lora.json", "w"), indent=2)


if __name__ == "__main__":
    main()
