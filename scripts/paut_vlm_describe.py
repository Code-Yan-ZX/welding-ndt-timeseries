#!/usr/bin/env python
"""PAUT 多模态 LLM 可解释性文本输出 (P2-④)。

让 Qwen3.6-27B 描述若干 B-scan 图 (缺陷+干净各若干), 记录 VLM 的文本输出, 验证其
能否识别超声回波结构/缺陷指示。用于可解释性对照。

Usage: .venv_p2/bin/python scripts/paut_vlm_describe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "data/processed/paut/images"
MODEL = "models/Qwen3.6-27B"
DESCRIBE_PROMPT = ("You are an expert ultrasonic weld NDT inspector. Describe this PAUT B-scan "
                   "image (beam axis x depth/time axis). What indications do you see? Is there a "
                   "potential weld defect? Be concise (3-4 sentences).")


def main():
    import numpy as np
    labels = np.load(REPO / "data/processed/paut/meta_label.npy")
    coupons = np.load(REPO / "data/processed/paut/meta_coupon.npy")
    # 选 5 缺陷 + 5 干净 (跨试件)
    rng = np.random.default_rng(42)
    defect_idx = rng.choice(np.nonzero(labels == 1)[0], 5, replace=False)
    clean_idx = rng.choice(np.nonzero(labels == 0)[0], 5, replace=False)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device

    outputs = []
    for img_type in ["bscan", "spec"]:
        for grp, idxs in [("defect", defect_idx), ("clean", clean_idx)]:
            for idx in idxs:
                img = Image.open(IMG_DIR / f"{idx:05d}_{img_type}.png").convert("RGB")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": DESCRIBE_PROMPT}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
                resp = proc.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip()
                outputs.append({"img_type": img_type, "group": grp, "idx": int(idx),
                                "coupon": str(coupons[idx]), "label": int(labels[idx]),
                                "response": resp})
                print(f"[{img_type}/{grp} idx={idx} {coupons[idx]} lab={labels[idx]}] {resp[:150]}")

    out = REPO / "experiments/results/paut_vlm_describe.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2, ensure_ascii=False)
    print(f"\n-> {out} ({len(outputs)} 条描述)")


if __name__ == "__main__":
    main()
