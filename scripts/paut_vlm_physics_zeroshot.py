#!/usr/bin/env python
"""PAUT 物理条件化 + 推理式 CoT 多模态缺陷检测 (P3)。

在 P2 的 Qwen3.6-27B + transformers 直推框架上,把「裸 B-scan + yes/no 单 token 打分」
升级为四模式消融:
  - bare        : P2 原 prompt(校准,应≈非PP4 AUC 0.593)
  - physics     : +物理条件化 prompt(组件 A),单 token 打分
  - physics_cot : +物理 + 5 步差分诊断 CoT(组件 B1),生成推理后取答案 token logprob 差
  - physics_json: +物理 + 结构化 JSON{reasoning,defect}(组件 B2),同法打分

物理参数(探头角度/声速为材料物理常数;offset/波束/分辨率/缺陷码从 meta_summary.json
动态读取,不硬编码)。5 折 LOOCV 对零样本=按 coupon 切片算 per-coupon AUC + 非PP4 池化。

打分:
  单 token 模式(bare/physics):批量 left-pad forward,取 logits[:,-1] 的 yes/no logprob 差。
  生成模式(cot/json):model.generate(output_scores=True, return_dict_in_generate=True),
    在生成的 token 序列中定位 yes/no token 所在 step,取该 step logits 算 logprob 差--
    忠实反映「给定模型自己生成的推理后的答案置信度」,无需额外 forward。

硬约束:本机 driver 535 / CUDA 12.2,禁 vLLM(需 CUDA13);用 .venv_p2(transformers 5.14
+ torch cu126 + flash-linear-attention)直推。模型跨 GPU:建议 CUDA_VISIBLE_DEVICES=1,2。

Usage:
  # smoke(16 张分层,校准速度与 prompt)
  CUDA_VISIBLE_DEVICES=1,2 .venv_p2/bin/python scripts/paut_vlm_physics_zeroshot.py \
      --mode physics_cot --max-samples 16 --batch-size 4 --tag smoke
  # 全量四模式(单进程依次跑,省模型重载)
  CUDA_VISIBLE_DEVICES=1,2 .venv_p2/bin/python scripts/paut_vlm_physics_zeroshot.py \
      --mode all --max-samples 0 --batch-size 8 --tag full
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "data/processed/paut/images"
MODEL = "models/Qwen3.6-27B"
PAUT = REPO / "data/processed/paut"

# ---- 物理常数(探头/材料物理,非数据派生;文档化常量)----
PROBE_ANGLE_DEG = 71.0          # G0 角探头的钢中折射角(剪波)
SHEAR_VEL_MS = 3230.0           # 钢剪波速 m/s
# cos(71°) 用于声程->深度换算:z = s·cos(71°)
import math
DEPTH_COS = math.cos(math.radians(PROBE_ANGLE_DEG))

# ---- P2 原 prompt(校准用,逐字保留)----
P2_PROMPT = ("You are an expert ultrasonic weld NDT inspector. This is a PAUT B-scan image "
             "(beam axis x depth/time axis). Is there a weld defect indication in this B-scan? "
             "Answer with a single word: yes or no.\nThe answer is")

# ---- 物理条件化前文(组件 A)----
PHYSICS_PREAMBLE = (
    "You are an expert ultrasonic weld inspector analyzing a phased-array ultrasonic testing (PAUT) B-scan.\n"
    "PHYSICAL SETUP (use this to interpret the image, do not ignore):\n"
    f"- Angle-beam probe: G0, nominal refract angle {PROBE_ANGLE_DEG:.0f}° (shear wave in steel).\n"
    f"- Steel shear-wave velocity: c_s ≈ {SHEAR_VEL_MS:.0f} m/s.\n"
    "- Acquisition: {n_beams} focal beams, {res_mm:.1f} mm beam spacing; scan offset {offset_mm:.0f} mm from the weld centerline.\n"
    "- B-scan axes: horizontal = scan position (mm); vertical = sound-path / time-of-flight (depth increases downward). "
    f"Physical depth from sound-path s: z = s·cos({PROBE_ANGLE_DEG:.0f}°) ≈ s·{DEPTH_COS:.3f}.\n"
    "- Intensity = echo reflectivity, percentile-normalized (2–98%).\n"
    "- Coupon ID: {coupon}. Candidate defect types: 1 porosity, 2 lack-of-fusion, 3 slag inclusion, "
    "4 metallic inclusion, 5 projection, 6 crack.\n"
    "INSPECTION RULE: a TRUE defect echo is a localized, coherent indication that does NOT coincide "
    "with the expected back-wall or corner-reflection geometry. Distinguish defect echoes from "
    "geometric echoes (back-wall, corner, mode-conversion), which are normal and not defects."
)

# ---- CoT 差分诊断脚手架(组件 B1)----
COT_SUFFIX = (
    "\nReason step by step (keep each step to one sentence), then end with exactly: Answer: yes  or  Answer: no\n"
    "1. Identify the strongest echoes and estimate their sound-path depth.\n"
    f"2. Convert to physical depth z = s·cos({PROBE_ANGLE_DEG:.0f}°); note whether each strong echo "
    "sits on the expected back-wall/corner line (geometric) or off it (suspect).\n"
    "3. Classify morphology: point-like (porosity), linear (lack-of-fusion/crack), extended (slag).\n"
    "4. Decide if a true defect indication is present (differential diagnosis vs geometric echo).\n"
)

# ---- 结构化 JSON(组件 B2)----
JSON_SUFFIX = (
    "\nOutput a JSON object with this exact schema (keep reasoning to ~4 sentences, nothing else):\n"
    '{"reasoning": "<step-by-step differential diagnosis, citing sound-path/depth and whether the '
    'echo is geometric or a suspect defect>", "defect": "yes" or "no"}'
)

MODES = ["bare", "physics", "physics_cot", "physics_json"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="physics",
                    help="bare/physics/physics_cot/physics_json,或逗号分隔(如 bare,physics),或 all")
    ap.add_argument("--image-type", default="bscan", choices=["bscan", "spec"])
    ap.add_argument("--max-samples", type=int, default=0, help="0=全部 3000")
    ap.add_argument("--batch-size", type=int, default=8, help="单 token 模式 batch")
    ap.add_argument("--gen-batch-size", type=int, default=4, help="生成模式 batch(显存更紧)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0=按模式自动(cot 200/json 240)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run", help="结果文件后缀")
    return ap.parse_args()


def build_prompt(mode: str, coupon: str, offset_mm: float, n_beams: int, res_mm: float) -> str:
    if mode == "bare":
        return P2_PROMPT
    pre = PHYSICS_PREAMBLE.format(n_beams=n_beams, res_mm=res_mm, offset_mm=offset_mm, coupon=coupon)
    if mode == "physics":
        return (pre + "\n\nIs there a weld defect indication in this B-scan? "
                "Answer with a single word: yes or no.\nThe answer is")
    if mode == "physics_cot":
        return pre + COT_SUFFIX
    if mode == "physics_json":
        return pre + JSON_SUFFIX
    raise ValueError(mode)


def stratified_subset(labels, coupons, n, seed):
    N = len(labels)
    if not n or n >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    strata = np.array([f"{coupons[i]}_{labels[i]}" for i in range(N)])
    sel = []
    for s in np.unique(strata):
        s_idx = np.nonzero(strata == s)[0]
        k = max(1, int(round(len(s_idx) * n / N)))
        sel.append(rng.choice(s_idx, min(k, len(s_idx)), replace=False))
    return np.concatenate(sel)


def main():
    args = parse_args()
    labels = np.load(PAUT / "meta_label.npy")
    coupons = np.load(PAUT / "meta_coupon.npy")
    with open(PAUT / "meta_summary.json") as fh:
        meta = json.load(fh)
    pc = meta["per_coupon"]
    N = len(labels)
    idxs = stratified_subset(labels, coupons, args.max_samples, args.seed)
    print(f"评估 {len(idxs)}/{N} 位置 | image={args.image_type} | batch={args.batch_size} | tag={args.tag}")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    t0 = time.time()
    print("加载 processor + 模型 (bf16, device_map=auto) ...")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tok = proc.tokenizer
    # left padding: 批量生成与单 token 打分都取 logits[:,-1]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # device_map="auto" 跨双卡(与 P2 同;曾试 max_memory 限制导致视觉编码器/LLM 分卡,
    # 每 forward 跨卡 P2P(旧驱动有问题)死锁,故不用 max_memory,改靠 batch 控显存)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    dev = next(model.parameters()).device
    # yes/no 候选 token(含前导空格/大写/小写)
    yes_ids, no_ids = set(), set()
    for w in [" yes", "Yes", "yes", '"yes"', ' "yes"']:
        yes_ids.update(tok.encode(w, add_special_tokens=False))
    for w in [" no", "No", "no", '"no"', ' "no"']:
        no_ids.update(tok.encode(w, add_special_tokens=False))
    yes_ids, no_ids = yes_ids - no_ids, no_ids - yes_ids
    print(f"模型加载 {time.time()-t0:.0f}s | yes_ids={sorted(yes_ids)} no_ids={sorted(no_ids)}")

    def msgs_for(prompt, img):
        return [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]

    def chat_text(prompt, img, thinking=False):
        """生成 prompt 文本。thinking=False 关闭 Qwen3 思考模式(空 <think> 块),用于 CoT 生成;
        单 token 模式用默认(thinking 不强制),与 P2 同口径。"""
        kw = {"tokenize": False, "add_generation_prompt": True}
        if not thinking:
            kw["enable_thinking"] = False
        return proc.apply_chat_template(msgs_for(prompt, img), **kw)

    @torch.no_grad()
    def _last_yesno_lp(logits):
        """logits (B, V) -> list[B] of (lp_yes - lp_no), 取 yes/no 候选 token 的最大 logprob 差。"""
        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        out = []
        for i in range(logits.shape[0]):
            ly = max((lp[i, t].item() for t in yes_ids), default=-1e9)
            ln = max((lp[i, t].item() for t in no_ids), default=-1e9)
            out.append(float(ly - ln))
        return out

    @torch.no_grad()
    def score_single(imgs, prompts):
        """单 token 模式:批量 forward,取每序列末位 yes/no logprob 差(与 P2 同口径)。"""
        texts = [chat_text(p, im, thinking=True) for p, im in zip(prompts, imgs)]
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        out = model(**inputs)
        return _last_yesno_lp(out.logits)

    @torch.no_grad()
    def score_gen(imgs, prompts, max_new):
        """生成模式:generate 推理(enable_thinking=False),再重打分--取 [prompt+推理+' Answer:']
        forward 末位 yes/no logprob 差。无论模型是否说到答案都给连续分数,稳健。"""
        prompt_texts = [chat_text(p, im, thinking=False) for p, im in zip(prompts, imgs)]
        inputs = proc(text=prompt_texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        plen = inputs["input_ids"].shape[1]
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new_ids = gen[:, plen:]                       # (B, T_new)
        resps = [tok.decode(new_ids[i], skip_special_tokens=True) for i in range(len(imgs))]
        # 重打分:prompt(含图)+ 推理(截断到 Answer: 之前)+ ' Answer:'
        rescore_texts = []
        for p, im, resp in zip(prompts, imgs, resps):
            reasoning = resp
            for marker in ["Answer:", "answer:"]:
                j = resp.rfind(marker)
                if j >= 0:
                    reasoning = resp[:j]
                    break
            rescore_texts.append(chat_text(p, im, thinking=False) + reasoning + " Answer:")
        rinp = proc(text=rescore_texts, images=imgs, return_tensors="pt", padding=True)
        rinp = {k: v.to(dev) for k, v in rinp.items()}
        rout = model(**rinp)
        scs = _last_yesno_lp(rout.logits)
        results = []
        for sc, resp in zip(scs, resps):
            low = resp.lower()
            ans = None
            for m in ["answer:", "answer :"]:
                j = low.rfind(m)
                if j >= 0:
                    tail = low[j + len(m):].strip()
                    if tail.startswith("yes"):
                        ans = "yes"
                    elif tail.startswith("no"):
                        ans = "no"
                    break
            results.append((sc, ans, resp))
        del gen, new_ids, rout, rinp
        return results

    modes = MODES if args.mode == "all" else [m.strip() for m in args.mode.split(",") if m.strip()]
    all_summary = {}
    from sklearn.metrics import roc_auc_score

    for mode in modes:
        max_new = args.max_new_tokens or (200 if mode == "physics_cot" else 240)
        is_gen = mode in ("physics_cot", "physics_json")
        bs = args.gen_batch_size if is_gen else args.batch_size
        print(f"\n=== 模式 {mode} (gen={is_gen}, max_new={max_new}, batch={bs}) ===")
        results = []
        t0 = time.time()
        for b0 in range(0, len(idxs), bs):
            batch = idxs[b0:b0 + bs]
            imgs = [Image.open(IMG_DIR / f"{int(i):05d}_{args.image_type}.png").convert("RGB") for i in batch]
            prompts = [build_prompt(mode, str(coupons[i]), float(pc[str(coupons[i])]["offset_mm"]),
                                    int(pc[str(coupons[i])]["n_beams"]), float(pc[str(coupons[i])]["res_mm"]))
                       for i in batch]
            try:
                if is_gen:
                    outs = score_gen(imgs, prompts, max_new)
                    for i, (sc, ans, resp) in zip(batch, outs):
                        results.append({"idx": int(i), "coupon": str(coupons[i]),
                                        "label": int(labels[i]), "score": sc,
                                        "ans": ans, "resp": resp[:800]})
                else:
                    scs = score_single(imgs, prompts)
                    for i, sc in zip(batch, scs):
                        results.append({"idx": int(i), "coupon": str(coupons[i]),
                                        "label": int(labels[i]), "score": sc})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  ! OOM batch@{b0}, 退化为单张")
                for i in batch:
                    img = Image.open(IMG_DIR / f"{int(i):05d}_{args.image_type}.png").convert("RGB")
                    pr = build_prompt(mode, str(coupons[i]), float(pc[str(coupons[i])]["offset_mm"]),
                                      int(pc[str(coupons[i])]["n_beams"]), float(pc[str(coupons[i])]["res_mm"]))
                    if is_gen:
                        (sc, ans, resp) = score_gen([img], [pr], max_new)[0]
                        results.append({"idx": int(i), "coupon": str(coupons[i]), "label": int(labels[i]),
                                        "score": sc, "ans": ans, "resp": resp[:800]})
                    else:
                        sc = score_single([img], [pr])[0]
                        results.append({"idx": int(i), "coupon": str(coupons[i]), "label": int(labels[i]), "score": sc})
            if (b0 // bs + 1) % 10 == 0 or b0 + bs >= len(idxs):
                el = time.time() - t0
                done = min(b0 + bs, len(idxs))
                print(f"  {done}/{len(idxs)} ({el/60:.1f}min, {el/done:.2f}s/张)")

        # AUC
        y = np.array([r["label"] for r in results])
        s = np.array([r["score"] for r in results])
        cps = np.array([r["coupon"] for r in results])
        per_coupon = {}
        for c in ["PP3", "PP4", "PP5", "PP6", "PP7"]:
            m = cps == c
            if len(np.unique(y[m])) == 2:
                per_coupon[c] = float(roc_auc_score(y[m], s[m]))
            else:
                per_coupon[c] = None
        nonpp4 = cps != "PP4"
        auc_all = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
        auc_np4 = float(roc_auc_score(y[nonpp4], s[nonpp4])) if len(np.unique(y[nonpp4])) == 2 else float("nan")
        summary = {"mode": mode, "n": len(y), "auc": auc_all, "auc_nonpp4": auc_np4,
                   "per_coupon": per_coupon, "max_new_tokens": max_new if is_gen else None,
                   "image_type": args.image_type}
        all_summary[mode] = summary
        print(f">> {mode}: AUC={auc_all:.4f} 非PP4={auc_np4:.4f} | per-coupon {per_coupon}")

        out = REPO / f"experiments/results/paut_vlm_physics_{mode}_{args.tag}.json"
        with open(out, "w") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2, ensure_ascii=False)
        print(f"   -> {out}")

    sumf = REPO / f"experiments/results/paut_vlm_physics_summary_{args.tag}.json"
    with open(sumf, "w") as fh:
        json.dump(all_summary, fh, indent=2, ensure_ascii=False)
    print(f"\n=== 汇总 ({args.tag}) ===\n{json.dumps(all_summary, indent=2, ensure_ascii=False)}")
    best = max((v.get("auc_nonpp4", 0) or 0) for v in all_summary.values())
    print(f"最佳非PP4 AUC={best:.4f} (P2 baseline=0.593, 目标≥0.62)")


if __name__ == "__main__":
    main()
