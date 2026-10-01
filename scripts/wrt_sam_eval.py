#!/usr/bin/env python
"""WRT-SAM 复现评估脚本.

在 GDXray-10 (全部 10 张, 或 train/val 子集) 上计算 Recall/Precision/AUC/IoU
(micro + macro), 并可选对 GDXray-58 做零样本推理可视化 (无 GT, 定性输出).

用法:
  python scripts/wrt_sam_eval.py --ckpt experiments/wrt-sam/wrtsam/best.pt \
      --out experiments/wrt-sam/wrtsam/eval.json [--gdxray58-vis 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrt_sam.data import GDXray10, GDXray58  # noqa: E402
from wrt_sam.metrics import aggregate, image_metrics  # noqa: E402
from wrt_sam.model import WRTSAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--inp-size", type=int, default=1024)
    ap.add_argument("--mode", default="strip", choices=["strip", "full"])
    ap.add_argument("--gdxray58-vis", type=int, default=0, help="GDXray-58 可视化张数")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    targs = bundle["args"]
    model = WRTSAM(inp_size=args.inp_size, use_fpg=targs["use_fpg"], use_mspg=targs["use_mspg"],
                   freeze_decoder=targs["freeze_decoder"], loss=targs["loss"]).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()

    ds = GDXray10(args.split, args.inp_size, args.mode)
    loader = DataLoader(ds, batch_size=1, num_workers=2)
    preds, gts, per_img = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = torch.sigmoid(model(batch["inp"].to(device)))[0, 0].cpu().numpy()
            gt = batch["gt"][0, 0].numpy()
            preds.append(pred)
            gts.append(gt)
            per_img.append(image_metrics(pred, gt))

    res = aggregate(per_img, preds, gts)
    res["per_image"] = [
        {**m, "image": ds.idxs[i]} for i, m in enumerate(per_img)]
    res["ckpt"] = args.ckpt
    res["split"] = args.split
    res["mode"] = args.mode

    # GDXray-58 零样本可视化 (无官方 GT, 仅定性)
    if args.gdxray58_vis > 0:
        vis_dir = Path(args.out).parent / "gdxray58_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        ds58 = GDXray58(args.inp_size, args.mode)
        with torch.no_grad():
            for i in range(min(args.gdxray58_vis, len(ds58))):
                batch = ds58[i]
                pred = torch.sigmoid(model(batch["inp"][None].to(device)))[0, 0].cpu().numpy()
                img = ((batch["inp"][0].numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
                mask = (pred > 0.5).astype(np.uint8) * 255
                h = args.inp_size
                canvas = np.concatenate([img, mask], axis=1)  # (H, 2W) 灰度: 左原图右预测
                Image.fromarray(canvas, mode="L").save(
                    vis_dir / f"{Path(batch['path']).stem}_pred.png")
        res["gdxray58_vis_dir"] = str(vis_dir)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("micro", "macro")}, indent=1))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
