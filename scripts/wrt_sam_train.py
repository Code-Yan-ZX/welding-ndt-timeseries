#!/usr/bin/env python
"""WRT-SAM 复现训练脚本 (GDXray-10, 8 训练 / 2 验证, AdamW lr2e-4 cosine->1e-7, 20ep).

用法:
  python scripts/wrt_sam_train.py --out experiments/wrt-sam/baseline --use-fpg 0 --use-mspg 0
  python scripts/wrt_sam_train.py --out experiments/wrt-sam/wrtsam --use-fpg 1 --use-mspg 1
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrt_sam.data import GDXray10  # noqa: E402
from wrt_sam.model import WRTSAM  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(REPO / "models/pretrained/sam_vit_b_01ec64.pth"))
    ap.add_argument("--inp-size", type=int, default=1024)
    ap.add_argument("--mode", default="strip", choices=["strip", "full"])
    ap.add_argument("--use-fpg", type=str2bool, default=True)
    ap.add_argument("--use-mspg", type=str2bool, default=True)
    ap.add_argument("--freeze-decoder", type=str2bool, default=True)
    ap.add_argument("--loss", default="iou", choices=["iou", "iou_only", "bce"])
    ap.add_argument("--strips-per-image", type=int, default=8,
                    help="strip 模式下训练集每图随机条带数 (1=仅中心条带)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train_ds = GDXray10("train", args.inp_size, args.mode, args.strips_per_image,
                        seed=args.seed)
    val_ds = GDXray10("val", args.inp_size, args.mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    model = WRTSAM(inp_size=args.inp_size, use_fpg=args.use_fpg, use_mspg=args.use_mspg,
                   freeze_decoder=args.freeze_decoder, loss=args.loss).to(device)
    model.load_sam_checkpoint(args.ckpt)
    n_grad = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_grad/1e6:.2f}M / total {n_total/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr_min)

    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            x = batch["inp"].to(device)
            y = batch["gt"].to(device)
            pred = model(x)
            loss = model.compute_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()

        # 验证: 每 epoch 在 2 张 val 上看 IoU (宏观)
        model.eval()
        with torch.no_grad():
            ious = []
            for batch in val_loader:
                pred = torch.sigmoid(model(batch["inp"].to(device))).cpu()
                b = pred > 0.5
                g = batch["gt"] > 0.5
                inter = torch.logical_and(b, g).sum().item()
                union = torch.logical_or(b, g).sum().item()
                ious.append(inter / union if union else 1.0)
        val_iou = sum(ious) / len(ious)
        rec = {"epoch": epoch, "train_loss": tot / n, "val_iou_macro": val_iou}
        history.append(rec)
        print(json.dumps(rec))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       out / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                   out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    print(f"best val_iou_macro {best_iou:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
