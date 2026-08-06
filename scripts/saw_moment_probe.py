#!/usr/bin/env python
"""MOMENT frozen-feature linear probe on SAW defect detection.

Equivalent to MomentClassifier training but ~10x faster: precompute the MOMENT
encoder embeddings for every window ONCE (backbone is frozen), cache them, then
train the same probe head (LayerNorm+Dropout+Linear) on the cached features for
3 seeds. Results are written in the same schema as saw_train.py.

Usage:
  python scripts/saw_moment_probe.py                    # precompute + 3 seeds
  python scripts/saw_moment_probe.py --seeds 42 43 44 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from wndt.data.saw_dataset import SAWSeriesDataset  # noqa: E402
from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402

FEAT_DIM = None  # set after first embedding


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/saw_moment.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--cache", default="data/processed/saw/moment_feats.npz")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    return ap.parse_args()


@torch.no_grad()
def compute_embeddings(model, ds, device, batch_size=512):
    model.eval()
    embs, labels = [], []
    idx = np.arange(len(ds))
    for i in range(0, len(idx), batch_size):
        batch = idx[i:i + batch_size]
        xs = torch.stack([ds[int(j)][0] for j in batch]).to(device)
        x = model._adapt(xs.float())
        B, C, L = x.shape
        mask = torch.ones(B, L, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.backbone.embed(x_enc=x, input_mask=mask, reduction="none")
        e = out.embeddings.float().mean(dim=2).flatten(1)  # (B, C*d)
        embs.append(e.cpu().numpy())
        labels.append(ds.labels[batch])
    return np.concatenate(embs), np.concatenate(labels)


class ProbeHead(nn.Module):
    def __init__(self, dim, n_classes=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout),
                                 nn.Linear(dim, n_classes))

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, Xva, yva, *, dim, seed, epochs, lr, bs, device, n_classes=2):
    set_seed(seed)
    head = ProbeHead(dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    # weighted sampler via class weights in loss
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    w = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    best_f1, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            va_logits = head(Xva_t)
            va_prob = torch.softmax(va_logits, 1)[:, 1].cpu().numpy()
        m = compute_metrics(yva, (va_prob > 0.5).astype(int), va_prob)
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_f1


@torch.no_grad()
def evaluate(head, X, y, device):
    head.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    prob = torch.softmax(head(Xt), 1)[:, 1].cpu().numpy()
    return compute_metrics(y, (prob > 0.5).astype(int), prob)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    processed = args.repo_root / cfg.data.get("processed_dir", "data/processed/saw")
    norm_mode = cfg.data.get("norm_mode", "global")
    splits = np.load(processed / "splits.npz", allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    if cache.exists():
        print(f"loading cached embeddings from {cache}")
        z = np.load(cache)
        Xtr, ytr = z["Xtr"], z["ytr"]
        Xva, yva = z["Xva"], z["yva"]
        Xte, yte = z["Xte"], z["yte"]
    else:
        print("loading MOMENT model ...")
        from wndt.models.moment_model import MomentClassifier
        m = MomentClassifier(n_classes=2, n_channels=4, seq_len=512, freeze=True,
                             use_bf16=True).to(device)
        train_ds = SAWSeriesDataset(processed, splits["train"], norm_mode)
        val_ds = SAWSeriesDataset(processed, splits["val"], norm_mode)
        test_ds = SAWSeriesDataset(processed, splits["test"], norm_mode)
        print(f"precomputing embeddings: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")
        t0 = time.time()
        Xtr, ytr = compute_embeddings(m, train_ds, device, 512)
        Xva, yva = compute_embeddings(m, val_ds, device, 512)
        Xte, yte = compute_embeddings(m, test_ds, device, 512)
        print(f"embedding done in {time.time()-t0:.0f}s | feat dim {Xtr.shape[1]}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    dim = Xtr.shape[1]
    print(f"feat dim={dim} | train {len(Xtr)} val {len(Xva)} test {len(Xte)} | "
          f"defect-rate tr {ytr.mean():.4f} va {yva.mean():.4f} te {yte.mean():.4f}")

    results_dir = args.repo_root / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        head, best_f1 = train_probe(Xtr, ytr, Xva, yva, dim=dim, seed=seed,
                                    epochs=args.epochs, lr=args.lr, bs=args.batch_size,
                                    device=device)
        test_m = evaluate(head, Xte, yte, device)
        val_m = evaluate(head, Xva, yva, device)
        res = {
            "dataset": "saw", "model": "moment", "seed": seed, "mode": "frozen_linear_probe",
            "n_classes": 2, "feat_dim": dim, "epochs": args.epochs,
            "val_metrics": val_m, "test_metrics": test_m,
            "val_macro_f1_best": best_f1, "train_wall_s": round(time.time() - t0, 1),
            "n_params_trainable": dim * 2 + dim * 2 + dim * 2,  # rough head params
            "n_params_total": 341_000_000,
            "majority_baseline_test": majority_baseline(yte),
        }
        out = results_dir / f"saw_moment_seed{seed}.json"
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"seed{seed}: val(f1m {val_m['f1_macro']:.4f} auc {val_m.get('auc',0):.4f}) | "
              f"TEST acc {test_m['acc']:.4f} f1bin {test_m['f1_bin']:.4f} f1m {test_m['f1_macro']:.4f} "
              f"auc {test_m.get('auc',0):.4f} -> {out}")


if __name__ == "__main__":
    main()
