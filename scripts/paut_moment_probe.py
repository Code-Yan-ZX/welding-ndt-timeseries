#!/usr/bin/env python
"""MOMENT frozen-feature linear probe on PAUT defect detection.

Embeds the per-position max-over-beams envelope (1 channel, 512 samples) - the
envelope preserves the defect echo (max over the 49-beam aperture) and carries
the position-level label directly, with no per-beam mislabeling (we have no
beam-level annotations).  Precomputes MOMENT-1 embeddings for every position
ONCE (backbone frozen), caches them, then trains a probe head (LayerNorm +
Dropout + Linear) for 3 seeds.  Position-level eval on PP7 (test).

Result schema matches paut_train.py / saw_moment_probe.py.
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

from wndt.eval.metrics import compute_metrics, majority_baseline  # noqa: E402
from wndt.utils.config import load_config  # noqa: E402
from wndt.utils.seed import set_seed  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paut_moment.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--cache", default="data/processed/paut/moment_feats.npz")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    return ap.parse_args()


@torch.no_grad()
def compute_embeddings(model, env, idx, ts_mean, ts_std, device, batch=512):
    """env: (N, 512) memmap; idx: position indices -> (len(idx), feat)."""
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        chunk = idx[i:i + batch]
        xs = np.array(env[chunk], dtype=np.float32)             # (B, 512)
        xs = (xs - ts_mean) / ts_std
        x = torch.from_numpy(xs).unsqueeze(1).to(device)        # (B, 1, 512)
        x = model._adapt(x.float())
        B, C, L = x.shape
        mask = torch.ones(B, L, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            o = model.backbone.embed(x_enc=x, input_mask=mask, reduction="none")
        e = o.embeddings.float().mean(dim=2).flatten(1)         # (B, C*d)
        out.append(e.cpu().numpy())
    return np.concatenate(out, axis=0)                          # (P, d)


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
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    w = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    best_auc, best_state = -1.0, None
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
            va_prob = torch.softmax(head(Xva_t), 1)[:, 1].cpu().numpy()
        m = compute_metrics(yva, (va_prob > 0.5).astype(int), va_prob)
        if m.get("auc", -1) > best_auc:
            best_auc = m.get("auc", -1)
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_auc


def best_threshold(y_val, s_val):
    from sklearn.metrics import f1_score
    cand = np.unique(s_val)
    if len(cand) > 200:
        cand = np.quantile(s_val, np.linspace(0.01, 0.99, 200))
    best_t, best_f = 0.5, -1.0
    for t in cand:
        f = f1_score(y_val, (s_val > t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


@torch.no_grad()
def evaluate(head, X, y, device, thr=0.5):
    head.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    prob = torch.softmax(head(Xt), 1)[:, 1].cpu().numpy()
    return compute_metrics(y, (prob > thr).astype(int), prob), prob


def main():
    args = parse_args()
    cfg = load_config(args.config)
    processed = args.repo_root / cfg.data.get("processed_dir", "data/processed/paut")
    splits = np.load(processed / "splits.npz", allow_pickle=True)
    env = np.load(processed / "env.npy", mmap_mode="r")
    labels = np.load(processed / "meta_label.npy")
    with open(processed / "norm_stats.json") as fh:
        stats = json.load(fh)
    ts_mean = np.asarray(stats["per_timestep"]["mean"], dtype=np.float32)
    ts_std = np.asarray(stats["per_timestep"]["std"], dtype=np.float32)
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
        m = MomentClassifier(n_classes=2, n_channels=1, seq_len=512, freeze=True,
                             use_bf16=True).to(device)
        print("precomputing envelope embeddings ...")
        t0 = time.time()
        Xtr = compute_embeddings(m, env, splits["train"], ts_mean, ts_std, device)
        Xva = compute_embeddings(m, env, splits["val"], ts_mean, ts_std, device)
        Xte = compute_embeddings(m, env, splits["test"], ts_mean, ts_std, device)
        print(f"embedding done in {time.time()-t0:.0f}s | shape={Xtr.shape}")
        ytr, yva, yte = labels[splits["train"]], labels[splits["val"]], labels[splits["test"]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    dim = Xtr.shape[-1]
    print(f"feat dim={dim} | train {len(Xtr)} val {len(Xva)} test {len(Xte)} | "
          f"defect-rate tr {ytr.mean():.4f} va {yva.mean():.4f} te {yte.mean():.4f}")

    results_dir = args.repo_root / "experiments/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        head, best_auc = train_probe(Xtr, ytr, Xva, yva, dim=dim, seed=seed,
                                     epochs=args.epochs, lr=args.lr, bs=args.batch_size,
                                     device=device)
        _, va_prob = evaluate(head, Xva, yva, device)
        thr = best_threshold(yva, va_prob)
        test_m, _ = evaluate(head, Xte, yte, device, thr)
        val_m, _ = evaluate(head, Xva, yva, device, thr)
        test_m["threshold"] = thr
        res = {
            "dataset": "paut", "model": "moment", "seed": seed,
            "mode": "frozen_linear_probe_env", "n_classes": 2,
            "feat_dim": dim, "input": "envelope(1,512)", "epochs": args.epochs,
            "val_metrics": val_m, "test_metrics": test_m,
            "val_macro_f1_best": best_auc, "train_wall_s": round(time.time() - t0, 1),
            "n_params_trainable": dim * 2 + dim * 2 + dim * 2,
            "n_params_total": 341_000_000,
            "majority_baseline_test": majority_baseline(yte),
        }
        out = results_dir / f"paut_moment_seed{seed}.json"
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"seed{seed}: val(f1m {val_m['f1_macro']:.4f} auc {val_m.get('auc',0):.4f}) | "
              f"TEST acc {test_m['acc']:.4f} f1bin {test_m['f1_bin']:.4f} "
              f"f1m {test_m['f1_macro']:.4f} auc {test_m.get('auc',0):.4f} -> {out}")


if __name__ == "__main__":
    main()
