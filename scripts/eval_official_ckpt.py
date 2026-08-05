#!/usr/bin/env python
"""Re-evaluate an OFFICIAL tmdt-buw checkpoint on OUR canonical splits.

Runs inside the official repo's env/imports so the exact same models and data
pipeline are used. We override only the (experiment, welding_run) split pairs,
passing them in the PAPER convention (val = small overlap-joint set, test =
T-joint set) regardless of the official repo's swapped variable names.

Usage (activate vqvae-welding env first):
  python scripts/eval_official_ckpt.py --model-kind transformer \
      --ckpt third_party/.../model_checkpoints/VQ-VAE-transformer/last.ckpt \
      --vqvae-ckpt third_party/.../model_checkpoints/VQ-VAE-Patch/<best>.ckpt
model-kind: transformer | mlp | gru | vqvae_mlp | vqvae_gru
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "third_party/VQ-VAE-Transformer-Arc-Welding"
sys.path.insert(0, str(OFFICIAL))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule  # noqa: E402
from dataloader.latentspace_dataloader import LatentPredDataModule  # noqa: E402
from model.mlp import MLP  # noqa: E402
from model.gru import GRU  # noqa: E402
from model.transformer_decoder import MyTransformerDecoder  # noqa: E402
from model.vq_vae_patch_embedd import VQVAEPatch  # noqa: E402

# CANONICAL (paper / Zenodo README) convention:
CANON_VAL = [(3, 32), (3, 18), (1, 27), (3, 19), (3, 17), (2, 21), (1, 20), (1, 11)]
CANON_TEST = [(3, 3), (2, 10), (1, 24), (3, 24), (1, 32), (2, 1), (1, 10), (1, 16)]


def to_split_ids(pairs):
    return [DataSplitId(experiment=e, welding_run=w) for e, w in pairs]


def metrics_from(y_true, prob_pos):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(prob_pos) > 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_bin": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob_pos))
    return out


def collect(model, loader, device, kind):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if kind == "transformer":
                x, cond, _ = batch
                x = x.to(device)
                logits = model(x, generate=False)
                y = cond.view(-1)
            elif kind in ("mlp", "gru"):
                x, y = batch
                x = x.to(device)
                logits = model(x)
            elif kind in ("vqvae_mlp", "vqvae_gru"):
                x, y = batch
                x = x.to(device)
                logits = model(x)
            else:
                raise ValueError(kind)
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-kind", required=True,
                    choices=["transformer", "mlp", "gru", "vqvae_mlp", "vqvae_gru"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vqvae-ckpt", default=None,
                    help="required for transformer / vqvae_* kinds")
    ap.add_argument("--n-cycles", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    # absolutize before chdir'ing into the official repo
    args.ckpt = str(Path(args.ckpt).resolve())
    if args.vqvae_ckpt:
        args.vqvae_ckpt = str(Path(args.vqvae_ckpt).resolve())
    if args.out:
        args.out = str(Path(args.out).resolve())
    os.chdir(OFFICIAL)  # official code resolves data/ relative to cwd
    torch.set_float32_matmul_precision("medium")
    device = torch.device(args.device)

    val_ids = to_split_ids(CANON_VAL)
    test_ids = to_split_ids(CANON_TEST)
    kind = args.model_kind

    if kind == "transformer":
        assert args.vqvae_ckpt, "--vqvae-ckpt required for transformer"
        n_cycles = args.n_cycles or 20
        vqvae = VQVAEPatch.load_from_checkpoint(args.vqvae_ckpt)
        dm = LatentPredDataModule(latent_space_model=vqvae, model_name="VQ-VAE-Patch",
                                  val_data_ids=val_ids, test_data_ids=test_ids,
                                  n_cycles=n_cycles, task="autoregressive_ids_classification",
                                  batch_size=args.batch_size, model_id=Path(args.ckpt).stem)
        model = MyTransformerDecoder.load_from_checkpoint(args.ckpt)
        model.switch_to_classification()
    elif kind in ("mlp", "gru"):
        n_cycles = args.n_cycles or 5
        dm = ASIMoWDataModule(task="classification", batch_size=args.batch_size,
                              n_cycles=n_cycles, val_data_ids=val_ids,
                              test_data_ids=test_ids)
        Model = MLP if kind == "mlp" else GRU
        model = Model.load_from_checkpoint(args.ckpt)
    else:  # vqvae_mlp / vqvae_gru
        assert args.vqvae_ckpt, "--vqvae-ckpt required for vqvae_* kinds"
        n_cycles = args.n_cycles or 5
        vqvae = VQVAEPatch.load_from_checkpoint(args.vqvae_ckpt)
        dm = LatentPredDataModule(latent_space_model=vqvae, model_name="VQ-VAE-Patch",
                                  val_data_ids=val_ids, test_data_ids=test_ids,
                                  n_cycles=n_cycles, task="classification",
                                  batch_size=args.batch_size, model_id=Path(args.ckpt).stem)
        Model = MLP if kind == "vqvae_mlp" else GRU
        model = Model.load_from_checkpoint(args.ckpt)

    model = model.to(device)
    dm.setup("fit")

    result = {"model": f"official_{kind}", "llm": None, "seed": 42,
              "norm_mode": "official(MyScaler/global)", "smoke": False,
              "config": {"ckpt": args.ckpt, "vqvae_ckpt": args.vqvae_ckpt,
                         "n_cycles": n_cycles,
                         "canonical_val_pairs": CANON_VAL,
                         "canonical_test_pairs": CANON_TEST},
              "note": "official tmdt-buw checkpoint re-evaluated on canonical "
                      "(paper-convention) splits"}

    for split, loader in (("val", dm.val_dataloader()), ("test", dm.test_dataloader())):
        y, p = collect(model, loader, device, kind)
        m = metrics_from(y, p)
        result["val_metrics" if split == "val" else "test_metrics"] = m
        print(f"[{split}] n={len(y)} pos_rate={y.mean():.4f} -> {m}", flush=True)

    out = Path(args.out) if args.out else REPO / "experiments/results" / f"official_{kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"written: {out}", flush=True)


if __name__ == "__main__":
    main()
