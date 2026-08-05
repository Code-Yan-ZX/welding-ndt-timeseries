"""Trainer for ITFormerTLM (QA-style likelihood model with frozen LLM).

Only bridge parameters (encoder, ITFormer, q_proj, ts_proj) are trained.
Validation uses likelihood scores: pred = score > 0, AUC from scores.
Checkpoints save only the trainable (bridge) state dict, never the LLM.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wndt.data.dataset import make_weighted_sampler
from wndt.eval.metrics import compute_metrics
from wndt.train.trainer_cls import _cosine_warmup
from wndt.utils.logging import get_logger
from wndt.utils.seed import set_seed

log = get_logger(__name__)


@torch.no_grad()
def eval_qa_model(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, ss = [], []
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score = model.score(x.to(device, non_blocking=True))
        ys.append(y.numpy())
        ss.append(score.float().cpu().numpy())
    y_true = np.concatenate(ys)
    scores = np.concatenate(ss)
    y_pred = (scores > 0).astype(int)
    return compute_metrics(y_true, y_pred, scores)


class QATrainer:
    def __init__(self, model, *, device: torch.device, run_dir: Path,
                 lr: float = 5e-5, weight_decay: float = 0.01, batch_size: int = 16,
                 accum_steps: int = 2, epochs: int = 5, warmup_steps: int = 300,
                 patience: int = 2, grad_clip: float = 1.0,
                 weighted_sampler: bool = True, num_workers: int = 2, seed: int = 42):
        self.model = model.to(device)
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.accum_steps = batch_size, accum_steps
        self.epochs, self.warmup_steps, self.patience = epochs, warmup_steps, patience
        self.grad_clip = grad_clip
        self.weighted_sampler = weighted_sampler
        self.num_workers = num_workers
        self.seed = seed
        self.history: list[dict] = []

    def _make_loader(self, ds: Dataset, train: bool) -> DataLoader:
        if train:
            sampler = (make_weighted_sampler(ds.labels)
                       if self.weighted_sampler else None)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=self.num_workers,
                              pin_memory=True, drop_last=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def _bridge_state(self) -> dict:
        return {
            "encoder": self.model.encoder.state_dict(),
            "itformer": self.model.itformer.state_dict(),
            "q_proj": self.model.q_proj.state_dict(),
            "ts_proj": self.model.ts_proj.state_dict(),
        }

    def fit(self, train_ds: Dataset, val_ds: Dataset) -> dict:
        set_seed(self.seed)
        train_loader = self._make_loader(train_ds, train=True)
        val_loader = self._make_loader(val_ds, train=False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = (len(train_loader) // self.accum_steps) * self.epochs

        # step-0 prior skew check (random bridge): |mean score| should be small
        prior = eval_qa_model(self.model, val_loader, self.device)
        log.info("step-0 val metrics (random bridge): %s", prior)

        best_f1, best_state, bad_epochs = -1.0, None, 0
        t0, step, accum = time.time(), 0, 0
        for epoch in range(self.epochs):
            self.model.train()
            tot_loss, n = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                lr_now = self.lr * _cosine_warmup(step, self.warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self.model(x, y) / self.accum_steps
                loss.backward()
                tot_loss += loss.item() * self.accum_steps * len(y)
                n += len(y)
                accum += 1
                if accum % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
            val_m = eval_qa_model(self.model, val_loader, self.device)
            train_loss = tot_loss / max(1, n)
            self.history.append({"epoch": epoch, "train_loss": train_loss, **
                                 {f"val_{k}": v for k, v in val_m.items()}})
            vram = (torch.cuda.max_memory_allocated(self.device) / 2**30
                    if torch.cuda.is_available() else 0.0)
            log.info("epoch %d | loss %.4f | val acc %.4f f1m %.4f auc %.4f | peak VRAM %.1fG",
                     epoch, train_loss, val_m["acc"], val_m["f1_macro"],
                     val_m.get("auc", float("nan")), vram)
            if val_m["f1_macro"] > best_f1 + 1e-4:
                best_f1 = val_m["f1_macro"]
                best_state = self._bridge_state()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    log.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self._load_bridge(best_state)
        torch.save(best_state, self.run_dir / "best_bridge.pt")
        wall = time.time() - t0
        with open(self.run_dir / "train_log.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        return {"val_macro_f1_best": best_f1, "wall_s": wall,
                "epochs_run": len(self.history),
                "peak_vram_gb": (torch.cuda.max_memory_allocated(self.device) / 2**30
                                 if torch.cuda.is_available() else 0.0)}

    def _load_bridge(self, state: dict) -> None:
        self.model.encoder.load_state_dict(state["encoder"])
        self.model.itformer.load_state_dict(state["itformer"])
        self.model.q_proj.load_state_dict(state["q_proj"])
        self.model.ts_proj.load_state_dict(state["ts_proj"])

    def evaluate(self, ds: Dataset, split: str) -> dict[str, float]:
        loader = self._make_loader(ds, train=False)
        metrics = eval_qa_model(self.model, loader, self.device)
        log.info("[%s] %s", split, metrics)
        return metrics
