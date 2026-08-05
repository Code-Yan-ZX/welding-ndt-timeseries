"""Trainer for classification models that output logits (probe, encoder_only,
MLP/LSTM/GRU, DLinear, TimesNet).

Protocol: AdamW + cosine schedule with linear warmup, WeightedRandomSampler
(official protocol) by default, early stopping on val macro-F1 with best-ckpt
restore, single test evaluation at the end.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from wndt.data.dataset import make_weighted_sampler
from wndt.eval.metrics import compute_metrics
from wndt.utils.logging import get_logger
from wndt.utils.seed import set_seed

log = get_logger(__name__)


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def eval_logits_model(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> dict[str, float]:
    model.eval()
    ys, ps, ss = [], [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        prob = torch.softmax(logits, dim=1)[:, 1].cpu()
        ys.append(y.numpy())
        ps.append((prob > 0.5).long().numpy())
        ss.append(prob.numpy())
    y_true = np.concatenate(ys)
    return compute_metrics(y_true, np.concatenate(ps), np.concatenate(ss))


class ClassificationTrainer:
    def __init__(self, model: nn.Module, *, device: torch.device, run_dir: Path,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256,
                 epochs: int = 30, warmup_steps: int = 300, patience: int = 8,
                 grad_clip: float = 1.0, weighted_sampler: bool = True,
                 num_workers: int = 4, seed: int = 42):
        self.model = model.to(device)
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs = batch_size, epochs
        self.warmup_steps, self.patience = warmup_steps, patience
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

    def fit(self, train_ds: Dataset, val_ds: Dataset) -> dict:
        set_seed(self.seed)
        train_loader = self._make_loader(train_ds, train=True)
        val_loader = self._make_loader(val_ds, train=False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        total_steps = len(train_loader) * self.epochs
        loss_fn = nn.CrossEntropyLoss()

        best_f1, best_state, bad_epochs = -1.0, None, 0
        t0 = time.time()
        step = 0
        for epoch in range(self.epochs):
            self.model.train()
            tot_loss, n = 0.0, 0
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                lr_now = self.lr * _cosine_warmup(step, self.warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                tot_loss += loss.item() * len(y)
                n += len(y)
                step += 1
            val_m = eval_logits_model(self.model, val_loader, self.device)
            train_loss = tot_loss / max(1, n)
            self.history.append({"epoch": epoch, "train_loss": train_loss, **
                                 {f"val_{k}": v for k, v in val_m.items()}})
            log.info("epoch %d | loss %.4f | val acc %.4f f1m %.4f",
                     epoch, train_loss, val_m["acc"], val_m["f1_macro"])
            if val_m["f1_macro"] > best_f1 + 1e-4:
                best_f1 = val_m["f1_macro"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    log.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save(best_state, self.run_dir / "best_model.pt")
        wall = time.time() - t0
        with open(self.run_dir / "train_log.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        return {"val_macro_f1_best": best_f1, "wall_s": wall,
                "epochs_run": len(self.history)}

    def evaluate(self, ds: Dataset, split: str) -> dict[str, float]:
        loader = self._make_loader(ds, train=False)
        metrics = eval_logits_model(self.model, loader, self.device)
        log.info("[%s] %s", split, metrics)
        return metrics
