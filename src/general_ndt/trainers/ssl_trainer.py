"""SSL trainer (M0 vanilla MAE): 数据加载/确定性 seed/优化/日志/checkpoint。

设计 (Phase 2A):
- 确定性: data_seed 控制 shuffle 与每步 mask 采样; model_seed 控制参数初始化;
  seed 职责分离 (data_seed / model_seed)。
- checkpoint: model state_dict + 训练配置 + 数据集指纹 (sample_id 哈希) + step/loss。
  load 时校验数据集指纹一致 (防错载/跨数据集串用)。
- 逐折评测由 scripts/general_ndt_probe.py 完成 (本 trainer 只负责预训练)。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from general_ndt.datasets.collate import collate_general_ndt
from general_ndt.datasets.schema import GeneralNDTSample

logger = logging.getLogger("general_ndt.ssl_trainer")


def dataset_fingerprint(samples: Sequence[GeneralNDTSample]) -> str:
    """样本 ID 的有序集合哈希 —— 检测 checkpoint 与数据集是否一致。"""
    ids = sorted(s.sample_id for s in samples)
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:16]


class SSLTrainer:
    def __init__(self, model: nn.Module, config: dict, device: str | torch.device = "cpu"):
        self.model = model
        self.cfg = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        self.model_seed = int(config.get("model_seed", 0))
        self.data_seed = int(config.get("data_seed", 0))

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """按 model_seed 重新初始化模型参数 (确定性)。"""
        torch.manual_seed(self.model_seed)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalize(self, nb) -> None:
        """per-sample z-score (只在有效位置计算均值/方差)。MAE 目标为原始 patch,
        标准化后 loss 量级 ~O(1), 且训练/探针一致。"""
        if self.cfg.get("normalize", "per_sample") != "per_sample":
            return
        sig = nb.padded_signal
        vm = nb.valid_mask
        for b in range(nb.batch_size):
            sel = sig[b, :, vm[b].astype(bool)]
            if sel.size == 0:
                continue
            mu, sd = float(sel.mean()), float(sel.std())
            if sd < 1e-8:
                sd = 1.0
            sig[b, :, vm[b].astype(bool)] = (sig[b, :, vm[b].astype(bool)] - mu) / sd

    def _build_batches(self, samples: Sequence[GeneralNDTSample], batch_size: int
                       ) -> list[dict]:
        """确定性 shuffle + collate + per-sample 标准化 → tensor batch dict 列表。"""
        rng = np.random.default_rng(self.data_seed)
        idx = rng.permutation(len(samples)).tolist()
        batches = []
        for i in range(0, len(idx), batch_size):
            chunk = [samples[j] for j in idx[i : i + batch_size]]
            nb = collate_general_ndt(chunk)
            self._normalize(nb)
            batches.append({
                "x": torch.from_numpy(nb.padded_signal).float(),
                "padded_signal": nb.padded_signal,      # numpy (token mask/patchify 用)
                "valid_mask": nb.valid_mask,            # numpy 样本级
                "shape_kind": nb.shape_kind,
                "modalities": nb.modalities,
                "sampling_rates": [s.sampling_rate for s in chunk],
                "sensor_ids": [s.sensor_id for s in chunk],
                "shapes": nb.shapes,
                "sample_ids": nb.sample_ids,
                "specimen_ids": nb.specimen_ids,
                "defect_regions": [s.metadata.get("defect_regions") or []
                                   for s in chunk],
            })
        return batches

    # ------------------------------------------------------------------
    def train(self, samples: Sequence[GeneralNDTSample], n_steps: int,
              batch_size: int = 16, log_every: int = 10, ckpt_every: int = 500,
              output_dir: str | Path = "experiments/runs/general_ndt_mae") -> Path:
        """训练 n_steps 步, 定期保存 checkpoint。返回最终 checkpoint 路径。"""
        self._init_weights()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        batches = self._build_batches(samples, batch_size)
        if not batches:
            raise ValueError("空数据集, 无法训练")
        fp = dataset_fingerprint(samples)
        n_batches = len(batches)
        step = 0
        best_loss = float("inf")
        final_ckpt = None
        while step < n_steps:
            for bi, tb in enumerate(batches):
                if step >= n_steps:
                    break
                tb = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in tb.items()}
                self.model.train()
                self.optimizer.zero_grad()
                mask_seed = self.data_seed * 1_000_000 + step
                out_d = self.model(tb, mask_seed=mask_seed)
                loss = out_d["loss"]
                loss.backward()
                self.optimizer.step()
                cur = float(loss.detach().item())
                if cur < best_loss:
                    best_loss = cur
                if (step + 1) % log_every == 0 or step == n_steps - 1:
                    logger.info(
                        f"[step {step+1}/{n_steps}] loss={cur:.6f} best={best_loss:.6f} "
                        f"masked_frac={float(out_d['mask'].sum())/max(1, int(out_d['valid'].sum())):.3f}")
                if (step + 1) % ckpt_every == 0 or step == n_steps - 1:
                    ckpt = out / f"mae_step{step+1}.pt"
                    self.save_checkpoint(ckpt, step=step + 1, loss=cur, fingerprint=fp)
                    final_ckpt = ckpt
                step += 1
        return final_ckpt

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sources(sources: list) -> list[tuple[str, Sequence[GeneralNDTSample], dict | None]]:
        """源规格归一化: (name, samples) 元组 → policy=None (random);
        dict {"name","samples","mask_policy"} → 带 per-source 掩码策略 (F 系列)。"""
        norm = []
        for src in sources:
            if isinstance(src, dict):
                norm.append((src["name"], src["samples"], src.get("mask_policy")))
            else:
                name, samples = src
                norm.append((name, samples, None))
        return norm

    def _training_signature(self, norm_sources: list, n_steps: int, batch_size: int,
                            fingerprint: str, order: list[int] | None = None) -> str:
        """训练签名 (F 系列): 数据指纹只看 sample_id, 区分不了同数据不同 mask_policy
        的训练 (F4 region_bias vs F5 random) — 签名把源规格+策略+训练配置一起哈希,
        load 时校验, 防止策略不同的 checkpoint 互串。"""
        adapter = getattr(self.model, "adapter", None)
        desc = {
            "sources": [
                (name, len(ss), json.dumps(mp, sort_keys=True) if mp else None)
                for name, ss, mp in norm_sources
            ],
            "order": order if order is None else [int(i) for i in order],
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "mask_ratio": self.cfg.get("mask_ratio"),
            "normalize": self.cfg.get("normalize", "per_sample"),
            "lr": self.cfg.get("lr"), "weight_decay": self.cfg.get("weight_decay"),
            "data_seed": self.data_seed,
            "per_modality_stem": getattr(adapter, "per_modality_stem", None),
            "dataset_fingerprint": fingerprint,
        }
        return hashlib.sha256(json.dumps(desc, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def train_multi(self, sources: list, n_steps: int, batch_size: int = 16,
                    log_every: int = 10, ckpt_every: int = 500,
                    output_dir: str | Path = "experiments/runs/general_ndt_mae_multi",
                    order: list[int] | None = None) -> Path:
        """多源自监督训练 (E2/E2b/F 系列): 多源在同一共享 token 空间联合预训练。

        - 每源独立构建 batch (per-sample 标准化 + data_seed shuffle);
        - **确定性交替采样**: order=None 时每步按源轮转 (E2/E2b 行为, 模态平衡);
          order=源索引序列 (F 系列: 按步数配额展开的加权轮转表, len == n_steps);
        - 每源可带独立 mask_policy (F 系列: PENELOPE/EddyCus/extUT random,
          SWRD region_bias) — 每步按当前源策略采样;
        - 全部源的样本联合指纹 + **训练签名** 写入 checkpoint (防跨数据集/跨策略串用)。
        """
        self._init_weights()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        norm_sources = self._normalize_sources(list(sources))
        batch_lists = []
        for name, samples, _ in norm_sources:
            bl = self._build_batches(samples, batch_size)
            if not bl:
                raise ValueError(f"源 '{name}' 为空, 无法训练")
            batch_lists.append(bl)
            logger.info(f"[multi] 源 '{name}': {len(samples)} samples, {len(bl)} batches")
        all_samples = [s for _, ss, _ in norm_sources for s in ss]
        fp = dataset_fingerprint(all_samples)
        sig = self._training_signature(norm_sources, n_steps, batch_size, fp, order=order)
        if order is None:
            order = []
            while len(order) < n_steps:
                for si in range(len(batch_lists)):
                    order.append(si)
        assert len(order) == n_steps, f"order 长度 {len(order)} ≠ n_steps {n_steps}"
        pointers = [0] * len(batch_lists)
        step, best_loss, final_ckpt = 0, float("inf"), None
        while step < n_steps:
            for si in order:
                if step >= n_steps:
                    break
                bl = batch_lists[si]
                tb = bl[pointers[si] % len(bl)]
                pointers[si] += 1
                tb = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in tb.items()}
                self.model.train()
                self.optimizer.zero_grad()
                mask_seed = self.data_seed * 1_000_000 + step
                out_d = self.model(tb, mask_seed=mask_seed,
                                   mask_policy=norm_sources[si][2])
                loss = out_d["loss"]
                loss.backward()
                self.optimizer.step()
                cur = float(loss.detach().item())
                if cur < best_loss:
                    best_loss = cur
                if (step + 1) % log_every == 0 or step == n_steps - 1:
                    logger.info(
                        f"[step {step+1}/{n_steps}] src={si}({norm_sources[si][0]}) "
                        f"loss={cur:.6f} best={best_loss:.6f}")
                if (step + 1) % ckpt_every == 0 or step == n_steps - 1:
                    ckpt = out / f"mae_step{step+1}.pt"
                    self.save_checkpoint(ckpt, step=step + 1, loss=cur,
                                         fingerprint=fp, signature=sig)
                    final_ckpt = ckpt
                step += 1
        return final_ckpt

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, step: int, loss: float,
                        fingerprint: str, signature: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.cfg,
            "step": step,
            "loss": loss,
            "dataset_fingerprint": fingerprint,
            "training_signature": signature,
        }, path)

    def load_checkpoint(self, path: str | Path, expected_fingerprint: str | None = None,
                        expected_signature: str | None = None) -> dict:
        path = Path(path)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        fp = ck.get("dataset_fingerprint")
        if expected_fingerprint is not None and fp is not None and fp != expected_fingerprint:
            raise ValueError(
                f"checkpoint 数据集指纹 {fp} 与当前数据集 {expected_fingerprint} 不一致; "
                f"禁止跨数据集串用 checkpoint")
        sg = ck.get("training_signature")
        if expected_signature is not None and sg is not None and sg != expected_signature:
            raise ValueError(
                f"checkpoint 训练签名 {sg} 与当前训练配置 {expected_signature} 不一致; "
                f"禁止跨策略 (mask_policy/步数/架构) 串用 checkpoint")
        self.model.load_state_dict(ck["model_state_dict"])
        return ck

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, samples: Sequence[GeneralNDTSample],
                         batch_size: int = 32) -> tuple[np.ndarray, list]:
        """冻结编码, 全视图 (unmasked) pooled 表征 → (N, d) + sample_ids。"""
        self.model.eval()
        feats = []
        ids = []
        batches = self._build_batches(samples, batch_size)
        for tb in batches:
            tb = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in tb.items()}
            pooled = self.model.encode_raw(tb)
            feats.append(pooled.cpu().numpy())
            ids.extend(tb["sample_ids"])
        return np.concatenate(feats, axis=0), ids
