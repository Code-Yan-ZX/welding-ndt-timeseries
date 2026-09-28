"""M0: vanilla Masked Autoencoder (random mask) — Phase 2A 最小闭环。

流程 (每 batch):
  1. ModalAdapter: 原始信号 → token 网格 (B, N, d), grid (1d: (C, n_col); 2d: (C, n_h, n_w))
  2. token 级 valid mask (B, N): 来自 batch (样本长度 padding + channel padding + native 空洞)
  3. MaskController(random, mask_ratio): 只在 valid token 中采样 mask (B, N)
  4. masked token 替换: 被掩 token 位置注入可学习 mask_token
  5. PatchTransformer encoder (含网格位置编码 + src_key_padding_mask)
  6. 轻量 decoder (decoder_embed + PatchTransformer + decoder_head)
  7. 预测被掩 patch 的原始值 → masked_recon_loss (只算 masked ∩ valid)
  8. encode_raw: 全视图 (unmasked) 编码 → pooled 表征, 供冻结线性探针

只启用 M0: vanilla MAE + random mask (E0–E2b); F 系列扩展: mask_policy
{"mode":"region_bias","region_frac":f} 缺陷感知掩码 (ssl/region_masks.py)。
时频双视图 / 跨传感器不变性 暂不启用。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from general_ndt.adapters.base import ModalAdapter
from general_ndt.models.backbone import PatchTransformer
from general_ndt.ssl.masking import MaskController
from general_ndt.ssl.objectives import masked_recon_loss
from general_ndt.ssl.region_masks import sample_region_bias_mask
from general_ndt.ssl.token_masks import patchify_target, token_valid_mask


class MaskedAutoencoder(nn.Module):
    """M0 vanilla MAE (random mask, masked reconstruction, token 级 valid mask)。"""

    def __init__(
        self,
        d_model: int = 128,
        patch_len: int = 16,
        patch2d: int = 16,
        n_layers_enc: int = 4,
        n_heads: int = 4,
        d_decoder: int = 128,
        n_layers_dec: int = 2,
        mask_ratio: float = 0.5,
        n_modalities: int = 8,
        n_sensors: int = 32,
        dropout: float = 0.1,
        max_rows: int = 128,
        per_modality_stem: bool = False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch2d = patch2d
        self.mask_ratio = mask_ratio
        self.adapter = ModalAdapter(
            d_model=d_model, patch_len=patch_len, patch2d=patch2d,
            n_modalities=n_modalities, n_sensors=n_sensors,
            per_modality_stem=per_modality_stem,
        )
        self.encoder = PatchTransformer(
            d_model=d_model, n_layers=n_layers_enc, n_heads=n_heads,
            dropout=dropout, max_rows=max_rows,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_embed = nn.Linear(d_model, d_decoder)
        self.decoder = PatchTransformer(
            d_model=d_decoder, n_layers=n_layers_dec, n_heads=n_heads,
            dropout=dropout, max_rows=max_rows,
        )
        # 解码头: 1d 预测 patch_len 原始点; 2d 预测 patch2d*patch2d 原始像素
        self.pred_1d = nn.Linear(d_decoder, patch_len)
        self.pred_2d = nn.Linear(d_decoder, patch2d * patch2d)
        self.mask_controller = MaskController("random", mask_ratio)

    # ------------------------------------------------------------------
    def _tokens_and_valid(self, batch: dict, device: torch.device):
        x = batch["x"].to(device)
        shape_kind = batch["shape_kind"]
        tokens, grid = self.adapter(
            x, shape_kind, batch["modalities"],
            batch.get("sampling_rates"), batch.get("sensor_ids"),
        )
        # numpy token valid (B, N) → tensor
        tv_np = token_valid_mask(batch, self.patch_len, self.patch2d)
        tv = torch.from_numpy(tv_np).to(device)
        return tokens, grid, tv, tv_np

    def _sample_mask(self, tv_np: np.ndarray, grid: tuple, device: torch.device,
                     mask_seed: int | None, mask_policy: dict | None = None,
                     defect_regions: list | None = None) -> torch.Tensor:
        """按样本采样掩码。mask_policy None/{"mode":"random"} → 既有 MaskController
        (random, E0–E2b 路径不变); {"mode":"region_bias","region_frac":f} →
        缺陷感知采样 (该样本无 defect_regions 时自动退化为 random)。"""
        B = tv_np.shape[0]
        policy = mask_policy or {"mode": "random"}
        if policy.get("mode", "random") not in ("random", "region_bias"):
            raise ValueError(f"未知 mask_policy.mode: {policy.get('mode')}")
        region_bias = policy.get("mode") == "region_bias"
        region_frac = float(policy.get("region_frac", 0.5))
        masks = []
        for b in range(B):
            seed = None if mask_seed is None else mask_seed + b
            grid_b = tv_np[b].reshape(grid)
            if region_bias:
                regions = (defect_regions or [])[b] if defect_regions else []
                masks.append(
                    sample_region_bias_mask(
                        grid, self.mask_ratio, grid_b, regions,
                        region_frac=region_frac, seed=seed,
                    )
                )
            else:
                masks.append(self.mask_controller(grid, valid=grid_b, seed=seed))
        return torch.from_numpy(np.stack(masks).reshape(B, -1)).to(device)

    def forward(self, batch: dict, mask_seed: int | None = None,
                mask_policy: dict | None = None) -> dict:
        tokens, grid, tv, tv_np = self._tokens_and_valid(batch, batch["x"].device)
        mask = self._sample_mask(tv_np, grid, batch["x"].device, mask_seed,
                                 mask_policy=mask_policy,
                                 defect_regions=batch.get("defect_regions"))
        # masked token 替换 (mask_token 注入被掩位置)
        mt = tokens * (~mask)[..., None].float() + self.mask_token * mask[..., None].float()
        enc = self.encoder(mt, valid_mask=tv, grid=grid)      # (B, N+1, d)
        pooled = enc[:, 0]
        dec = self.decoder_embed(enc[:, 1:])
        dec_out = self.decoder(dec, valid_mask=tv, grid=grid)  # (B, N+1, d_dec)
        head = self.pred_1d if batch["shape_kind"] == "1d" else self.pred_2d
        pred = head(dec_out[:, 1:])                            # (B, N, patch_dim)
        target_np = patchify_target(batch, self.patch_len, self.patch2d)
        target = torch.from_numpy(target_np).to(batch["x"].device)
        loss = masked_recon_loss(pred, target, mask, tv)
        return {
            "loss": loss,
            "pooled": pooled,
            "pred": pred,
            "target": target,
            "mask": mask,
            "valid": tv,
            "grid": grid,
        }

    def encode_raw(self, batch: dict) -> torch.Tensor:
        """全视图 (unmasked) 编码 → CLS pooled 表征 (B, d)。供冻结线性探针。"""
        tokens, grid, tv, _ = self._tokens_and_valid(batch, batch["x"].device)
        h = self.encoder(tokens, valid_mask=tv, grid=grid)
        return h[:, 0]
