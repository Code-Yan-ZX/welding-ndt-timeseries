"""WRT-SAM 复现模型 (arXiv:2502.11338)。

结构 = SAM-Adapter (Chen et al., ICCVW 2023; baseline) + FPG + MSPG:
  - 冻结 SAM ViT image encoder，仅保留其内部 PromptGenerator (即 baseline 的 adapter) 可训练
  - FPG  (Frequency Prompt Generator): FcaNet 式 patch 级 2D DCT -> fc -> Conv2d, 论文式(1)-(4)
  - MSPG (Multi-Scale Prompt Generator): SegNeXt MSCA 式多尺度条带卷积注意力, 论文式(5)
  - P_f + P_ms + no_mask_embed 相加后作为 dense prompt 注入 (可冻结的) Mask Decoder
超参与 baseline 仓库默认一致: AdamW lr 2e-4, cosine -> 1e-7, 20 epochs, loss=iou(BCE+IoU)。
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

VIT_B = dict(
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    global_attn_indexes=[2, 5, 8, 11], window_size=14, patch_size=16,
    out_chans=256, prompt_embed_dim=256, use_rel_pos=True, qkv_bias=True,
)


class FPG(nn.Module):
    """Frequency Prompt Generator.

    论文式(1)-(4): P_f = Conv2d(fc(MSCDCT(X)))。
    MSCDCT: 将输入图按 feat_size 分 patch，每 patch 施加选定 (u,v) 分量的 2D DCT 基，
    得到 (B, n, feat, feat) 频域图；top1 模式即 [u,v]=(0,0) 单分量。随后 fc 升维 + Conv2d
    生成 (B, out_dim, feat, feat) 的频率 prompt。
    """

    def __init__(self, inp_size: int = 1024, feat_size: int = 64, out_dim: int = 256,
                 uv: Tuple[int, int] = (0, 0), zero_init: bool = True):
        super().__init__()
        self.inp_size = inp_size
        self.feat_size = feat_size
        h = torch.arange(inp_size // feat_size).float()
        w = torch.arange(inp_size // feat_size).float()
        basis_h = torch.cos(torch.pi * h / (inp_size // feat_size) * (uv[0] + 0.5))
        basis_w = torch.cos(torch.pi * w / (inp_size // feat_size) * (uv[1] + 0.5))
        self.register_buffer("basis", basis_h[:, None] * basis_w[None, :])  # (p, p)
        hidden = 16
        self.fc = nn.Linear(1, hidden)
        self.conv = nn.Conv2d(hidden, out_dim, 3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.inp_size // self.feat_size
        patches = x.reshape(B * C, self.feat_size, p, self.feat_size, p)
        patches = patches.permute(0, 1, 3, 2, 4)  # (B*C, feat, feat, p, p)
        coeff = (patches * self.basis).mean(dim=(-2, -1))  # (B*C, feat, feat)
        coeff = coeff.reshape(B, C, self.feat_size, self.feat_size).mean(dim=1, keepdim=True)
        h = self.fc(coeff.permute(0, 2, 3, 1))  # (B, feat, feat, hidden)
        h = h.permute(0, 3, 1, 2)
        return self.conv(h)


class MSCA(nn.Module):
    """Multi-Scale Convolutional Attention (SegNeXt) 用于 MSPG。

    论文式(5): P_ms = (Conv1x1(sum_i Scale_i(Dconv(X)))) ⊗ X,
    Scale_i 为两条深度条带卷积分支, kernel = 7, 11, 21。
    """

    def __init__(self, dim: int = 256, kernels: Tuple[int, ...] = (7, 11, 21)):
        super().__init__()
        self.dconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim),
                nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim),
                nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim),
                nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim),
            ))
        self.conv1x1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.dconv(x)
        for br in self.branches:
            attn = attn + br(attn)
        attn = self.conv1x1(attn)
        return x * attn


class MSPG(nn.Module):
    """Multi-Scale Prompt Generator: 输入图 -> conv stem 下采样到 (out_dim, feat, feat) -> MSCA."""

    def __init__(self, inp_size: int = 1024, feat_size: int = 64, out_dim: int = 256,
                 zero_init: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(3, out_dim, kernel_size=inp_size // feat_size,
                              stride=inp_size // feat_size)
        self.msca = MSCA(out_dim)
        if zero_init:
            # MSCA 输出 = x * attn, 零初始化 1x1 卷积使初始 attn 恒 0 (起步等价于恒等/无贡献)
            nn.init.zeros_(self.msca.conv1x1.weight)
            nn.init.zeros_(self.msca.conv1x1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.msca(self.stem(x))


class WRTSAM(nn.Module):
    def __init__(self, inp_size: int = 1024, vit_mode: str = "vit_b",
                 use_fpg: bool = True, use_mspg: bool = True,
                 freeze_decoder: bool = True, loss: str = "iou",
                 fpg_uv: Tuple[int, int] = (0, 0)):
        super().__init__()
        mode = dict(VIT_B) if vit_mode == "vit_b" else None
        assert mode is not None, f"unsupported vit_mode {vit_mode}"
        self.inp_size = inp_size
        self.embed_dim = mode["embed_dim"]
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=mode["patch_size"],
            in_chans=3,
            embed_dim=mode["embed_dim"],
            depth=mode["depth"],
            num_heads=mode["num_heads"],
            mlp_ratio=mode["mlp_ratio"],
            out_chans=mode["out_chans"],
            qkv_bias=mode["qkv_bias"],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=mode["use_rel_pos"],
            rel_pos_zero_init=True,
            window_size=mode["window_size"],
            global_attn_indexes=mode["global_attn_indexes"],
        )
        self.prompt_embed_dim = mode["prompt_embed_dim"]
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        )
        self.pe_layer = _PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.image_embedding_size = inp_size // mode["patch_size"]
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

        feat_size = self.image_embedding_size
        self.use_fpg = use_fpg
        self.use_mspg = use_mspg
        if use_fpg:
            self.fpg = FPG(inp_size, feat_size, self.prompt_embed_dim, uv=fpg_uv)
        if use_mspg:
            self.mspg = MSPG(inp_size, feat_size, self.prompt_embed_dim)

        assert loss in ("iou", "iou_only", "bce")
        self.loss_mode = loss

        # 冻结: baseline 协议 —— encoder 仅留 prompt_generator (adapter);
        # 论文称 Mask Decoder 冻结 (baseline 仓库默认不冻结), 用 freeze_decoder 控制。
        self.freeze_decoder = freeze_decoder
        self.set_requires_grad(self.image_encoder, False)
        self.set_requires_grad(self.image_encoder.prompt_generator, True)
        if freeze_decoder:
            self.set_requires_grad(self.mask_decoder, False)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def set_requires_grad(net: nn.Module, requires_grad: bool) -> None:
        for p in net.parameters():
            p.requires_grad_(requires_grad)

    def load_sam_checkpoint(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("prompt_encoder.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # prompt_encoder 未使用 (sparse prompt 为空); 允许缺失的仅限新初始化的可训练模块
        allowed = ("image_encoder.prompt_generator.", "fpg.", "mspg.",
                   "no_mask_embed.", "pe_layer.")
        bad_missing = [k for k in missing if not k.startswith(allowed)]
        assert not unexpected and not bad_missing, \
            f"unexpected={unexpected[:5]} bad_missing={bad_missing[:5]}"

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # ---------------------------------------------------------------- forward
    def _dense_prompts(self, x: torch.Tensor, bs: int) -> torch.Tensor:
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size)
        extra = None
        if self.use_fpg:
            extra = self.fpg(x)
        if self.use_mspg:
            p = self.mspg(x)
            extra = p if extra is None else extra + p
        if extra is not None:
            dense = dense + extra
        return dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        features = self.image_encoder(x)
        dense = self._dense_prompts(x, bs)
        sparse = torch.empty((bs, 0, self.prompt_embed_dim), device=x.device)
        low_res, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        masks = F.interpolate(low_res, (self.inp_size, self.inp_size),
                              mode="bilinear", align_corners=False)
        return masks  # logits

    # ------------------------------------------------------------------- loss
    def compute_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "bce":
            return F.binary_cross_entropy_with_logits(pred_logits, gt)
        iou = _soft_iou_loss(torch.sigmoid(pred_logits), gt)
        if self.loss_mode == "iou_only":
            return iou
        return F.binary_cross_entropy_with_logits(pred_logits, gt) + iou


class _PositionEmbeddingRandom(nn.Module):
    """与 baseline 仓库一致的随机空间频率位置编码."""

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


def _soft_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    return (1 - inter / union.clamp(min=1e-6)).mean()
