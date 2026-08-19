"""多模态 NDT 模型接口草案（M0-1 统一数据底座，只设计接口，不训练）。

结构（自下而上）:
    ModalitySpecificStem  : 每模态一个专属 stem，把原生 tensor 编码到统一
                            embedding 维（超声 A/B/C-scan、涡流 I/Q 各自 stem）。
    NDTEncoder            : 聚合各模态 embedding 为序列/全局 token，接受
                            availability mask（支持缺失模态）。
    FusionHead            : early / intermediate / late 三类融合。
    TaskHead              : 分类 / 回归 / 分割等任务头。

约束:
1. 超声与涡流必须各自使用 modality-specific stem，不得共享卷积主干；
   它们只通过统一 embedding 维度对接。
2. 无配对 UT+ECT 数据时，只允许单模态训练与接口单元测试；监督融合头
   必须经过 ``PairedDataGuard``（src/wndt/data/adapters/base.py）校验，
   禁止用 unpaired 样本训练（见 ``ensure_paired_fusion``）。
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wndt.data.adapters.base import NDTBatch, PairedDataGuard, UnpairedDataError

FusionType = Literal["early", "intermediate", "late"]


class TaskKind(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"


class ModalitySpecificStem(nn.Module, ABC):
    """把某模态的原生 tensor 编码到统一 embedding 维度 ``out_dim``。

    每个模态一个实现：``UltrasonicStem`` 吃 (B, beam, time) 的 B-scan 或
    (B, time) 的 A-scan；``EddyCurrentStem`` 吃 (B, time, iq) 的 I/Q 曲线。
    输出形状约定为 ``(B, L, out_dim)``（序列 token），供 ``NDTEncoder``
    统一聚合；无序列维的模态可令 L=1。
    """

    modality_name: str

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, available: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, ...) 模态原生 tensor -> (B, L, out_dim)。

        ``available``: (B,) bool 可选，缺失样本可返回全零 embedding
        （由 encoder/fusion 的 availability mask 决定如何使用）。
        """
        raise NotImplementedError


class NDTEncoder(nn.Module, ABC):
    """聚合多模态 embedding 为统一的序列/全局 token。

    输入为各模态 stem 的输出字典 + (B, M) availability mask；输出
    ``(B, L_fused, d_model)`` 或 ``(B, d_model)``。实现自由（拼接、
    注意力 MIL、跨模态注意力均可），只要满足 availability mask 语义：
    缺失模态的 token 不得污染输出。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class FusionHead(nn.Module, ABC):
    """融合头。``fusion_type`` 决定融合发生的层级：

    - early:        在 stem 之前的原生 tensor 层融合（要求严格配准、同尺寸，
                    仅成对数据可用）；
    - intermediate: 在 encoder token 层融合（默认，可用 gating/注意力）；
    - late:         在各模态 TaskHead 分数层融合（score-level，可独立训练
                    单模态再融合）。
    """

    fusion_type: FusionType

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """modality_tokens: 模态名 -> (B, L, d_model) 或 (B, d_model) 统一后的 token。"""
        raise NotImplementedError


class TaskHead(nn.Module, ABC):
    """任务头：分类/回归/分割。单模态训练时直接接在 stem/encoder 之后。"""

    def __init__(self, in_dim: int, task: TaskKind = TaskKind.CLASSIFICATION):
        super().__init__()
        self.in_dim = in_dim
        self.task = task

    @abstractmethod
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 轻量参考实现：仅用于接口单元测试与单模态训练桩（M0-2 前不承载正式训练）。
# ---------------------------------------------------------------------------
class UltrasonicStem(ModalitySpecificStem):
    """参考超声 stem：B-scan (B, beam, time) -> flatten -> Linear -> (B, 1, D)。

    正式实现应是 2D/3D 卷积或 SSF 风格谱-空-频分支；此桩只验证接口形状。
    """

    modality_name = "ultrasonic"

    def __init__(self, in_channels: int, seq_len: int, out_dim: int):
        super().__init__(out_dim)
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.proj = nn.Linear(in_channels * seq_len, out_dim)

    def forward(self, x: torch.Tensor, available: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        emb = self.proj(x.reshape(b, -1)).unsqueeze(1)  # (B, 1, D)
        if available is not None:
            emb = emb * available.view(b, 1, 1)
        return emb


class EddyCurrentStem(ModalitySpecificStem):
    """参考涡流 stem：I/Q 曲线 (B, time, 2) -> 1D Conv + pool -> (B, 1, D)。"""

    modality_name = "eddy_current"

    def __init__(self, in_channels: int = 2, seq_len: int = 32, out_dim: int = 64):
        super().__init__(out_dim)
        self.in_channels = in_channels
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.proj = nn.Linear(16 * (seq_len // 2), out_dim)
        self.pool = nn.AdaptiveAvgPool1d(seq_len // 2)

    def forward(self, x: torch.Tensor, available: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, time, 2) -> (B, 2, time)
        b = x.shape[0]
        h = F.relu(self.conv(x.transpose(1, 2)))
        h = self.pool(h)
        emb = self.proj(h.reshape(b, -1)).unsqueeze(1)  # (B, 1, D)
        if available is not None:
            emb = emb * available.view(b, 1, 1)
        return emb


class MeanPoolEncoder(NDTEncoder):
    """参考 encoder：各模态 token 均值池化后拼接（intermediate 层），
    缺失模态以 0 填充并按其 availability 归一化。"""

    def __init__(self, d_model: int, modality_order: Sequence[str]):
        super().__init__(d_model)
        self.modality_order = list(modality_order)
        self.proj = nn.Linear(d_model * len(modality_order), d_model)

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        for j, mod in enumerate(self.modality_order):
            e = embeddings.get(mod)
            if e is None:
                parts.append(torch.zeros(availability.shape[0], self.d_model,
                                         device=availability.device))
                continue
            pooled = e.mean(dim=1)                      # (B, D)
            avail = availability[:, j].unsqueeze(1)
            pooled = pooled * avail                     # 缺失置零
            parts.append(pooled)
        fused = torch.cat(parts, dim=-1)
        return self.proj(fused)                          # (B, d_model)


class ConcatFusionHead(FusionHead):
    """intermediate 融合：encoder 输出按 **modality_order** 拼接后过 MLP。

    M0-1.5: 显式接收 ``modality_order``，禁止依赖 ``dict.values()`` 顺序
    （dict 顺序与 availability 列不对齐会静默错配模态）。
    """

    fusion_type: FusionType = "intermediate"

    def __init__(self, d_model: int, modality_order: Sequence[str]):
        super().__init__(d_model)
        self.modality_order = list(modality_order)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * len(modality_order), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        for j, mod in enumerate(self.modality_order):
            tok = modality_tokens.get(mod)
            if tok is None:
                parts.append(torch.zeros(availability.shape[0], self.d_model,
                                         device=availability.device))
                continue
            t = tok.mean(dim=1) if tok.dim() == 3 else tok      # (B, D)
            t = t * availability[:, j].unsqueeze(1)
            parts.append(t)
        return self.mlp(torch.cat(parts, dim=-1))


class ScoreFusionHead(FusionHead):
    """late 融合：各模态任务分数加权和。

    M0-1.5 修复:
      - 显式 ``modality_order``；
      - 权重为 **B×M** 可学习（每样本每模态一个权重）；
      - 按 availability mask 逐样本屏蔽缺失模态权重，并在**可用模态**上
        重新归一化 —— 缺失模态不得拉低/污染输出。
    """

    fusion_type: FusionType = "late"

    def __init__(self, d_model: int, modality_order: Sequence[str], n_classes: int):
        super().__init__(d_model)
        self.modality_order = list(modality_order)
        m = len(modality_order)
        self.logits = nn.ModuleList([nn.Linear(d_model, n_classes) for _ in range(m)])
        self.w = nn.Parameter(torch.ones(m))       # 全局先验, 展开为 B×M

    def forward(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        b = availability.shape[0]
        # (B, M) 权重: softmax 在全局先验上 → 按 availability 屏蔽 → 可用模态重归一化
        w = torch.softmax(self.w, dim=0).unsqueeze(0).expand(b, -1)      # (B, M)
        w = w * availability.float()                                     # 屏蔽缺失
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w = w / denom                                                    # 重归一化
        out = torch.zeros(b, self.logits[0].out_features,
                          device=availability.device)
        for j, mod in enumerate(self.modality_order):
            tok = modality_tokens.get(mod)
            if tok is None:
                continue
            t = tok.mean(dim=1) if tok.dim() == 3 else tok
            out = out + w[:, j:j + 1] * self.logits[j](t)
        return out


class GatedFusionHead(FusionHead):
    """intermediate 门控融合：学习逐模态门控权重。

    M0-1.5 修复:
      - 显式 ``modality_order``；
      - 缺失模态的 token 在进入任何 MLP（gate 或 fusion MLP）**之前必须置零**，
        否则缺失模态的随机占位值会通过 gate/MLP 污染输出。
    """

    fusion_type: FusionType = "intermediate"

    def __init__(self, d_model: int, modality_order: Sequence[str]):
        super().__init__(d_model)
        self.modality_order = list(modality_order)
        m = len(modality_order)
        self.gate = nn.Linear(d_model * m, m)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * (m + 1), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        availability: torch.Tensor,
    ) -> torch.Tensor:
        stacked = []
        for j, mod in enumerate(self.modality_order):
            tok = modality_tokens.get(mod)
            if tok is None:
                # 缺失模态: 零 token, 且该列 availability=0 → 不参与任何 MLP
                t = torch.zeros(availability.shape[0], self.d_model,
                                device=availability.device)
            else:
                t = tok.mean(dim=1) if tok.dim() == 3 else tok
                # 关键: 缺失样本的 token 进入 MLP 前置零 (占位随机值必须清掉)
                t = t * availability[:, j:j + 1]
            stacked.append(t)
        x = torch.cat(stacked, dim=-1)
        g = torch.sigmoid(self.gate(x)) * availability.float()     # (B, M)，缺失=0
        g = g / g.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted = sum(gi.unsqueeze(1) * ti for gi, ti in zip(g.unbind(1), stacked))
        return self.mlp(torch.cat([weighted, x], dim=-1))


class MLPTaskHead(TaskHead):
    """参考任务头：分类 logits / 回归标量。"""

    def __init__(self, in_dim: int, n_classes: int = 2, task: TaskKind = TaskKind.CLASSIFICATION):
        super().__init__(in_dim, task)
        self.head = nn.Linear(in_dim, 1 if task == TaskKind.REGRESSION else n_classes)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.head(fused)


# ---------------------------------------------------------------------------
# 融合训练守卫：没有成对数据就不允许训练监督融合头。
# ---------------------------------------------------------------------------
def ensure_paired_fusion(
    batch: NDTBatch,
    guard: PairedDataGuard,
    modality_a: str = "ultrasonic",
    modality_b: str = "eddy_current",
    instances: Sequence["object"] | None = None,
    fusion_type: str = "intermediate",
) -> None:
    """训练/验证融合头前的强制校验。unpaired 样本触发 ``UnpairedDataError``。

    语义：同 specimen、同坐标、已配准的 UT+ECT 成对样本才允许进入监督
    融合训练；任何单模态缺失、无融合链接、或跨 specimen 拼凑的样本都会
    被拦截。``instances`` 传入构建本批的 ``NDTInstance`` 时执行逐实例校验。
    ``fusion_type``: "early" 额外要求严格坐标矩阵（见 ``PairedDataGuard``）。
    """
    if not (batch.has(modality_a) and batch.has(modality_b)):
        raise UnpairedDataError(
            f"cannot train fusion head: batch has modalities {batch.modalities}")
    guard.require_paired(batch, modality_a, modality_b, instances=instances,
                         fusion_type=fusion_type)


def build_ultrasonic_only_pipeline(
    d_model: int = 64,
    in_channels: int = 4,
    seq_len: int = 32,
    n_classes: int = 2,
) -> nn.Module:
    """**单模态**超声训练桩（无配对数据时的唯一合法训练通路）。

    返回一个可端到端 forward 的 ``nn.Module``：UltrasonicStem -> 单模态
    MeanPoolEncoder -> MLPTaskHead。与融合完全无关，用于 M0-2 之前的
    单模态基线冒烟与接口单测。
    """
    stem = UltrasonicStem(in_channels=in_channels, seq_len=seq_len, out_dim=d_model)
    enc = MeanPoolEncoder(d_model=d_model, modality_order=["ultrasonic"])
    head = MLPTaskHead(in_dim=d_model, n_classes=n_classes)

    class _Pipeline(nn.Module):
        def __init__(self, s, e, h):
            super().__init__()
            self.stem, self.encoder, self.head = s, e, h

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avail = torch.ones(x.shape[0], 1, device=x.device)
            emb = {"ultrasonic": self.stem(x, available=avail[:, 0])}
            return self.head(self.encoder(emb, avail))

    return _Pipeline(stem, enc, head)
