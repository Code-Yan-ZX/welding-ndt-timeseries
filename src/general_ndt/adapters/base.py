"""模态适配器: stem 分派 + metadata 嵌入。

把不同模态/通道数/时长的信号统一到共享 token 空间 (d_model)。
- stem 分派: 1d → Stem1D (C, n_col); 2d → Stem2D (C, n_h, n_w);
  **位置编码不在 adapter 内添加**, 由 PatchTransformer 按 grid 统一添加 (可审计)。
- metadata (modality / sensor) 以加法嵌入注入, 使骨干对"模态、传感器"可见而不改维度。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from general_ndt.adapters.stems import Stem1D, Stem2D, StemTF


class ModalAdapter(nn.Module):
    """根据样本 (signal, shape_kind, modality) 产出 token 网格。

    输入: batch 的 padded_signal
      - 1d: (B, C_max, L_max)
      - 2d: (B, C_max, H_max, W_max)
    输出: (tokens, grid)  tokens: (B, N, d); grid: (C, n_col) 或 (C, n_h, n_w)
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_len: int = 16,     # 1D 时间 patch
        patch2d: int = 16,       # 2D / 时频 patch
        n_modalities: int = 8,
        n_sensors: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.patch2d = patch2d
        # Stem1D 为 per-channel 共享 conv (通道数无关), 无需按通道缓存
        self.stem1d = Stem1D(in_channels=1, patch_len=patch_len, d_model=d_model)
        self.stem2d = Stem2D(patch=patch2d, d_model=d_model)
        self.stem_tf = StemTF(patch=patch2d, d_model=d_model)
        self.modality_embed = nn.Embedding(n_modalities, d_model)
        self.sensor_embed = nn.Embedding(n_sensors, d_model)

    def _modality_id(self, modality: str) -> int:
        order = ["ultrasonic", "guided_wave", "eddy_current", "acoustic_emission",
                 "vibration", "process", "fusion", "unknown"]
        return order.index(modality) if modality in order else len(order) - 1

    def forward(
        self,
        x: torch.Tensor,
        shape_kind: str,
        modalities: list[str],
        sampling_rates: list[float] | None = None,
        sensor_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        B = x.shape[0]
        if shape_kind == "1d":
            z, grid = self.stem1d(x)                       # (B, C*n_col, d), (C, n_col)
        elif shape_kind == "2d":
            if x.ndim == 3:                                # (B, H, W) → (B, 1, H, W)
                x = x.unsqueeze(1)
            z, grid = self.stem2d(x)                       # (B, C*nh*nw, d), (C, nh, nw)
        else:
            raise ValueError(f"未知 shape_kind: {shape_kind}")

        N = z.shape[1]
        # modality / sensor 嵌入 (逐 token 加法)
        mod_ids = torch.as_tensor(
            [self._modality_id(m) for m in modalities], device=x.device
        ).view(B, 1)
        mod = self.modality_embed(mod_ids).expand(B, N, -1)
        sens = torch.zeros_like(mod)
        if sensor_ids is not None:
            for b, sid in enumerate(sensor_ids):
                if sid:
                    sens[b] = self.sensor_embed(torch.tensor(hash(sid) % self.sensor_embed.num_embeddings, device=x.device))

        return z + mod + sens, grid


def sample_to_batch(samples: list[object], device: str = "cpu") -> dict:
    """把样本列表转成 adapter 所需的批字典 (供 smoke/测试用)。"""
    from general_ndt.datasets.collate import collate_general_ndt

    batch = collate_general_ndt(samples)
    x = torch.from_numpy(batch.padded_signal).float().to(device)
    return {
        "x": x,
        "shape_kind": batch.shape_kind,
        "modalities": batch.modalities,
        "sampling_rates": [s.metadata.get("sampling_rate") for s in samples],
        "sensor_ids": [None] * len(samples),
        "valid_mask": torch.from_numpy(batch.valid_mask).to(device),
        "shapes": batch.shapes,
    }
