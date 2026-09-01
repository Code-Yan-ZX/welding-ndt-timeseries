"""模态适配器: stem 分派 + metadata 嵌入 + 采样率感知位置编码。

把不同模态/通道数/采样率/时长的信号统一到共享 token 空间 (d_model)。
metadata (modality / sensor / sampling_rate / spatial) 以加法嵌入注入,
使骨干对"模态、传感器、时间物理尺度"可见而不改变 token 维度。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from general_ndt.adapters.stems import Stem1D, Stem2D, StemTF
from general_ndt.datasets.schema import GeneralNDTSample


class ModalAdapter(nn.Module):
    """根据样本 (signal, shape_kind, modality, sampling_rate) 产出 token 网格。

    输入: 一个 GeneralNDTBatch 的 padded_signal (B, C_max, L_max) 或 (B, H_max, W_max)
    输出: (tokens, grid)  tokens: (B, n_row*n_col, d); grid: (n_row, n_col) 最大网格
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_len: int = 16,     # 1D 时间 patch
        patch2d: int = 16,       # 2D / 时频 patch
        n_modalities: int = 8,
        n_sensors: int = 32,
        max_len: int = 4096,     # 位置编码上限
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.patch2d = patch2d
        # 1D stem 的 conv 按通道数动态构建 (见 _stem1d_for_channels)
        self._stem1d_cache: dict[int, Stem1D] = {}
        self.stem2d = Stem2D(patch=patch2d, d_model=d_model)
        self.stem_tf = StemTF(patch=patch2d, d_model=d_model)
        self.modality_embed = nn.Embedding(n_modalities, d_model)
        self.sensor_embed = nn.Embedding(n_sensors, d_model)
        # 采样率感知位置: 连续位置 = t*sampling_rate 的三角函数嵌入 (替代固定整数位置)
        self._build_pe(max_len)

    def _build_pe(self, max_len: int) -> None:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        t = torch.arange(max_len).float()
        pos = t[:, None] * inv_freq[None, :]
        pe = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # (max_len, d)
        self.register_buffer("pe", pe, persistent=False)

    def _stem1d_for_channels(self, c: int) -> Stem1D:
        if c not in self._stem1d_cache:
            self._stem1d_cache[c] = Stem1D(
                in_channels=c, patch_len=self.patch_len, d_model=self.d_model
            ).to(next(self.parameters()).device)
        return self._stem1d_cache[c]

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
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        B = x.shape[0]
        if shape_kind == "1d":
            z, grid = self._stem1d_for_channels(x.shape[1])(x)
        elif shape_kind == "2d":
            z, grid = self.stem2d(x.unsqueeze(1))
        else:
            raise ValueError(f"未知 shape_kind: {shape_kind}")

        n_row, n_col = grid
        N = n_row * n_col
        # 位置嵌入 (采样率感知): token 在网格中的展平索引, 按采样率缩放
        pos_idx = torch.arange(N, device=x.device).float()
        sr = torch.as_tensor(
            [s or 1.0 for s in (sampling_rates or [1.0] * B)], device=x.device
        ).view(B, 1)
        scale = torch.clamp(sr, 1e-3, 1e3).float()
        # 第一版用整数位置嵌入近似 (采样式连续位置编码列为扩展)
        pe = self.pe[: min(N, self.pe.shape[0])].unsqueeze(0).expand(B, -1, -1)
        if N > pe.shape[1]:  # 超上限截断并 pad
            pe = torch.cat([pe, torch.zeros(B, N - pe.shape[1], self.d_model, device=x.device)], dim=1)

        # modality / sensor 嵌入 (逐 token 加法)
        mod_ids = torch.as_tensor(
            [self._modality_id(m) for m in modalities], device=x.device
        ).view(B, 1)
        mod = self.modality_embed(mod_ids).expand(B, N, -1)
        sens = torch.zeros_like(mod)
        if sensor_ids is not None:
            for b, sid in enumerate(sensor_ids):
                if sid:
                    sens[b] = self.sensor_embed(torch.tensor(hash(sid) % 32, device=x.device))

        return z + pe + mod + sens, grid

    def position_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """(连续) 采样率感知位置嵌入 — 第一版用整数位置近似, 预留扩展。"""
        return self.pe[min(int(idx.item()), self.pe.shape[0] - 1)]


def sample_to_batch(samples: list[GeneralNDTSample], device: str = "cpu") -> dict:
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
    }
