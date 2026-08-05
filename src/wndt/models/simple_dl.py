"""Simple deep baselines: MLP / LSTM / GRU on raw (normalized) cycles."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Flatten 2x200 -> hidden layers -> logits."""

    def __init__(self, seq_len: int = 200, n_vars: int = 2,
                 hidden_sizes: tuple[int, ...] = (256, 128),
                 dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = seq_len * n_vars
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNClassifier(nn.Module):
    """LSTM or GRU over per-step (V_t, I_t) pairs; head on last hidden state."""

    def __init__(self, kind: str = "lstm", n_vars: int = 2, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.1, n_classes: int = 2):
        super().__init__()
        assert kind in ("lstm", "gru")
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(input_size=n_vars, hidden_size=hidden, num_layers=n_layers,
                       batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 200) -> (B, 200, 2)
        out, _ = self.rnn(x.transpose(1, 2))
        return self.head(out[:, -1])
