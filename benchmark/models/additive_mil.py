"""Additive gated-attention MIL.

The bag logit is the sum of per-window contributions plus one bag-level bias.
The bias stays outside the sum so it is not repeated once per window.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .attention_mil import AttentionMILModel


class _AdditiveGatedNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int,
                 output_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attn_v = nn.Linear(hidden_dim, attention_dim)
        self.attn_u = nn.Linear(hidden_dim, attention_dim)
        self.attn_w = nn.Linear(attention_dim, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, bag: torch.Tensor):
        h = self.encoder(bag)
        attn_logits = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(1)
        weights = torch.softmax(attn_logits, dim=0)
        contrib = self.psi(weights[:, None] * h)
        return contrib.sum(dim=0) + self.bias, contrib


class AdditiveMILModel(AttentionMILModel):
    """Gated attention, then a bias-free head on each weighted window.

    ``window_contributions`` is the positive-class column of those heads.
    Summing it and adding the bag bias recovers the positive-class logit.
    """

    def _make_net(self, output_dim: int) -> nn.Module:
        return _AdditiveGatedNet(
            self._input_dim, self.hidden_dim, self.attention_dim, output_dim, self.dropout
        )

    def window_contributions(self, bag: np.ndarray) -> np.ndarray:
        """Signed positive-class contribution of each window. Shape ``(n_windows,)``."""
        self._net.eval()
        with torch.no_grad():
            _, contrib = self._net(self._bag_tensor(bag))
            return contrib[:, self._positive_index()].detach().cpu().numpy().astype(np.float64, copy=False)

    def positive_bias(self) -> float:
        self._net.eval()
        return float(self._net.bias.detach()[self._positive_index()].cpu())
