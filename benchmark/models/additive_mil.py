"""Additive attention MIL.

Attention is the plain (non-gated) Ilse form. The embedding is the window
vector itself, so the trained parameters are V, w, W, and b.
"""
from __future__ import annotations

import torch
from torch import nn

from .attention_mil import AttentionMILModel


class _AdditiveAttentionNet(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.attn_w = nn.Linear(attention_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, bag: torch.Tensor):
        logits = self.attn_w(torch.tanh(self.attn_v(bag))).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        pooled = torch.sum(weights[:, None] * bag, dim=0)
        return self.head(self.drop(pooled)), weights

    def leave_one_out(self, bag: torch.Tensor) -> torch.Tensor:
        """``p_bag - p_bag,-j``. The embedding is the window vector. Dropout is off."""
        from .attention_mil import _attention_leave_one_out

        energy = self.attn_w(torch.tanh(self.attn_v(bag))).squeeze(-1)
        return _attention_leave_one_out(bag, energy, self.head)


class AdditiveMILModel(AttentionMILModel):
    """Bag probability is sigmoid(Wz + b). The loss is BCE, with no L1 term.

    Dropout 0.1 is applied to the pooled vector while training. Window scores
    and leave-one-out probabilities are computed with dropout disabled.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("weight_decay", 0.0)
        kwargs.setdefault("dropout", 0.1)
        super().__init__(**kwargs)

    def _make_net(self, output_dim: int) -> nn.Module:
        return _AdditiveAttentionNet(self._input_dim, self.attention_dim, self.dropout)

    def _penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.device_)
