"""Additive attention MIL.

Attention is the plain (non-gated) Ilse form. The embedding is the window
vector itself, so the trained parameters are V, w, W, and b.
"""
from __future__ import annotations

import torch
from torch import nn

from .attention_mil import AttentionMILModel


class _AdditiveAttentionNet(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.attn_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.attn_w = nn.Linear(attention_dim, 1, bias=False)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, bag: torch.Tensor):
        logits = self.attn_w(torch.tanh(self.attn_v(bag))).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        pooled = torch.sum(weights[:, None] * bag, dim=0)
        return self.head(pooled), weights


class AdditiveMILModel(AttentionMILModel):
    """Bag probability is sigmoid(Wz + b). Window score is the attention weight."""

    def __init__(self, l1_lambda: float = 1e-5, **kwargs) -> None:
        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(**kwargs)
        self.l1_lambda = float(l1_lambda)

    # 建立注意力矩阵V,打分向量omega,袋分类层W和b
    def _make_net(self, output_dim: int) -> nn.Module:
        return _AdditiveAttentionNet(self._input_dim, self.attention_dim)

    def _penalty(self) -> torch.Tensor:
        net = self._net
        penalty = (
            net.attn_v.weight.abs().sum()
            + net.attn_w.weight.abs().sum()
            + net.head.weight.abs().sum()
        )
        return self.l1_lambda * penalty
