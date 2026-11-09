"""Mean-pooled linear MIL with an L1 penalty on the window weights."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .attention_mil import AttentionMILModel


class _MeanPoolNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, bag: torch.Tensor):
        window = torch.sigmoid(self.score(bag).squeeze(-1))
        return window.mean(), window


class L1MILModel(AttentionMILModel):
    """p_j = sigmoid(w·x_j + b), p_bag = mean(p_j), loss = BCE + λ||w||_1.

    The window score is p_j. The bag label is the only training target.
    """

    def __init__(self, l1_lambda: float = 1e-5, **kwargs) -> None:
        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(**kwargs)
        self.l1_lambda = float(l1_lambda)
        self._bag_probability = True

    def _make_net(self, output_dim: int) -> nn.Module:
        return _MeanPoolNet(self._input_dim)

    def _classification_loss(self, outputs: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
        # 防止0或1的极端值(类似拉普拉斯平滑)
        prob = outputs.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
        return nn.functional.binary_cross_entropy(prob, y.reshape(-1).float())

    def _penalty(self) -> torch.Tensor:
        return self.l1_lambda * self._net.score.weight.abs().sum()

    def window_probabilities(self, bag: np.ndarray) -> np.ndarray:
        """Per-window sigmoid scores. Shape ``(n_windows,)``."""
        self._net.eval()
        with torch.no_grad():
            if len(bag) == 0:
                return np.zeros(0, dtype=np.float64)
            _, window = self._net(self._bag_tensor(bag))
            return window.detach().cpu().numpy().astype(np.float64, copy=False)

    def leave_one_out_scores(self, bag: np.ndarray) -> np.ndarray:
        """``s_j = p_bag - mean(p without j)``. One window has no defined score."""
        p = self.window_probabilities(bag)
        k = int(p.size)
        if k < 2:
            return np.full(k, np.nan, dtype=np.float64)
        p_bag = float(p.mean())
        held_out = (float(p.sum()) - p) / (k - 1)
        return p_bag - held_out
