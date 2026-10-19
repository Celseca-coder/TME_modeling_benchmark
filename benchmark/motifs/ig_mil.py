"""Integrated Gradients window importance for gated-attention MIL bags."""
from __future__ import annotations

import numpy as np
import torch

from benchmark.models.attention_mil import AttentionMILModel


def ig_window_importance(
    model: AttentionMILModel,
    bag: np.ndarray,
    steps: int = 32,
) -> np.ndarray:
    """Signed IG of the positive-class logit. Shape ``(n_windows,)``.

    The bag is scaled the same way as ``AttentionMILModel._bag_tensor``.
    The baseline is the zero vector in that scaled space (the training-set
    mean). Each step runs the full gated-attention forward, so the softmax
    weights move with the interpolation. ``model.max_instances`` must be
    unset, otherwise the returned scores would not line up with every window.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    n = int(len(bag))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if model.max_instances and n > int(model.max_instances):
        raise ValueError(
            "ig_window_importance requires max_instances=None so every window is scored"
        )
    model._net.eval()
    x = model._bag_tensor(bag).detach()
    pos = model._positive_index()
    acc = torch.zeros_like(x)
    for k in range(1, steps + 1):
        inp = ((k / steps) * x).detach().requires_grad_(True)
        logit = model._net(inp)[0][pos]
        model._net.zero_grad(set_to_none=True)
        logit.backward()
        grad = inp.grad
        if grad is not None:
            acc = acc + grad.detach()
    scores = (x * acc / steps).sum(dim=1)
    return scores.detach().cpu().numpy().astype(np.float64, copy=False)


def ig_completeness_error(
    model: AttentionMILModel,
    bag: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Relative gap ``|sum(s) - (F(x) - F(0))| / (|F(x) - F(0)| + 1e-6)``."""
    if int(len(bag)) == 0:
        return 0.0
    model._net.eval()
    x = model._bag_tensor(bag).detach()
    pos = model._positive_index()
    with torch.inference_mode():
        fx = float(model._net(x)[0][pos].detach().cpu())
        f0 = float(model._net(torch.zeros_like(x))[0][pos].detach().cpu())
    gap = fx - f0
    return abs(float(np.sum(scores)) - gap) / (abs(gap) + 1e-6)
