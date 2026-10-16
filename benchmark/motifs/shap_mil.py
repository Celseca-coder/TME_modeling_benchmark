"""Monte Carlo Shapley instance importance for Attention MIL bags."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from benchmark.models.attention_mil import AttentionMILModel


def _encode_bag(model: AttentionMILModel, bag: np.ndarray) -> tuple[torch.Tensor, int]:
    """Encode each window once. The last row is the scaled zero vector used for an empty coalition."""
    raw = np.asarray(bag, dtype=np.float32)
    raw = np.where(np.isnan(raw), model._median[None, :], raw)
    scaled = (raw - model._mean[None, :]) / model._std[None, :]
    zero = (np.zeros((1, raw.shape[1]), dtype=np.float32) - model._mean[None, :]) / model._std[None, :]
    x = np.concatenate([scaled, zero], axis=0)
    t = torch.as_tensor(x, dtype=torch.float32, device=model.device_)
    with torch.inference_mode():
        h = model._net.encoder(t)
    return h, int(len(raw))


def _batched_proba(
    model: AttentionMILModel,
    h: torch.Tensor,
    coalitions: list[np.ndarray],
    chunk: int = 512,
) -> np.ndarray:
    """Positive-class probability for many coalitions. Padded rows are masked out of the softmax."""
    net = model._net
    out = np.empty(len(coalitions), dtype=np.float64)
    pos = model._positive_index()
    with torch.inference_mode():
        for start in range(0, len(coalitions), chunk):
            block = coalitions[start:start + chunk]
            width = max(len(c) for c in block)
            idx = np.zeros((len(block), width), dtype=np.int64)
            mask = np.zeros((len(block), width), dtype=bool)
            for i, c in enumerate(block):
                idx[i, :len(c)] = c
                mask[i, :len(c)] = True
            idx_t = torch.as_tensor(idx, device=h.device)
            mask_t = torch.as_tensor(mask, device=h.device)
            gathered = h[idx_t]
            logits = net.attn_w(
                torch.tanh(net.attn_v(gathered)) * torch.sigmoid(net.attn_u(gathered))
            ).squeeze(-1)
            logits = logits.masked_fill(~mask_t, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1) * mask_t
            pooled = torch.sum(weights.unsqueeze(-1) * gathered, dim=1)
            prob = torch.softmax(net.head(pooled), dim=-1)[:, pos]
            out[start:start + len(block)] = prob.detach().cpu().numpy()
    return out


def _shap_sequential(
    model: AttentionMILModel,
    bag: np.ndarray,
    players: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Original one-coalition forward. Used when max_instances subsamples bags."""
    n = int(len(bag))
    phi = np.zeros(n, dtype=float)
    dim = bag.shape[1]
    empty = np.zeros((1, dim), dtype=np.float32)
    for w in players:
        others = [i for i in range(n) if i != int(w)]
        acc = 0.0
        for _ in range(n_samples):
            k = int(rng.integers(0, len(others) + 1)) if others else 0
            s = rng.choice(others, size=k, replace=False) if k else np.array([], dtype=int)
            bag_s = bag[s] if len(s) else empty
            bag_sw = bag[np.append(s, int(w))]
            acc += model.positive_proba(bag_sw) - model.positive_proba(bag_s)
        phi[int(w)] = acc / n_samples
    return phi


def shap_window_importance(
    model: AttentionMILModel,
    bag: np.ndarray,
    n_samples: int = 32,
    seed: int = 0,
    top_m: int | None = None,
    labeled: np.ndarray | None = None,
) -> np.ndarray:
    """Players = rows of bag. Returns phi, shape (n_windows,)."""
    rng = np.random.default_rng(seed)
    n = int(len(bag))
    phi = np.zeros(n, dtype=float)
    if n == 0:
        return phi
    players = np.arange(n, dtype=int)
    if labeled is not None:
        players = players[np.asarray(labeled, dtype=bool)]
    if top_m is not None and n > top_m:
        feat = pd.DataFrame({"bag": [bag]})
        attn = model.attention_weights(feat)[0]
        players = np.argsort(-attn)[:top_m]
    if model.max_instances:
        return _shap_sequential(model, bag, players, n_samples, rng)

    model._net.eval()
    h, empty_i = _encode_bag(model, bag)
    coalitions: list[np.ndarray] = []
    owners: list[int] = []
    for w in players:
        others = [i for i in range(n) if i != int(w)]
        for _ in range(n_samples):
            k = int(rng.integers(0, len(others) + 1)) if others else 0
            s = rng.choice(others, size=k, replace=False) if k else np.array([], dtype=int)
            if k == 0:
                coalitions.append(np.array([empty_i], dtype=np.int64))
                coalitions.append(np.array([int(w)], dtype=np.int64))
            else:
                coalitions.append(np.asarray(s, dtype=np.int64))
                coalitions.append(np.append(s, int(w)).astype(np.int64, copy=False))
            owners.append(int(w))
    if not coalitions:
        return phi
    prob = _batched_proba(model, h, coalitions)
    for i, w in enumerate(owners):
        phi[w] += prob[2 * i + 1] - prob[2 * i]
    for w in players:
        phi[int(w)] /= n_samples
    return phi
