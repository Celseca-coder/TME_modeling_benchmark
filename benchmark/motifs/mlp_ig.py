"""Window MLP with Integrated Gradients.

Each labeled window is one sample. Signed IG summed over features is the
window score. Mean absolute IG across windows in a fold is the global
feature importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from benchmark.motifs.recovery import MIL_FEATURE_RULES, rank_recovery


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowMLP:
    """Binary MLP on a window feature table. Target is y_w."""

    def __init__(
        self,
        seed: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        patience: int = 5,
        batch_size: int = 2048,
        device: str = "auto",
    ) -> None:
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device_name = device

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "WindowMLP":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        y_raw = target.loc[features.index].to_numpy(float)
        classes = np.asarray(sorted(pd.unique(y_raw).tolist()))
        if len(classes) < 2:
            raise ValueError("Window MLP requires two classes")
        self.classes_ = classes
        self._class_to_i = {c: i for i, c in enumerate(classes)}
        self.device_ = self._device()

        raw = features.to_numpy(np.float64)
        self._columns = list(features.columns)
        self._median = np.nanmedian(raw, axis=0)
        self._median = np.nan_to_num(self._median, nan=0.0)
        filled = np.where(np.isnan(raw), self._median, raw)
        self._mean = filled.mean(axis=0)
        self._std = filled.std(axis=0)
        keep = self._std >= 1e-8
        if not np.any(keep):
            raise ValueError("no non-constant features")
        self._keep = keep
        self._columns = [c for c, flag in zip(features.columns, keep) if flag]
        self._median = self._median[keep]
        self._mean = self._mean[keep]
        self._std = self._std[keep]
        x = torch.as_tensor(self._scale_matrix(features), dtype=torch.float32, device=self.device_)
        y = torch.as_tensor(
            [self._class_to_i[v] for v in y_raw], dtype=torch.long, device=self.device_,
        )
        counts = torch.bincount(y, minlength=len(classes)).float()
        weights = len(y) / (len(classes) * counts.clamp_min(1.0))

        self._net = _MLP(x.shape[1], self.hidden_dim, self.dropout).to(self.device_)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss, best_state, stale = float("inf"), None, 0
        order = np.arange(len(y))
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self._net.train()
            rng.shuffle(order)
            total, seen = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                idx = torch.as_tensor(batch, device=self.device_)
                opt.zero_grad()
                loss = nn.functional.cross_entropy(self._net(x[idx]), y[idx], weight=weights)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
            value = total / max(seen, 1)
            if value < best_loss - 1e-5:
                best_loss, stale = value, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.to(self.device_)
        return self

    def _scale_matrix(self, features: pd.DataFrame) -> np.ndarray:
        raw = features[self._columns].to_numpy(np.float64) if set(self._columns) <= set(features.columns) else None
        if raw is None:
            raw = features.to_numpy(np.float64)[:, self._keep]
        raw = np.where(np.isnan(raw), self._median, raw)
        return ((raw - self._mean) / self._std).astype(np.float32)

    def _positive_index(self) -> int:
        classes = list(self.classes_)
        if 1 in classes:
            return classes.index(1)
        if 1.0 in classes:
            return classes.index(1.0)
        return len(classes) - 1


def integrated_gradients(
    model: WindowMLP,
    features: pd.DataFrame,
    steps: int = 32,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed IG of the positive logit.

    Returns ``(ig, s_w)``. ``ig`` has shape ``(n_windows, n_features)`` in the
    kept-column order. ``s_w`` is the sum over features.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    model._net.eval()
    x_np = model._scale_matrix(features)
    pos = model._positive_index()
    ig = np.empty_like(x_np)
    for start in range(0, len(x_np), chunk):
        x = torch.as_tensor(x_np[start:start + chunk], dtype=torch.float32, device=model.device_)
        acc = torch.zeros_like(x)
        for k in range(1, steps + 1):
            inp = ((k / steps) * x).detach().requires_grad_(True)
            logit = model._net(inp)[:, pos]
            model._net.zero_grad(set_to_none=True)
            logit.sum().backward()
            if inp.grad is not None:
                acc = acc + inp.grad.detach()
        ig[start:start + len(x)] = (x * acc / steps).detach().cpu().numpy()
    scores = ig.sum(axis=1)
    return ig, scores


def global_feature_importance(ig: np.ndarray) -> np.ndarray:
    """``x_f``: mean absolute attribution across windows. Shape ``(n_features,)``."""
    if len(ig) == 0:
        return np.zeros(ig.shape[1], dtype=float)
    return np.abs(ig).mean(axis=0)


def mean_feature_rank(fold_orders: list[list[str]]) -> pd.DataFrame:
    """Average 1-based ranks across folds. Lower mean rank is more important."""
    ranks: dict[str, list[int]] = {}
    for order in fold_orders:
        for rank, name in enumerate(order, start=1):
            ranks.setdefault(name, []).append(rank)
    rows = [
        {"feature": name, "mean_rank": float(np.mean(vals)), "n_folds": len(vals)}
        for name, vals in ranks.items()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mean_rank", "feature"]).reset_index(drop=True)


def recovery_at_k(ranked: list[str], motif: str, k: int = 5) -> dict:
    """Hit@K and Precision@K against the window-feature recovery rules."""
    rule = MIL_FEATURE_RULES[f"motif_{motif}"]
    rec = rank_recovery(ranked, rule, k=k)
    n_hit = int(rec["n_hit_in_top"])
    return {
        "hit_at_k": int(bool(rec["recovered"])),
        "precision_at_k": n_hit / k,
        "best_hit_rank": rec["best_hit_rank"],
        "top1": rec["top1"],
        "miss_as_top1": bool(rec["miss_as_top1"]),
        "passed": bool(rec["passed"]),
        "k": k,
    }
