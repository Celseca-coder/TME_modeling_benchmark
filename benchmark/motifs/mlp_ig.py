"""Bag-level window MLP with Integrated Gradients.

Each region is one bag. The MLP maps a window to a logit, the bag logit is
the mean, and the bag probability is trained against the region label.
Window score is the sum of absolute IG of that probability. The baseline is
the zero tensor in scaled space, which is the training-set mean.
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WindowMLP:
    """Mean-pooled window MLP. Training target is the region label."""

    def __init__(
        self,
        seed: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-4,
        l1_lambda: float = 1e-5,
        epochs: int = 80,
        patience: int = 15,
        device: str = "auto",
    ) -> None:
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.patience = patience
        self.device_name = device

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def fit(self, features: pd.DataFrame, target: pd.Series, columns: list[str] | None = None) -> "WindowMLP":
        """``features`` has a ``bag`` column of ``(n_windows, n_features)`` arrays."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if "bag" not in features.columns:
            raise ValueError("WindowMLP requires a 'bag' column")
        target = target.loc[features.index]
        y_raw = target.to_numpy(float)
        classes = np.asarray(sorted(pd.unique(y_raw).tolist()))
        if len(classes) < 2:
            raise ValueError("Window MLP requires two region classes")
        self.classes_ = classes
        self.device_ = self._device()
        bags = [np.asarray(bag, dtype=np.float64) for bag in features["bag"]]
        if columns is None:
            width = next(bag.shape[1] for bag in bags if bag.ndim == 2 and len(bag))
            columns = [f"f{i}" for i in range(width)]
        nonempty = [bag for bag in bags if bag.ndim == 2 and len(bag)]
        if not nonempty:
            raise ValueError("all training bags are empty")
        raw = np.concatenate(nonempty, axis=0)
        if raw.shape[1] != len(columns):
            raise ValueError(f"bag width {raw.shape[1]} does not match {len(columns)} columns")
        median = np.nanmedian(raw, axis=0)
        median = np.nan_to_num(median, nan=0.0)
        filled = np.where(np.isnan(raw), median, raw)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        keep = std >= 1e-8
        if not np.any(keep):
            raise ValueError("no non-constant features")
        self._columns = [col for col, flag in zip(columns, keep) if flag]
        # 去除不能标准化的列
        self._keep = keep
        self._median = median[keep]
        self._mean = mean[keep]
        self._std = std[keep]
        y = torch.as_tensor(y_raw, dtype=torch.float32, device=self.device_)
        self._net = _MLP(int(keep.sum()), self.hidden_dim, self.dropout).to(self.device_)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self._net.train()
            opt.zero_grad()
            losses = []
            for bag, label in zip(bags, y):
                logit = self._net(self._tensor(bag)).mean()
                losses.append(nn.functional.binary_cross_entropy_with_logits(logit, label))
            penalty = self.l1_lambda * sum(param.abs().sum() for param in self._net.parameters() if param.ndim > 1)
            loss = torch.stack(losses).mean() + penalty
            loss.backward()
            opt.step()
            value = float(loss.detach().cpu())
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

    def _tensor(self, bag: np.ndarray) -> torch.Tensor:
        raw = np.asarray(bag, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self._keep):
            raise ValueError(f"Invalid bag shape {raw.shape}; expected (*, {len(self._keep)})")
        if len(raw) == 0:
            raw = np.zeros((1, raw.shape[1]), dtype=np.float64)
        raw = raw[:, self._keep]
        raw = np.where(np.isnan(raw), self._median, raw)
        scaled = ((raw - self._mean) / self._std).astype(np.float32)
        return torch.as_tensor(scaled, dtype=torch.float32, device=self.device_)

    def bag_probability(self, bag: np.ndarray) -> float:
        self._net.eval()
        with torch.no_grad():
            logit = self._net(self._tensor(bag)).mean()
            return float(torch.sigmoid(logit).cpu())


def integrated_gradients(
    model: WindowMLP,
    bag: np.ndarray,
    steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """IG of the bag probability. Baseline is the zero tensor.

    Returns ``(ig, s_w)``. ``ig`` has shape ``(n_windows, n_features)``.
    ``s_w`` is the sum of absolute feature attributions.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    n = int(len(bag))
    width = len(model._columns)
    if n == 0:
        return np.zeros((0, width), dtype=np.float64), np.zeros(0, dtype=np.float64)
    model._net.eval()
    x = model._tensor(bag).detach()
    acc = torch.zeros_like(x)
    for k in range(1, steps + 1):
        inp = ((k / steps) * x).detach().requires_grad_(True)
        prob = torch.sigmoid(model._net(inp).mean())
        model._net.zero_grad(set_to_none=True)
        prob.backward()
        if inp.grad is not None:
            acc = acc + inp.grad.detach()
    ig = (x * acc / steps).detach().cpu().numpy().astype(np.float64, copy=False)
    scores = np.abs(ig).sum(axis=1)
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
