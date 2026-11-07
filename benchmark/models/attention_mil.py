"""Gated-attention MIL for interpretable handcrafted local features."""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import RegionModel


class _GatedAttentionNet(nn.Module):
    """Two-layer encoder, gated attention, dropout on the pooled vector."""

    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int,
                 output_dim: int, dropout: float) -> None:
        super().__init__()
        # Encoder加了两层ReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 加入可偏置开关和dropout
        self.attn_v = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_u = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_w = nn.Linear(attention_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, bag: torch.Tensor):
        h = self.encoder(bag)
        logits = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        pooled = torch.sum(weights[:, None] * h, dim=0)
        return self.head(self.drop(pooled)), weights


class AttentionMILModel(RegionModel):
    """Trainable gated-attention pooling over local handcrafted feature bags.

    Supports binary/multiclass classification and Cox survival.  Instance feature
    imputation/scaling is fitted on training bags only. ``attention_weights`` can
    subsequently map predictions back to local-window centers.
    """

    def __init__(self, task_type: str = "binary", seed: int = 0,
                 hidden_dim: int = 64, attention_dim: int = 32,
                 dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 200,
                 patience: int = 25, device: str = "auto",
                 max_instances: int | None = 512) -> None:
        self.task_type = task_type
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device_name = device
        self.max_instances = max_instances
        self.classes_ = None

    def _seed_all(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _device(self):
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    @staticmethod
    def _raw_bags(features: pd.DataFrame) -> list[np.ndarray]:
        if "bag" not in features.columns:
            raise ValueError("AttentionMILModel requires a 'bag' feature column")
        return [np.asarray(x, dtype=np.float32) for x in features["bag"]]

    def _fit_scaler(self, bags: list[np.ndarray]):
        nonempty = [x for x in bags if x.ndim == 2 and len(x)]
        if not nonempty:
            raise ValueError("All training regions have empty local-window bags")
        all_x = np.concatenate(nonempty, axis=0).astype(float)
        self._median = np.nanmedian(all_x, axis=0)
        self._median = np.nan_to_num(self._median, nan=0.0)
        filled = np.where(np.isnan(all_x), self._median[None, :], all_x)
        self._mean = filled.mean(axis=0)
        self._std = filled.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        self._input_dim = all_x.shape[1]

    def keep_indices(self, n: int, training: bool = False) -> np.ndarray:
        """Windows actually scored when a bag exceeds ``max_instances``."""
        if n <= 0:
            return np.zeros(0, dtype=int)
        if not self.max_instances or n <= self.max_instances:
            return np.arange(n, dtype=int)
        if training:
            rng = getattr(self, "_rng", np.random.default_rng(self.seed))
            return np.sort(rng.choice(n, self.max_instances, replace=False))
        return np.linspace(0, n - 1, self.max_instances, dtype=int)

    def _bag_tensor(self, bag: np.ndarray, training: bool = False):
        if bag.ndim != 2 or bag.shape[1] != self._input_dim:
            raise ValueError(f"Invalid bag shape {bag.shape}; expected (*, {self._input_dim})")
        if len(bag) == 0:
            bag = np.zeros((1, self._input_dim), dtype=np.float32)
        else:
            bag = bag[self.keep_indices(len(bag), training=training)]
        bag = np.where(np.isnan(bag), self._median[None, :], bag)
        bag = (bag - self._mean[None, :]) / self._std[None, :]
        return torch.as_tensor(bag, dtype=torch.float32, device=self.device_)

    def _make_net(self, output_dim: int) -> nn.Module:
        return _GatedAttentionNet(
            self._input_dim, self.hidden_dim, self.attention_dim, output_dim, self.dropout
        )

    def _penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.device_)

    # 交叉熵函数计算分类损失
    def _classification_loss(self, outputs: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Binary bags use one logit and BCE. Multiclass stays cross-entropy."""
        if n_classes == 2:
            return nn.functional.binary_cross_entropy_with_logits(
                outputs.reshape(-1), y.reshape(-1).float(),
            )
        counts = torch.bincount(y, minlength=n_classes).float()
        weights = len(y) / (n_classes * counts.clamp_min(1.0))
        return nn.functional.cross_entropy(outputs, y, weight=weights)

    def _positive_index(self) -> int:
        classes = list(self.classes_)
        if 1 in classes:
            return classes.index(1)
        # 有时候标签可能是1.0
        if 1.0 in classes:
            return classes.index(1.0)
        return len(classes) - 1

    # 返回分数正/负的一维
    def _positive_scalar(self, raw: torch.Tensor) -> torch.Tensor:
        flat = raw.reshape(-1)
        if flat.numel() == 1:
            return flat[0]
        return raw.reshape(-1)[self._positive_index()]

    # sigma
    def positive_logit(self, bag: np.ndarray) -> float:
        """Positive-class logit for one bag. Probability models return the bag score."""
        self._net.eval()
        with torch.no_grad():
            raw, _ = self._net(self._bag_tensor(bag))
            return float(self._positive_scalar(raw).cpu())
    # sigmoid
    def positive_proba(self, bag: np.ndarray) -> float:
        """Positive-class probability for one bag."""
        self._net.eval()
        with torch.no_grad():
            raw, _ = self._net(self._bag_tensor(bag))
            scalar = self._positive_scalar(raw)
            if getattr(self, "_bag_probability", False):
                return float(scalar.cpu())
            if len(self.classes_) == 2:
                return float(torch.sigmoid(scalar).cpu())
            return float(torch.softmax(raw.reshape(-1), dim=0)[self._positive_index()].cpu())

    def forward_scaled(self, tensor: torch.Tensor):
        """Gated-attention forward on an already-scaled bag tensor."""
        return self._net(tensor)

    @staticmethod
    def _cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        order = torch.argsort(time, descending=True)
        risk, event = risk[order], event[order]
        log_risk = torch.logcumsumexp(risk, dim=0)
        observed = event > 0
        if not torch.any(observed):
            raise ValueError("Training survival fold has no observed events")
        return -(risk[observed] - log_risk[observed]).mean()

    def fit(self, features: pd.DataFrame, target):
        self._seed_all()
        self._rng = np.random.default_rng(self.seed)
        features = features.loc[list(target.index)]
        bags = self._raw_bags(features)
        self._fit_scaler(bags)
        self.device_ = self._device()

        survival = self.task_type == "survival"
        n_classes = 2
        if survival:
            output_dim = 1
            times = torch.as_tensor(target["time"].to_numpy(float), dtype=torch.float32,
                                    device=self.device_)
            events = torch.as_tensor(target["event"].to_numpy(float), dtype=torch.float32,
                                     device=self.device_)
        else:
            self.classes_ = np.asarray(sorted(pd.Series(target).unique().tolist()))
            if len(self.classes_) < 2:
                raise ValueError("Attention MIL classification requires at least two classes")
            class_to_i = {c: i for i, c in enumerate(self.classes_)}
            y = torch.as_tensor([class_to_i[v] for v in target.to_numpy()], dtype=torch.long,
                                device=self.device_)
            n_classes = len(self.classes_)
            output_dim = 1 if n_classes == 2 else n_classes

        self._net = self._make_net(output_dim).to(self.device_)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self._net.train()
            optimizer.zero_grad()
            outputs = torch.stack([
                self._net(self._bag_tensor(bag, training=True))[0].reshape(-1) for bag in bags
            ])
            if survival:
                loss = self._cox_loss(outputs[:, 0], times, events)
            else:
                loss = self._classification_loss(outputs, y, n_classes)
            loss = loss + self._penalty()
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            if value < best_loss - 1e-5:
                best_loss, stale = value, 0
                best_state = copy.deepcopy(self._net.state_dict())
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._net.eval()
        outputs = []
        with torch.no_grad():
            for bag in self._raw_bags(features):
                raw, _ = self._net(self._bag_tensor(bag))
                if self.task_type == "survival":
                    # 越大预后越差
                    outputs.append(float(raw.reshape(-1)[0].cpu()))
                elif getattr(self, "_bag_probability", False):
                    p1 = float(raw.reshape(-1)[0].cpu())
                    # P_{bag}
                    outputs.append(np.array([1.0 - p1, p1], dtype=np.float64))
                elif len(self.classes_) == 2:
                    p1 = float(torch.sigmoid(raw.reshape(-1)[0]).cpu())
                    outputs.append(np.array([1.0 - p1, p1], dtype=np.float64))
                else:
                    outputs.append(torch.softmax(raw.reshape(-1), dim=0).cpu().numpy())
        return np.asarray(outputs)

    def attention_weights(self, features: pd.DataFrame) -> list[np.ndarray]:
        """Return normalized weights in window order.

        If a bag exceeds ``max_instances``, weights correspond to the deterministic
        evenly-spaced subset used by :meth:`predict`.
        """
        self._net.eval()
        result = []
        with torch.no_grad():
            for bag in self._raw_bags(features):
                _, weights = self._net(self._bag_tensor(bag))
                result.append(weights.cpu().numpy())
        return result
