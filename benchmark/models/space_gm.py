"""SPACE-GM style graph neural network region models: a GIN encoder + global
mean pooling, trained end-to-end. Unlike the tabular baselines, the "learning"
happens entirely inside `fit` — the featurizer (`SpaceGMGraphBuilder`) only
builds the graph and does not touch model weights.

Do NOT inherit `_TabularModel`: its median-impute + StandardScaler assumes a
2D numeric feature table and will either crash or silently mangle the
`"graph"` column if applied here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool

from .base import RegionModel


class _SpaceGMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 3, out_dim: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            d = hidden_dim
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, out_dim))

    def forward(self, batch: Batch) -> torch.Tensor:
        h = batch.x
        for conv in self.convs:
            h = F.relu(conv(h, batch.edge_index))
        pooled = global_mean_pool(h, batch.batch)     # (n_graphs, hidden_dim)
        return self.head(pooled)                       # (n_graphs, out_dim)


class _SpaceGMBase(RegionModel):
    def __init__(self, seed: int = 0, hidden_dim: int = 64, n_layers: int = 3,
                 lr: float = 1e-3, epochs: int = 100, weight_decay: float = 1e-4,
                 batch_size: int = 16, device: str = "cpu"):
        torch.manual_seed(seed)
        self.seed = seed
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.lr, self.epochs, self.weight_decay = lr, epochs, weight_decay
        self.batch_size = batch_size
        self.device = device
        self._encoder = None

    @staticmethod
    def _graphs(features: pd.DataFrame) -> list:
        return features["graph"].tolist()

    def _init_encoder(self, in_dim: int, out_dim: int):
        self._encoder = _SpaceGMEncoder(in_dim, self.hidden_dim, self.n_layers, out_dim).to(self.device)
        return torch.optim.Adam(self._encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class SpaceGMClassifier(_SpaceGMBase):
    """Binary / multiclass classification. Mini-batched (batch_size) is fine
    here since cross-entropy per-sample doesn't need a shared risk set."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SpaceGMClassifier":
        graphs = self._graphs(features.loc[list(target.index)])
        self.classes_ = np.sort(target.unique())
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y = target.map(class_to_int).to_numpy()
        opt = self._init_encoder(graphs[0].x.shape[1], out_dim=len(self.classes_))

        n = len(graphs)
        self._encoder.train()
        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                sel = perm[start:start + self.batch_size]
                batch = Batch.from_data_list([graphs[i] for i in sel]).to(self.device)
                yb = torch.as_tensor(y[sel], dtype=torch.long, device=self.device)
                opt.zero_grad()
                loss = F.cross_entropy(self._encoder(batch), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        graphs = self._graphs(features)
        self._encoder.eval()
        n = len(graphs)
        probs = np.zeros((n, len(self.classes_)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                sel = np.arange(start, min(start + self.batch_size, n))
                batch = Batch.from_data_list([graphs[i] for i in sel]).to(self.device)
                probs[sel] = F.softmax(self._encoder(batch), dim=1).cpu().numpy()
        return probs


class SpaceGMCox(_SpaceGMBase):
    """Cox proportional-hazards risk score via a Breslow partial-likelihood
    loss. `predict` returns the raw log-hazard (higher = worse prognosis,
    same monotone direction as `LinearCox.predict`'s partial hazard — a
    rank-based metric like C-index is invariant to the exp()).

    IMPORTANT: unlike SpaceGMClassifier, this ignores `batch_size` during
    `fit` and always uses the full training cohort as one batch — the Breslow
    approximation needs every sample in the risk set, so mini-batching here
    would silently compute the wrong (incomplete-risk-set) likelihood.
    """

    task_type = "survival"

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "SpaceGMCox":
        graphs = self._graphs(features.loc[list(target.index)])
        time = target["time"].to_numpy()
        event = target["event"].to_numpy().astype(np.float32)

        order = np.argsort(-time)                      # descending time -> risk sets are prefixes
        graphs = [graphs[i] for i in order]
        event_t = torch.as_tensor(event[order], dtype=torch.float32, device=self.device)

        opt = self._init_encoder(graphs[0].x.shape[1], out_dim=1)
        self._encoder.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            batch = Batch.from_data_list(graphs).to(self.device)   # full batch, see docstring
            risk = self._encoder(batch).squeeze(-1)                 # (n,) log-hazard
            log_cumsum = torch.logcumsumexp(risk, dim=0)
            loss = -((risk - log_cumsum) * event_t).sum() / event_t.sum().clamp(min=1)
            loss.backward()
            opt.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        graphs = self._graphs(features)
        self._encoder.eval()
        with torch.no_grad():
            batch = Batch.from_data_list(graphs).to(self.device)
            risk = self._encoder(batch).squeeze(-1)
        return risk.cpu().numpy()