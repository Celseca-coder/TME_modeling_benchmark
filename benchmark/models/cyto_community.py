"""Cyto-Community region-level models.

This baseline follows a cell -> community -> tissue hierarchy. A GraphSAGE
encoder embeds cells, a soft assignment layer pools cells into a fixed number
of latent communities, and a task head predicts region phenotype or survival
risk from the community summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool

from .base import RegionModel


class _CytoCommunityNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, n_communities: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_communities = n_communities
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_projection = nn.Linear(max(in_dim, 1), hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.assignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_communities),
        )
        self.head = nn.Sequential(
            nn.Linear((n_communities + 1) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode_nodes(self, batch: Batch) -> torch.Tensor:
        h = F.relu(self.input_projection(batch.x))
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _region_embedding(self, h: torch.Tensor, batch_index: torch.Tensor,
                          n_graphs: int) -> torch.Tensor:
        global_embedding = global_mean_pool(h, batch_index, size=n_graphs)
        logits = self.assignment(h)
        region_parts = []
        for graph_i in range(n_graphs):
            mask = batch_index == graph_i
            if not bool(mask.any()):
                region_parts.append(h.new_zeros(self.n_communities * h.shape[1]))
                continue
            weights = F.softmax(logits[mask], dim=0)
            community = weights.T @ h[mask]
            region_parts.append(community.reshape(-1))
        community_embedding = torch.stack(region_parts, dim=0)
        return torch.cat([community_embedding, global_embedding], dim=1)

    def forward(self, batch: Batch) -> torch.Tensor:
        n_graphs = int(batch.num_graphs)
        if batch.x.numel() == 0:
            return self.head(batch.x.new_zeros((n_graphs, (self.n_communities + 1) * self.hidden_dim)))
        h = self._encode_nodes(batch)
        z = self._region_embedding(h, batch.batch, n_graphs)
        return self.head(z)


class _CytoCommunityBase(RegionModel):
    def __init__(self, seed: int = 0, hidden_dim: int = 128, n_layers: int = 2,
                 n_communities: int = 8, lr: float = 1e-3, epochs: int = 75,
                 weight_decay: float = 1e-4, dropout: float = 0.1,
                 device: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_communities = n_communities
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _CytoCommunityNet | None = None

    @staticmethod
    def _graphs(features: pd.DataFrame) -> list:
        return features["graph"].tolist()

    def _initialise(self, graphs: list, out_dim: int):
        first = next((graph for graph in graphs if graph.x is not None), None)
        in_dim = int(first.num_node_features) if first is not None else 1
        self.net = _CytoCommunityNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_communities=self.n_communities,
            dropout=self.dropout,
        ).to(self.device)

    def _batch(self, graphs: list) -> Batch:
        return Batch.from_data_list(graphs).to(self.device)

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class CytoCommunityClassifier(_CytoCommunityBase):
    task_type = "binary"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "CytoCommunityClassifier":
        graphs = self._graphs(features.loc[list(target.index)])
        self.classes_ = np.sort(target.unique())
        class_to_int = {label: i for i, label in enumerate(self.classes_)}
        y = torch.tensor(target.map(class_to_int).to_numpy(), dtype=torch.long, device=self.device)
        self._initialise(graphs, len(self.classes_))
        batch = self._batch(graphs)
        optimizer = self._optimizer()
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.net(batch), y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        batch = self._batch(self._graphs(features))
        with torch.no_grad():
            probs = F.softmax(self.net(batch), dim=1).cpu().numpy()
        return probs


class CytoCommunityCox(_CytoCommunityBase):
    task_type = "survival"

    @staticmethod
    def _breslow_loss(risk, time, event):
        loss = risk.new_tensor(0.0)
        n_events = event.sum().clamp(min=1.0)
        for event_time in torch.unique(time[event > 0]):
            deaths = (time == event_time) & (event > 0)
            risk_set = time >= event_time
            loss -= risk[deaths].sum() - deaths.sum() * torch.logsumexp(risk[risk_set], dim=0)
        return loss / n_events

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "CytoCommunityCox":
        graphs = self._graphs(features.loc[list(target.index)])
        self._initialise(graphs, 1)
        time = torch.tensor(target["time"].to_numpy(), dtype=torch.float32, device=self.device)
        event = torch.tensor(target["event"].to_numpy(), dtype=torch.float32, device=self.device)
        batch = self._batch(graphs)
        optimizer = self._optimizer()
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            risk = self.net(batch).squeeze(-1)
            self._breslow_loss(risk, time, event).backward()
            optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        batch = self._batch(self._graphs(features))
        with torch.no_grad():
            risk = self.net(batch).squeeze(-1).cpu().numpy()
        return risk
