"""SPACE-GM models with expression pre-training and region-level aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool

from .base import RegionModel


class _EdgeGINLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_embedding = nn.Embedding(3, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_type):
        source, target = edge_index
        messages = h[source] + self.edge_embedding(edge_type)
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, target, messages)
        return self.mlp((1.0 + self.eps) * h + aggregated)


class _SpaceGMBackbone(nn.Module):
    def __init__(self, n_cell_types: int, hidden_dim: int = 512, n_layers: int = 3):
        super().__init__()
        self.type_embedding = nn.Embedding(max(n_cell_types, 1), hidden_dim)
        self.aux_projection = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([_EdgeGINLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, batch: Batch):
        h = self.type_embedding(batch.cell_type) + self.aux_projection(batch.aux)
        for layer in self.layers:
            h = F.leaky_relu(layer(h, batch.edge_index, batch.edge_type), 0.1)
        pooled = global_max_pool(h, batch.batch)
        centers = h[batch.aux[:, 1] > 0.5]
        return centers, pooled


def _three_layer_head(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, in_dim), nn.LeakyReLU(0.1),
        nn.Linear(in_dim, in_dim // 2), nn.LeakyReLU(0.1),
        nn.Linear(in_dim // 2, out_dim),
    )


class _SpaceGMBase(RegionModel):
    def __init__(self, seed: int = 0, hidden_dim: int = 512, n_layers: int = 3,
                 lr: float = 1e-4, pretrain_epochs: int = 20, epochs: int = 100,
                 weight_decay: float = 1e-4, micro_batch_size: int = 32,
                 device: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.lr, self.pretrain_epochs, self.epochs = lr, pretrain_epochs, epochs
        self.weight_decay = weight_decay
        self.micro_batch_size = micro_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.head = None

    @staticmethod
    def _region_graphs(features: pd.DataFrame) -> list[list]:
        return features["graphs"].tolist()

    def _initialise(self, region_graphs: list[list], out_dim: int):
        first = next(graph for graphs in region_graphs for graph in graphs)
        n_cell_types = max(int(graph.cell_type.max()) for graphs in region_graphs for graph in graphs) + 1
        self.backbone = _SpaceGMBackbone(n_cell_types, self.hidden_dim, self.n_layers).to(self.device)
        self.head = _three_layer_head(self.hidden_dim, out_dim).to(self.device)
        self._pretrain(region_graphs, int(first.center_expression.numel()))

    def _encode_graphs(self, graphs: list, need_centers: bool = False):
        if not graphs:
            raise ValueError("SPACE-GM received a region with no microenvironments")
        center_parts, pooled_parts = [], []
        for start in range(0, len(graphs), self.micro_batch_size):
            batch = Batch.from_data_list(graphs[start:start + self.micro_batch_size]).to(self.device)
            centers, pooled = self.backbone(batch)
            center_parts.append(centers)
            pooled_parts.append(pooled)
        centers = torch.cat(center_parts) if need_centers else None
        return centers, torch.cat(pooled_parts)

    def _pretrain(self, region_graphs: list[list], n_markers: int):
        if self.pretrain_epochs <= 0 or n_markers == 0:
            return
        expression_head = _three_layer_head(self.hidden_dim, n_markers).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(expression_head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        all_graphs = [graph for graphs in region_graphs for graph in graphs]
        self.backbone.train(); expression_head.train()
        for _ in range(self.pretrain_epochs):
            order = np.random.permutation(len(all_graphs))
            for start in range(0, len(order), self.micro_batch_size):
                selected = [all_graphs[i] for i in order[start:start + self.micro_batch_size]]
                batch = Batch.from_data_list(selected).to(self.device)
                optimizer.zero_grad()
                centers, _ = self.backbone(batch)
                target = batch.center_expression.reshape(len(selected), n_markers)
                loss = F.mse_loss(expression_head(centers), target)
                loss.backward(); optimizer.step()

    def _region_output(self, graphs: list):
        _, embeddings = self._encode_graphs(graphs)
        # Mean aggregation across all microenvironments in the same tissue region.
        return self.head(embeddings).mean(dim=0)

    def _optimizer(self):
        return torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )


class SpaceGMClassifier(_SpaceGMBase):
    task_type = "binary"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SpaceGMClassifier":
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self.classes_ = np.sort(target.unique())
        class_to_int = {label: i for i, label in enumerate(self.classes_)}
        y = target.map(class_to_int).to_numpy()
        self._initialise(region_graphs, len(self.classes_))
        optimizer = self._optimizer()
        self.backbone.train(); self.head.train()
        for _ in range(self.epochs):
            for i in np.random.permutation(len(region_graphs)):
                optimizer.zero_grad()
                logits = self._region_output(region_graphs[i]).unsqueeze(0)
                label = torch.tensor([y[i]], dtype=torch.long, device=self.device)
                F.cross_entropy(logits, label).backward(); optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.backbone.eval(); self.head.eval()
        outputs = []
        with torch.no_grad():
            for graphs in self._region_graphs(features):
                outputs.append(F.softmax(self._region_output(graphs), dim=0).cpu().numpy())
        return np.asarray(outputs)


class SpaceGMCox(_SpaceGMBase):
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

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "SpaceGMCox":
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self._initialise(region_graphs, 1)
        time = torch.tensor(target["time"].to_numpy(), dtype=torch.float32, device=self.device)
        event = torch.tensor(target["event"].to_numpy(), dtype=torch.float32, device=self.device)
        optimizer = self._optimizer()
        self.backbone.train(); self.head.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            risk = torch.stack([self._region_output(graphs).squeeze() for graphs in region_graphs])
            self._breslow_loss(risk, time, event).backward(); optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.backbone.eval(); self.head.eval()
        with torch.no_grad():
            risk = [self._region_output(graphs).item() for graphs in self._region_graphs(features)]
        return np.asarray(risk)
