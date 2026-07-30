"""SORBET-style graph neural network models.

This implementation follows the SORBET idea at benchmark scale: local
cell-neighborhood subgraphs are embedded with a GCN, then neighborhood evidence
is aggregated into one region-level classifier or Cox risk score.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from .base import RegionModel


class _SORBETNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.dropout = dropout
        self.input_projection = nn.Linear(max(in_dim, 1), hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        if batch.x.numel() == 0:
            return batch.x.new_zeros((batch.num_graphs, self.head[-1].out_features))
        h = F.relu(self.input_projection(batch.x))
        for conv in self.convs:
            h = F.relu(conv(h, batch.edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = global_mean_pool(h, batch.batch, size=batch.num_graphs)
        center_weight = getattr(batch, "center_mask", None)
        if center_weight is None:
            center = pooled
        else:
            selected = h * center_weight.reshape(-1, 1)
            center = global_mean_pool(selected, batch.batch, size=batch.num_graphs)
        return self.head(torch.cat([pooled, center], dim=1))


class _SORBETBase(RegionModel):
    def __init__(self, seed: int = 0, hidden_dim: int = 128, n_layers: int = 2,
                 lr: float = 1e-3, epochs: int = 75, weight_decay: float = 1e-4,
                 dropout: float = 0.15, micro_batch_size: int = 64,
                 region_batch_size: int = 4, device: str | None = None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.micro_batch_size = micro_batch_size
        self.region_batch_size = region_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _SORBETNet | None = None

    @staticmethod
    def _region_graphs(features: pd.DataFrame) -> list[list]:
        return features["graphs"].tolist()

    def _initialise(self, region_graphs: list[list], out_dim: int) -> None:
        first = next((graph for graphs in region_graphs for graph in graphs), None)
        in_dim = int(first.num_node_features) if first is not None else 1
        self.net = _SORBETNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _subgraph_logits(self, graphs: list) -> torch.Tensor:
        if not graphs:
            out_dim = self.net.head[-1].out_features
            return next(self.net.parameters()).new_zeros((1, out_dim))
        outputs = []
        for start in range(0, len(graphs), self.micro_batch_size):
            batch = Batch.from_data_list(graphs[start:start + self.micro_batch_size]).to(self.device)
            outputs.append(self.net(batch))
        return torch.cat(outputs, dim=0)

    def _region_logits(self, graphs: list) -> torch.Tensor:
        return self._subgraph_logits(graphs).mean(dim=0)

    def _region_batch_logits(self, region_graphs: list[list]) -> torch.Tensor:
        return torch.stack([self._region_logits(graphs) for graphs in region_graphs])

    @staticmethod
    def _check_finite(name: str, tensor: torch.Tensor) -> None:
        if os.environ.get("BENCHMARK_RAISE_ERRORS") and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"SORBET produced non-finite {name}")


class SORBETClassifier(_SORBETBase):
    task_type = "binary"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SORBETClassifier":
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self.classes_ = np.sort(target.unique())
        class_to_int = {label: i for i, label in enumerate(self.classes_)}
        y = torch.tensor(target.map(class_to_int).to_numpy(), dtype=torch.long, device=self.device)
        self._initialise(region_graphs, len(self.classes_))
        optimizer = self._optimizer()
        self.net.train()
        indices = np.arange(len(region_graphs))
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.region_batch_size):
                batch_idx = indices[start:start + self.region_batch_size]
                optimizer.zero_grad()
                logits = self._region_batch_logits([region_graphs[i] for i in batch_idx])
                self._check_finite("classification logits", logits)
                loss = F.cross_entropy(logits, y[batch_idx])
                self._check_finite("classification loss", loss)
                loss.backward()
                optimizer.step()
            if os.environ.get("BENCHMARK_PROGRESS") and (
                epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.epochs
            ):
                print(
                    f"      SORBET classifier epoch {epoch + 1}/{self.epochs}",
                    flush=True,
                )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for graphs in self._region_graphs(features):
                logits = self._region_logits(graphs)
                self._check_finite("prediction logits", logits)
                outputs.append(F.softmax(logits, dim=0).cpu().numpy())
        return np.asarray(outputs)


class SORBETCox(_SORBETBase):
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

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "SORBETCox":
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self._initialise(region_graphs, 1)
        time = torch.tensor(target["time"].to_numpy(), dtype=torch.float32, device=self.device)
        event = torch.tensor(target["event"].to_numpy(), dtype=torch.float32, device=self.device)
        optimizer = self._optimizer()
        self.net.train()
        for epoch in range(self.epochs):
            # A naïve full-batch Cox backward retains the autograd activations of
            # every subgraph in every region until the loss is formed.  Large
            # cohorts can therefore consume the entire GPU even with a small
            # micro_batch_size.  Compute the exact same gradient in two passes:
            #
            #   1. evaluate region risks without autograd and differentiate the
            #      Cox loss with respect to those scalar risks;
            #   2. recompute one region at a time and immediately backpropagate
            #      its precomputed d(loss)/d(risk).
            #
            # RNG states are restored before the second pass, so dropout masks
            # match the first pass.  This is an exact low-memory gradient, not a
            # mini-batch approximation to the Cox objective.
            cpu_states = []
            cuda_states = []
            risk_values = []
            with torch.no_grad():
                for graphs in region_graphs:
                    cpu_states.append(torch.random.get_rng_state())
                    if str(self.device).startswith("cuda"):
                        cuda_states.append(torch.cuda.get_rng_state(self.device))
                    risk_values.append(self._region_logits(graphs).squeeze())

            detached_risk = torch.stack(risk_values).detach().requires_grad_(True)
            self._check_finite("cox risk", detached_risk)
            detached_loss = self._breslow_loss(detached_risk, time, event)
            self._check_finite("cox loss", detached_loss)
            risk_gradient, = torch.autograd.grad(detached_loss, detached_risk)

            optimizer.zero_grad()
            for region_i, graphs in enumerate(region_graphs):
                torch.random.set_rng_state(cpu_states[region_i])
                if str(self.device).startswith("cuda"):
                    torch.cuda.set_rng_state(cuda_states[region_i], self.device)
                risk = self._region_logits(graphs).squeeze()
                self._check_finite("recomputed cox risk", risk)
                risk.backward(gradient=risk_gradient[region_i])
            optimizer.step()
            if os.environ.get("BENCHMARK_PROGRESS") and (
                epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.epochs
            ):
                print(
                    f"      SORBET Cox epoch {epoch + 1}/{self.epochs} "
                    f"loss={float(detached_loss):.6f}",
                    flush=True,
                )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        risks = []
        with torch.no_grad():
            for graphs in self._region_graphs(features):
                risk = self._region_logits(graphs).squeeze()
                self._check_finite("prediction risk", risk)
                risks.append(float(risk.cpu()))
        return np.asarray(risks)
