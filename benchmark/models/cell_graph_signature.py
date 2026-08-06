"""GIN-TopK implementation of Cell-Graph Signature."""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, TopKPooling, global_max_pool, global_mean_pool

from .base import RegionModel


def _gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Execute the gin mlp operation.
    
        Args:
            in_dim (int): Dimensionality of the in representation.
            out_dim (int): Dimensionality of the out representation.
    
        Returns:
            nn.Sequential: The operation result.
    
    Args:
        in_dim (int): Dimensionality of the in representation."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


class _GINTopK(nn.Module):
    """Four GIN/TopK blocks followed by the paper's three-layer head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, pooling_ratio: float,
                 dropout: float):
        """Initialize the instance.
        
                Args:
                    in_dim (int): Dimensionality of the in representation.
                    out_dim (int): Dimensionality of the out representation.
                    hidden_dim (int): Dimensionality of the hidden representation.
                    pooling_ratio (float): Ratio controlling pooling.
                    dropout (float): Dropout probability used for neural-network regularization.
        
        Args:
            in_dim (int): Dimensionality of the in representation."""
        super().__init__()
        dims = [in_dim, hidden_dim, hidden_dim, hidden_dim]
        self.convs = nn.ModuleList(
            GINConv(_gin_mlp(dim, hidden_dim)) for dim in dims
        )
        self.pools = nn.ModuleList(
            TopKPooling(hidden_dim, ratio=pooling_ratio) for _ in range(4)
        )
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = nn.Linear(hidden_dim // 2, out_dim)
        self.dropout = dropout

    def forward(self, batch: Batch) -> torch.Tensor:
        """Execute the forward operation.
        
                Args:
                    batch (Batch): Batch of graphs or tensors processed together.
        
                Returns:
                    torch.Tensor: The operation result.
        
        Args:
            batch (Batch): Batch of graphs or tensors processed together."""
        x, edge_index, graph_index = batch.x, batch.edge_index, batch.batch
        summaries = []
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, graph_index, _, _ = pool(
                x, edge_index, None, graph_index
            )
            summaries.append(torch.cat((
                global_max_pool(x, graph_index),
                global_mean_pool(x, graph_index),
            ), dim=1))
        x = torch.stack(summaries).sum(dim=0)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin3(F.relu(self.lin2(x)))


class _CellGraphSignatureBase(RegionModel):
    """Represent cell graph signature base."""
    def __init__(
        self,
        *,
        seed: int = 0,
        hidden_dim: int = 256,
        pooling_ratio: float = 0.8,
        dropout: float = 0.5,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Initialize the instance.
        
                Args:
                    seed (int): Random seed used for reproducibility.
                    hidden_dim (int): Dimensionality of the hidden representation.
                    pooling_ratio (float): Ratio controlling pooling.
                    dropout (float): Dropout probability used for neural-network regularization.
                    lr (float): Learning rate used by the optimizer.
                    weight_decay (float): L2 penalty applied by the optimizer.
                    epochs (int): Number of epochs epochs.
                    patience (int): Epochs without improvement allowed before early stopping.
                    batch_size (int): Size of each batch.
                    device (str | None): Compute device on which tensors and models are allocated.
        
        Args:
            seed (int): Random seed used to make results reproducible."""
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.pooling_ratio = pooling_ratio
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network: _GINTopK | None = None

    @staticmethod
    def _region_graphs(features: pd.DataFrame) -> list[list]:
        """Execute the region graphs operation.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    list[list]: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        if "graphs" not in features:
            raise ValueError("Cell-Graph Signature expects a 'graphs' feature column")
        graphs = features["graphs"].tolist()
        if any(not region_graphs for region_graphs in graphs):
            raise ValueError("Cell-Graph Signature received a region with no cells")
        return graphs

    def _initialise(self, region_graphs: list[list], out_dim: int) -> None:
        """Execute the initialise operation.
        
                Args:
                    region_graphs (list[list]): Graphs grouped by their source tissue region.
                    out_dim (int): Dimensionality of the out representation.
        
        Args:
            region_graphs (list[list]): Graphs grouped by their source tissue region."""
        in_dim = int(region_graphs[0][0].x.shape[1])
        self.network = _GINTopK(
            in_dim, out_dim, self.hidden_dim, self.pooling_ratio, self.dropout
        ).to(self.device)

    def _graph_logits(self, graphs: list, *, training: bool) -> torch.Tensor:
        """Execute the graph logits operation.
        
                Args:
                    graphs (list): Graph objects used for training or inference.
                    training (bool): Whether the module is operating in training mode.
        
                Returns:
                    torch.Tensor: The operation result.
        
        Args:
            graphs (list): Graph objects used for training or inference."""
        assert self.network is not None
        parts = []
        for start in range(0, len(graphs), self.batch_size):
            batch = Batch.from_data_list(graphs[start:start + self.batch_size]).to(self.device)
            parts.append(self.network(batch))
        return torch.cat(parts)

    def _region_logits(self, region_graphs: list[list]) -> torch.Tensor:
        """Execute the region logits operation.
        
                Args:
                    region_graphs (list[list]): Graphs grouped by their source tissue region.
        
                Returns:
                    torch.Tensor: The operation result.
        
        Args:
            region_graphs (list[list]): Graphs grouped by their source tissue region."""
        return torch.stack([
            self._graph_logits(graphs, training=self.network.training).mean(dim=0)
            for graphs in region_graphs
        ])

    def _optimizer(self):
        """Execute the optimizer operation.

        Returns:
            Any: The operation result."""
        assert self.network is not None
        return torch.optim.Adam(
            self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class CellGraphSignatureClassifier(_CellGraphSignatureBase):
    """Binary/multiclass GIN-TopK classifier with graph-to-region averaging."""

    task_type = "binary"

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fit.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
                    target (pd.Series): Target labels or outcomes associated with the samples.
        
                Returns:
                    Any: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self.classes_ = np.sort(target.unique())
        if len(self.classes_) < 2:
            raise ValueError("classification requires at least two training classes")
        label_map = {label: i for i, label in enumerate(self.classes_)}
        labels = target.map(label_map).to_numpy()
        self._initialise(region_graphs, len(self.classes_))
        optimizer = self._optimizer()

        # Match the reference implementation: each 100-cell graph inherits its
        # tissue label.  Monitor training loss for bounded early stopping because
        # the benchmark supplies validation folds only outside model.fit().
        best_loss, stale, best_state = float("inf"), 0, None
        for _ in range(self.epochs):
            self.network.train()
            order = np.random.permutation(len(region_graphs))
            epoch_loss = 0.0
            for region_i in order:
                graphs = region_graphs[region_i]
                for start in range(0, len(graphs), self.batch_size):
                    selected = graphs[start:start + self.batch_size]
                    batch = Batch.from_data_list(selected).to(self.device)
                    y = torch.full(
                        (len(selected),), int(labels[region_i]),
                        dtype=torch.long, device=self.device,
                    )
                    optimizer.zero_grad()
                    loss = F.cross_entropy(self.network(batch), y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach()) * len(selected)
            n_graphs = sum(map(len, region_graphs))
            epoch_loss /= n_graphs
            if epoch_loss < best_loss - 1e-7:
                best_loss, stale = epoch_loss, 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for graphs in self._region_graphs(features):
                probability = F.softmax(self._graph_logits(graphs, training=False), dim=1)
                outputs.append(probability.mean(dim=0).cpu().numpy())
        return np.asarray(outputs)


class CellGraphSignatureCox(_CellGraphSignatureBase):
    """Survival extension using region-aggregated logits and Breslow Cox loss."""

    task_type = "survival"

    @staticmethod
    def _breslow_loss(risk, time, event):
        """Execute the breslow loss operation.
        
                Args:
                    risk (Any): Predicted risk scores for survival samples.
                    time (Any): Observed survival or follow-up times.
                    event (Any): Event indicators for survival outcomes.
        
                Returns:
                    Any: The operation result.
        
        Args:
            risk (Any): Predicted risk scores for survival samples."""
        loss = risk.new_tensor(0.0)
        for event_time in torch.unique(time[event > 0]):
            deaths = (time == event_time) & (event > 0)
            loss -= risk[deaths].sum()
            loss += deaths.sum() * torch.logsumexp(risk[time >= event_time], dim=0)
        return loss / event.sum().clamp(min=1.0)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame):
        """Fit.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
                    target (pd.DataFrame): Target labels or outcomes associated with the samples.
        
                Returns:
                    Any: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        region_graphs = self._region_graphs(features.loc[list(target.index)])
        self._initialise(region_graphs, 1)
        time = torch.tensor(target["time"].to_numpy(), dtype=torch.float32, device=self.device)
        event = torch.tensor(target["event"].to_numpy(), dtype=torch.float32, device=self.device)
        if not bool((event > 0).any()):
            raise ValueError("Cox training requires at least one observed event")
        optimizer = self._optimizer()
        best_loss, stale, best_state = float("inf"), 0, None
        for epoch in range(self.epochs):
            self.network.train()

            # Keeping the autograd graph for every region until the full Cox
            # loss is formed makes memory grow with cohort size.  Compute the
            # exact full-cohort gradient in two passes: first differentiate the
            # Cox objective with respect to detached scalar region risks, then
            # recompute and backpropagate one region at a time.  Restoring RNG
            # states keeps dropout identical between the two passes.
            cpu_states = []
            cuda_states = []
            risk_values = []
            with torch.no_grad():
                for graphs in region_graphs:
                    cpu_states.append(torch.random.get_rng_state())
                    if str(self.device).startswith("cuda"):
                        cuda_states.append(torch.cuda.get_rng_state(self.device))
                    risk_values.append(
                        self._graph_logits(
                            graphs, training=self.network.training
                        ).mean(dim=0).squeeze()
                    )

            detached_risk = torch.stack(risk_values).detach().requires_grad_(True)
            detached_loss = self._breslow_loss(detached_risk, time, event)
            risk_gradient, = torch.autograd.grad(detached_loss, detached_risk)

            optimizer.zero_grad()
            for region_i, graphs in enumerate(region_graphs):
                torch.random.set_rng_state(cpu_states[region_i])
                if str(self.device).startswith("cuda"):
                    torch.cuda.set_rng_state(cuda_states[region_i], self.device)
                risk = self._graph_logits(
                    graphs, training=self.network.training
                ).mean(dim=0).squeeze()
                risk.backward(gradient=risk_gradient[region_i])
            optimizer.step()
            value = float(detached_loss.detach())
            if os.environ.get("BENCHMARK_PROGRESS") and (
                epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.epochs
            ):
                print(
                    f"      Cell-Graph Cox epoch {epoch + 1}/{self.epochs} "
                    f"loss={value:.6f}",
                    flush=True,
                )
            if value < best_loss - 1e-7:
                best_loss, stale = value, 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        self.network.eval()
        with torch.no_grad():
            return self._region_logits(self._region_graphs(features)).squeeze(1).cpu().numpy()
