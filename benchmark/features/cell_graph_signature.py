"""Cell-Graph Signature graph construction.

The reference implementation splits every tissue into consecutive 100-cell
graphs and connects cells whose centres are less than 20 microns apart.  This
adapter keeps that construction while learning the marker vocabulary and
per-marker scaling from the training fold only.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class CellGraphSignatureBuilder(BaseFeatureExtractor):
    """Build the collection of cell graphs associated with each tissue region.

    Parameters
    ----------
    graph_size:
        Maximum number of cells per graph.  The paper's implementation uses 100.
    radius_um:
        Strict distance threshold for cell-cell edges, in microns.
    """

    def __init__(self, graph_size: int = 100, radius_um: float = 20.0):
        """Initialize the instance.
        
                Args:
                    graph_size (int): Number of nodes retained in each cell graph.
                    radius_um (float): Radius measured in micrometers.
        
        Args:
            graph_size (int): Number of nodes retained in each cell graph."""
        if graph_size < 1:
            raise ValueError("graph_size must be at least 1")
        if radius_um <= 0:
            raise ValueError("radius_um must be positive")
        self.graph_size = int(graph_size)
        self.radius_um = float(radius_um)
        self._markers: list[str] | None = None
        self._scale: np.ndarray | None = None

    def fit(self, regions: list[RegionData]) -> "CellGraphSignatureBuilder":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'CellGraphSignatureBuilder': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        marker_sets = [set(region.expression.columns) for region in regions]
        self._markers = sorted(set.intersection(*marker_sets)) if marker_sets else []
        if not self._markers:
            raise ValueError("Cell-Graph Signature requires shared expression markers")

        maxima = np.zeros(len(self._markers), dtype=np.float64)
        for region in regions:
            values = region.expression.reindex(columns=self._markers).to_numpy(float)
            finite = np.where(np.isfinite(values), values, -np.inf)
            if len(values):
                maxima = np.maximum(maxima, finite.max(axis=0))
        self._scale = np.where(np.isfinite(maxima) & (maxima > 0), maxima, 1.0)
        return self

    def extract_region(self, region: RegionData) -> dict[str, list[Data]]:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, list[Data]]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self._markers is None or self._scale is None:
            raise RuntimeError("call fit() before extract_region()")

        cell_ids = region.coordinates.index
        coords = (
            region.coordinates.reindex(cell_ids)[["x", "y"]]
            .to_numpy(dtype=np.float64)
        )
        expression = (
            region.expression.reindex(index=cell_ids, columns=self._markers)
            .to_numpy(dtype=np.float32)
        )
        expression = np.nan_to_num(expression, nan=0.0, posinf=0.0, neginf=0.0)
        expression = expression / self._scale.astype(np.float32)

        graphs: list[Data] = []
        for start in range(0, len(cell_ids), self.graph_size):
            stop = min(start + self.graph_size, len(cell_ids))
            chunk_coords = coords[start:stop]
            pairs = sorted(cKDTree(chunk_coords).query_pairs(self.radius_um))

            directed: list[tuple[int, int]] = []
            weights: list[float] = []
            for source, target in pairs:
                distance = float(np.linalg.norm(chunk_coords[source] - chunk_coords[target]))
                # query_pairs may include a pair exactly on the boundary.
                if distance >= self.radius_um or distance == 0:
                    continue
                directed.extend(((source, target), (target, source)))
                weight = self.radius_um / distance
                weights.extend((weight, weight))

            edge_index = (
                torch.tensor(directed, dtype=torch.long).T.contiguous()
                if directed else torch.empty((2, 0), dtype=torch.long)
            )
            graphs.append(
                Data(
                    x=torch.tensor(expression[start:stop], dtype=torch.float32),
                    edge_index=edge_index,
                    edge_attr=torch.tensor(weights, dtype=torch.float32).reshape(-1, 1),
                    pos=torch.tensor(chunk_coords, dtype=torch.float32),
                )
            )
        return {"graphs": graphs}

    @property
    def n_markers(self) -> int:
        """Execute the n markers operation.

        Returns:
            int: The operation result."""
        return len(self._markers or [])

    @property
    def markers(self) -> tuple[str, ...]:
        """Marker order used in every graph node feature."""
        return tuple(self._markers or ())
