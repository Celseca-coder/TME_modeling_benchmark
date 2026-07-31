"""Cyto-Community graph construction.

The builder represents each tissue region as one cell graph. Nodes are cells,
edges connect local spatial neighbours, and node features contain a train-fold
cell-type vocabulary plus the shared marker-expression panel. The matching
``benchmark.models.cyto_community`` model learns soft cell-to-community
assignments and pools communities into one region-level prediction.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class CytoCommunityGraphBuilder(BaseFeatureExtractor):
    """Represent cyto community graph builder."""
    def __init__(
        self,
        cell_type_col: str = "cell_type",
        coord_cols=("x", "y"),
        include_expression: bool = True,
        radius_um: float = 50.0,
        k_neighbors: int = 8,
        max_cells: int | None = 4096,
        seed: int = 0,
    ) -> None:
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
                    coord_cols (Any): Names of columns containing coord.
                    include_expression (bool): Whether to include expression in the output.
                    radius_um (float): Radius measured in micrometers.
                    k_neighbors (int): Value controlling or representing k neighbors.
                    max_cells (int | None): Maximum allowed cells.
                    seed (int): Random seed used for reproducibility.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.include_expression = include_expression
        self.radius_um = radius_um
        self.k_neighbors = k_neighbors
        self.max_cells = max_cells
        self.seed = seed
        self._cell_types: list[str] = []
        self._markers: list[str] = []

    def fit(self, regions: list[RegionData]) -> "CytoCommunityGraphBuilder":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'CytoCommunityGraphBuilder': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        types: set[str] = set()
        marker_sets = []
        for region in regions:
            col = self._cell_type_column(region)
            types.update(region.cell_types[col].dropna().astype(str).unique())
            if self.include_expression:
                marker_sets.append(set(region.expression.columns))
        self._cell_types = sorted(types)
        self._markers = sorted(set.intersection(*marker_sets)) if marker_sets else []
        return self

    def _cell_type_column(self, region: RegionData) -> str:
        """Execute the cell type column operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    str: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        raise ValueError(f"Region {region.region_id} has no cell-type column")

    def _coords_um(self, region: RegionData) -> np.ndarray:
        """Execute the coords um operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        coords_um = getattr(region, "coordinates_um", None)
        if coords_um is not None:
            return coords_um[self.coord_cols].to_numpy(float)
        coords = region.coordinates[self.coord_cols].to_numpy(float)
        return coords * float(getattr(region, "microns_per_pixel", 1.0))

    def _sample_cells(self, region: RegionData, n: int) -> np.ndarray:
        """Execute the sample cells operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    n (int): Value controlling or representing n.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.max_cells is None or n <= self.max_cells:
            return np.arange(n)
        stable = sum(ord(ch) for ch in str(region.region_id))
        rng = np.random.default_rng(self.seed + stable)
        return np.sort(rng.choice(n, self.max_cells, replace=False))

    def _node_features(self, region: RegionData, keep: np.ndarray) -> np.ndarray:
        """Execute the node features operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    keep (np.ndarray): Boolean mask or indices selecting retained items.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        labels = region.cell_types[self._cell_type_column(region)].reindex(region.coordinates.index)
        type_map = {name: i + 1 for i, name in enumerate(self._cell_types)}
        type_ids = np.array([type_map.get(str(x), 0) for x in labels.iloc[keep]], dtype=int)
        one_hot = np.zeros((len(keep), len(self._cell_types) + 1), dtype=np.float32)
        if len(keep):
            one_hot[np.arange(len(keep)), type_ids] = 1.0

        parts = [one_hot]
        if self.include_expression and self._markers:
            expr = region.expression.reindex(region.coordinates.index).iloc[keep]
            expr = expr.reindex(columns=self._markers).to_numpy(np.float32)
            parts.append(np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _edge_index(self, coords: np.ndarray) -> torch.Tensor:
        """Execute the edge index operation.
        
                Args:
                    coords (np.ndarray): Two-dimensional cell-coordinate array.
        
                Returns:
                    torch.Tensor: The operation result.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array."""
        n = len(coords)
        if n < 2:
            return torch.zeros((2, 0), dtype=torch.long)

        tree = cKDTree(coords)
        edge_set: set[tuple[int, int]] = set()
        k = min(self.k_neighbors + 1, n)
        distances, neighbours = tree.query(coords, k=k)
        if k == 1:
            neighbours = neighbours[:, None]
            distances = distances[:, None]
        for i in range(n):
            for dist, j in zip(distances[i, 1:], neighbours[i, 1:]):
                if np.isfinite(dist) and dist <= self.radius_um:
                    a, b = int(i), int(j)
                    edge_set.add((a, b))
                    edge_set.add((b, a))

        if not edge_set:
            return torch.zeros((2, 0), dtype=torch.long)
        edges = np.asarray(sorted(edge_set), dtype=np.int64)
        return torch.as_tensor(edges.T, dtype=torch.long)

    def extract_region(self, region: RegionData) -> dict:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        assert self._cell_types is not None, "call fit() before extract_region()"
        coords = self._coords_um(region)
        keep = self._sample_cells(region, len(coords))
        coords = coords[keep]
        x = self._node_features(region, keep)
        graph = Data(
            x=torch.as_tensor(x, dtype=torch.float32),
            edge_index=self._edge_index(coords),
            pos=torch.as_tensor(coords, dtype=torch.float32),
        )
        return {"graph": graph}

    @property
    def n_cell_types(self) -> int:
        """Execute the n cell types operation.

        Returns:
            int: The operation result."""
        return len(self._cell_types)

    @property
    def n_markers(self) -> int:
        """Execute the n markers operation.

        Returns:
            int: The operation result."""
        return len(self._markers)
