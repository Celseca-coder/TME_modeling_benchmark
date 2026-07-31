"""SORBET-style cell-neighborhood graph construction.

SORBET models a tissue sample as a graph of neighboring cells, samples local
cell neighborhoods, embeds each neighborhood with a GNN, and aggregates the
neighborhood evidence into a sample-level prediction. This builder creates the
local subgraphs used by ``benchmark.models.sorbet``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class SORBETGraphBuilder(BaseFeatureExtractor):
    """Represent s o r b e t graph builder."""
    def __init__(
        self,
        cell_type_col: str = "cell_type",
        coord_cols=("x", "y"),
        include_expression: bool = True,
        radius_um: float = 50.0,
        k_neighbors: int = 12,
        max_centers: int = 192,
        max_nodes_per_subgraph: int = 96,
        seed: int = 0,
    ) -> None:
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
                    coord_cols (Any): Names of columns containing coord.
                    include_expression (bool): Whether to include expression in the output.
                    radius_um (float): Radius measured in micrometers.
                    k_neighbors (int): Value controlling or representing k neighbors.
                    max_centers (int): Maximum allowed centers.
                    max_nodes_per_subgraph (int): Maximum allowed nodes per subgraph.
                    seed (int): Random seed used for reproducibility.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.include_expression = include_expression
        self.radius_um = radius_um
        self.k_neighbors = k_neighbors
        self.max_centers = max_centers
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.seed = seed
        self._cell_types: list[str] = []
        self._markers: list[str] = []

    def fit(self, regions: list[RegionData]) -> "SORBETGraphBuilder":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'SORBETGraphBuilder': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        cell_types: set[str] = set()
        marker_sets: list[set[str]] = []
        for region in regions:
            col = self._cell_type_column(region)
            cell_types.update(region.cell_types[col].dropna().astype(str).unique())
            if self.include_expression:
                marker_sets.append(set(region.expression.columns))
        self._cell_types = sorted(cell_types)
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

    def _stable_rng(self, region_id: str) -> np.random.Generator:
        """Execute the stable rng operation.
        
                Args:
                    region_id (str): Unique identifier of a tissue region.
        
                Returns:
                    np.random.Generator: The operation result.
        
        Args:
            region_id (str): Unique identifier of a tissue region."""
        stable = sum(ord(ch) for ch in str(region_id))
        return np.random.default_rng(self.seed + stable)

    def _node_features(self, region: RegionData, node_idx: np.ndarray) -> np.ndarray:
        """Execute the node features operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    node_idx (np.ndarray): Index of the node used to construct a subgraph.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        labels = region.cell_types[self._cell_type_column(region)].reindex(region.coordinates.index)
        type_map = {name: i + 1 for i, name in enumerate(self._cell_types)}
        type_ids = np.array([type_map.get(str(x), 0) for x in labels.iloc[node_idx]], dtype=int)
        one_hot = np.zeros((len(node_idx), len(self._cell_types) + 1), dtype=np.float32)
        if len(node_idx):
            one_hot[np.arange(len(node_idx)), type_ids] = 1.0

        parts = [one_hot]
        if self.include_expression and self._markers:
            expr = region.expression.reindex(region.coordinates.index).iloc[node_idx]
            expr = expr.reindex(columns=self._markers).to_numpy(np.float32)
            parts.append(np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _subgraph_edges(self, coords: np.ndarray) -> torch.Tensor:
        """Execute the subgraph edges operation.
        
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
        k = min(self.k_neighbors + 1, n)
        distances, neighbours = tree.query(coords, k=k)
        if k == 1:
            distances = distances[:, None]
            neighbours = neighbours[:, None]
        edge_set: set[tuple[int, int]] = set()
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

    def _center_indices(self, region: RegionData, n_cells: int) -> np.ndarray:
        """Execute the center indices operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    n_cells (int): Number of cells.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if n_cells == 0:
            return np.asarray([], dtype=int)
        if n_cells <= self.max_centers:
            return np.arange(n_cells)
        rng = self._stable_rng(region.region_id)
        return np.sort(rng.choice(n_cells, self.max_centers, replace=False))

    def _neighborhood_nodes(self, tree: cKDTree, coords: np.ndarray, center: int,
                            rng: np.random.Generator) -> np.ndarray:
        """Execute the neighborhood nodes operation.
        
                Args:
                    tree (cKDTree): Spatial search tree used for neighbor queries.
                    coords (np.ndarray): Two-dimensional cell-coordinate array.
                    center (int): Coordinates or index of the center cell.
                    rng (np.random.Generator): Random-number generator used for reproducible sampling.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            tree (cKDTree): Spatial search tree used for neighbor queries."""
        nodes = tree.query_ball_point(coords[center], r=self.radius_um)
        nodes = np.asarray(sorted(set(nodes + [center])), dtype=int)
        if len(nodes) > self.max_nodes_per_subgraph:
            others = nodes[nodes != center]
            keep = rng.choice(others, self.max_nodes_per_subgraph - 1, replace=False)
            nodes = np.sort(np.concatenate([[center], keep]))
        return nodes

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
        n_cells = len(coords)
        if n_cells == 0:
            return {"graphs": []}
        tree = cKDTree(coords)
        rng = self._stable_rng(region.region_id)
        graphs: list[Data] = []
        for center in self._center_indices(region, n_cells):
            nodes = self._neighborhood_nodes(tree, coords, int(center), rng)
            local_coords = coords[nodes]
            x = self._node_features(region, nodes)
            center_local = int(np.where(nodes == center)[0][0])
            center_mask = np.zeros(len(nodes), dtype=np.float32)
            center_mask[center_local] = 1.0
            graph = Data(
                x=torch.as_tensor(x, dtype=torch.float32),
                edge_index=self._subgraph_edges(local_coords),
                pos=torch.as_tensor(local_coords, dtype=torch.float32),
                center_mask=torch.as_tensor(center_mask, dtype=torch.float32),
            )
            graphs.append(graph)
        return {"graphs": graphs}
