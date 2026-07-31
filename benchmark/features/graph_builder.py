"""SPACE-GM microenvironment graph construction.

Each region is represented by one local graph per sampled centre cell.  A graph
contains the centre's three-hop Delaunay neighbourhood, restricted to 75 um.
Node inputs contain cell-type IDs plus cell-size/centre indicators; protein
expression is stored only as a pre-training target and never used as input.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError
from torch_geometric.data import Data

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class SpaceGMGraphBuilder(BaseFeatureExtractor):
    """Represent space g m graph builder."""
    def __init__(self, cell_type_col: str = "cell_type", coord_cols=("x", "y"),
                 cell_size_col: str | None = None, n_hops: int = 3,
                 radius_um: float = 75.0, near_edge_um: float = 20.0,
                 max_centers: int | None = 256, seed: int = 0):
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
                    coord_cols (Any): Names of columns containing coord.
                    cell_size_col (str | None): Name of the column containing cell size.
                    n_hops (int): Number of hops.
                    radius_um (float): Radius measured in micrometers.
                    near_edge_um (float): Near edge measured in micrometers.
                    max_centers (int | None): Maximum allowed centers.
                    seed (int): Random seed used for reproducibility.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.cell_size_col = cell_size_col
        self.n_hops = n_hops
        self.radius_um = radius_um
        self.near_edge_um = near_edge_um
        self.max_centers = max_centers
        self.seed = seed
        self._cell_types: list[str] | None = None
        self._markers: list[str] = []
        self._size_min = 0.0
        self._size_max = 1.0

    def fit(self, regions: list[RegionData]) -> "SpaceGMGraphBuilder":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'SpaceGMGraphBuilder': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        types: set[str] = set()
        marker_sets = []
        sizes = []
        for region in regions:
            col = self._cell_type_column(region)
            types.update(region.cell_types[col].dropna().astype(str).unique())
            marker_sets.append(set(region.expression.columns))
            values = self._raw_sizes(region)
            if values is not None:
                sizes.append(values[np.isfinite(values) & (values >= 0)])
        self._cell_types = sorted(types)
        self._markers = sorted(set.intersection(*marker_sets)) if marker_sets else []
        if sizes and any(len(x) for x in sizes):
            logged = np.log1p(np.concatenate([x for x in sizes if len(x)]))
            self._size_min, self._size_max = float(logged.min()), float(logged.max())
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

    def _raw_sizes(self, region: RegionData) -> np.ndarray | None:
        """Execute the raw sizes operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    np.ndarray | None: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        candidates = ([self.cell_size_col] if self.cell_size_col else []) + [
            "cell_size", "cell_area", "area", "size"
        ]
        for col in candidates:
            if col and col in region.cell_types.columns:
                return region.cell_types[col].reindex(region.coordinates.index).to_numpy(float)
            if col and col in region.expression.columns:
                return region.expression[col].reindex(region.coordinates.index).to_numpy(float)
        return None

    def _scaled_sizes(self, region: RegionData, n: int) -> np.ndarray:
        """Execute the scaled sizes operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    n (int): Value controlling or representing n.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        raw = self._raw_sizes(region)
        if raw is None:
            return np.zeros(n, dtype=np.float32)
        logged = np.log1p(np.clip(np.nan_to_num(raw, nan=0.0), 0, None))
        span = self._size_max - self._size_min
        return np.clip((logged - self._size_min) / span, 0, 1).astype(np.float32) if span > 0 else np.zeros(n, np.float32)

    @staticmethod
    def _delaunay_edges(coords: np.ndarray) -> np.ndarray:
        """Execute the delaunay edges operation.
        
                Args:
                    coords (np.ndarray): Two-dimensional cell-coordinate array.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array."""
        n = len(coords)
        if n < 2:
            return np.empty((0, 2), dtype=int)
        if n == 2:
            return np.array([[0, 1]], dtype=int)
        try:
            simplices = Delaunay(coords).simplices
            edges = {tuple(sorted((int(s[i]), int(s[(i + 1) % 3]))))
                     for s in simplices for i in range(3)}
            return np.asarray(sorted(edges), dtype=int)
        except QhullError:
            order = np.argsort(coords[:, 0] + coords[:, 1] * 1e-9)
            return np.column_stack([order[:-1], order[1:]])

    def _hop_neighbourhood(self, center: int, adjacency: list[set[int]]) -> set[int]:
        """Execute the hop neighbourhood operation.
        
                Args:
                    center (int): Coordinates or index of the center cell.
                    adjacency (list[set[int]]): Graph adjacency matrix describing connections between cells.
        
                Returns:
                    set[int]: The operation result.
        
        Args:
            center (int): Coordinates or index of the center cell."""
        seen = {center}
        queue = deque([(center, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth == self.n_hops:
                continue
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, depth + 1))
        return seen

    def extract_region(self, region: RegionData) -> dict:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        assert self._cell_types is not None, "call fit() before extract_region()"
        coords = region.coordinates_um[self.coord_cols].to_numpy(float)
        n = len(coords)
        if n == 0:
            return {"graphs": []}

        base_edges = self._delaunay_edges(coords)
        adjacency = [set() for _ in range(n)]
        for a, b in base_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)

        centers = np.arange(n)
        if self.max_centers is not None and n > self.max_centers:
            rng = np.random.default_rng(self.seed)
            centers = np.sort(rng.choice(n, self.max_centers, replace=False))

        labels = region.cell_types[self._cell_type_column(region)].reindex(region.coordinates.index)
        # Index 0 is reserved for labels not observed in the training fold.
        type_map = {name: i + 1 for i, name in enumerate(self._cell_types)}
        type_ids = np.array([type_map.get(str(x), 0) for x in labels], dtype=np.int64)
        sizes = self._scaled_sizes(region, n)
        expression = (
            region.expression
            .reindex(index=region.coordinates.index, columns=self._markers)
            .to_numpy(np.float32)
        )

        graphs = []
        for center in centers:
            nodes = self._hop_neighbourhood(int(center), adjacency)
            distance = np.linalg.norm(coords - coords[center], axis=1)
            nodes = sorted(i for i in nodes if distance[i] <= self.radius_um)
            if center not in nodes:
                nodes.append(int(center)); nodes.sort()
            local = {old: new for new, old in enumerate(nodes)}

            directed_edges = []
            edge_types = []
            for a, b in base_edges:
                if a in local and b in local:
                    edge_type = 0 if np.linalg.norm(coords[a] - coords[b]) < self.near_edge_um else 1
                    directed_edges.extend([(local[a], local[b]), (local[b], local[a])])
                    edge_types.extend([edge_type, edge_type])
            for old in nodes:
                directed_edges.append((local[old], local[old]))
                edge_types.append(2)

            edge_index = torch.tensor(directed_edges, dtype=torch.long).T.contiguous()
            center_flag = np.array([old == center for old in nodes], dtype=np.float32)
            aux = np.column_stack([sizes[nodes], center_flag]).astype(np.float32)
            graphs.append(Data(
                cell_type=torch.tensor(type_ids[nodes], dtype=torch.long),
                aux=torch.tensor(aux, dtype=torch.float32),
                edge_index=edge_index,
                edge_type=torch.tensor(edge_types, dtype=torch.long),
                center_index=torch.tensor(local[int(center)], dtype=torch.long),
                center_expression=torch.tensor(np.nan_to_num(expression[center]), dtype=torch.float32),
                pos=torch.tensor(coords[nodes], dtype=torch.float32),
            ))
        return {"graphs": graphs}

    @property
    def n_cell_types(self) -> int:
        """Execute the n cell types operation.

        Returns:
            int: The operation result."""
        return len(self._cell_types or [])

    @property
    def n_markers(self) -> int:
        """Execute the n markers operation.

        Returns:
            int: The operation result."""
        return len(self._markers)
