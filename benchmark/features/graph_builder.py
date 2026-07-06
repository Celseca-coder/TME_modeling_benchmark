"""SPACE-GM style graph construction: Delaunay triangulation edges (pruned by
a max edge length) + one-hot cell-type node features (+ optional marker
expression channels).

Unlike density/mixing/point_pattern, `extract_region` does NOT return
`dict[str, float]` in spirit — it returns `{"graph": data}` where `data` is a
`torch_geometric.data.Data` object. `BaseFeatureExtractor.transform` still
works unchanged (it just builds a DataFrame with one object-dtype column
instead of many float columns), so this plugs into `cross_validate` /
`cohort_split_test` with zero changes to crossval.py. The matching models —
`SpaceGMClassifier` / `SpaceGMCox` in `benchmark/models/space_gm.py` — unpack
that single column instead of calling `_TabularModel`'s scaler/imputer (there
is nothing to standardise in a graph object).

ASSUMPTION (only confirmed via `benchmark/models/mil.py`'s usage): RegionData
exposes `region.coordinates` (DataFrame with `coord_cols`, indexed by cell id)
and `region.cell_types` (DataFrame with `cell_type_col`). Marker expression is
assumed to live on `region.expression` with the same index — adjust
`_node_features` if your actual attribute names differ.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class SpaceGMGraphBuilder(BaseFeatureExtractor):
    def __init__(self, cell_type_col: str = "cell_type", coord_cols=("x", "y"),
                 marker_cols: list[str] | None = None,
                 max_edge_length: float | None = 50.0, min_cells: int = 4):
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.marker_cols = marker_cols or []
        self.max_edge_length = max_edge_length
        self.min_cells = min_cells
        self._cell_types: list[str] | None = None

    def fit(self, regions: list[RegionData]) -> "SpaceGMGraphBuilder":
        # 固定全局细胞类型词表 -> 保证train/val的one-hot维度一致,不会因为某个
        # fold里缺某个细胞类型而导致node feature维度对不齐
        types = set()
        for r in regions:
            types.update(r.cell_types[self.cell_type_col].dropna().unique().tolist())
        self._cell_types = sorted(types)
        return self

    def extract_region(self, region: RegionData) -> dict:
        assert self._cell_types is not None, "call fit() before extract_region()"
        coords = region.coordinates[self.coord_cols].to_numpy(dtype=float)
        n = len(coords)
        if n < self.min_cells:
            # 细胞太少建不出有意义的图:返回一个形状正确的空图,而不是报错让
            # cross_validate里的try/except把它吞成一整行NaN
            x = torch.zeros((0, self._n_node_features()), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return {"graph": Data(x=x, edge_index=edge_index)}

        edge_index = self._build_edges(coords)
        x = self._node_features(region, n)
        return {"graph": Data(x=x, edge_index=edge_index,
                              pos=torch.as_tensor(coords, dtype=torch.float32))}

    # ------------------------------------------------------------------
    def _n_node_features(self) -> int:
        return len(self._cell_types) + len(self.marker_cols)

    def _node_features(self, region: RegionData, n: int) -> torch.Tensor:
        type_to_int = {t: i for i, t in enumerate(self._cell_types)}
        labels = region.cell_types[self.cell_type_col].to_numpy()
        onehot = np.zeros((n, len(self._cell_types)), dtype=np.float32)
        for i, t in enumerate(labels):
            j = type_to_int.get(t)
            if j is not None:
                onehot[i, j] = 1.0

        if self.marker_cols:
            expr = region.expression[self.marker_cols].to_numpy(dtype=np.float32)
            feats = np.concatenate([onehot, np.nan_to_num(expr, nan=0.0)], axis=1)
        else:
            feats = onehot
        return torch.as_tensor(feats, dtype=torch.float32)

    def _build_edges(self, coords: np.ndarray) -> torch.Tensor:
        n = len(coords)
        if n < 3:  # Delaunay至少要3个不共线的点
            edges = np.array([[0, 1]]) if n >= 2 else np.empty((0, 2), dtype=int)
        else:
            tri = Delaunay(coords)
            edge_set = set()
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
                    edge_set.add((min(a, b), max(a, b)))
            edges = np.array(sorted(edge_set))

        if self.max_edge_length is not None and len(edges):
            d = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
            edges = edges[d <= self.max_edge_length]
            if len(edges) == 0:          # 所有边都被剪掉了:退化成自环,保证不空
                edges = np.array([[0, 0]])

        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T
        return torch.as_tensor(edge_index, dtype=torch.long)