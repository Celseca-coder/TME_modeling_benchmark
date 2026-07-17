"""Build genuine SPACE-GM cellular graphs (networkx) for each region.

This featurizer uses the real ``spacegm`` package's graph-construction utility
(:func:`spacegm.graph_build.assign_attributes`) so the graphs are exactly what
:class:`spacegm.data.CellularGraphDataset` expects.  It is paired with the
runners in :mod:`benchmark.models.space_gm_real_cv`, which drive the authentic
SPACE-GM training / inference loops.

Design notes
------------
* Coordinates are converted to **microns** (``coordinates × microns_per_pixel``)
  so the SPACE-GM radius / edge-cutoff parameters are physical and comparable
  across datasets.  Edge cutoffs are therefore evaluated with ``um_per_pixel=1``.
* Cell-type vocabulary is discovered on the **training** regions only (in
  :meth:`fit`).  Any cell type unseen in training — or missing — is remapped to
  ``"Unassigned"`` at transform time, so the validation/test graphs never
  introduce an unknown category (which would break feature processing).
* Protein expression is z-scored per marker on the **training** statistics and
  stored on the graph; the downstream ``CellularGraphDataset`` applies the
  SPACE-GM ``linear`` transform (clip to ±3 → min-max to [0, 1]).  Fitting the
  scaler on train only keeps the validation split leakage-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import spacegm as sg

from benchmark.data.dataset import RegionData
from benchmark.features.base import BaseFeatureExtractor

UNASSIGNED = "Unassigned"

# SPACE-GM node/edge feature items produced by this builder (see
# spacegm.features.process_feature). Kept in sync with the model wrapper.
NODE_FEATURES = ["cell_type", "biomarker_expression", "neighborhood_composition", "center_coord"]
EDGE_FEATURES = ["edge_type", "distance"]


def coords_in_microns(region: RegionData, coord_cols=("x", "y")) -> np.ndarray:
    """Return the region's cell centroids in microns (Nx2).

    Prefers a ``coordinates_um`` attribute when present, else scales the raw
    pixel coordinates by ``microns_per_pixel``.
    """
    cu = getattr(region, "coordinates_um", None)
    if cu is not None:
        return cu[list(coord_cols)].to_numpy(float)
    px = region.coordinates[list(coord_cols)].to_numpy(float)
    return px * float(region.microns_per_pixel)


class SpaceGMGraphBuilder(BaseFeatureExtractor):
    """Featurizer emitting one SPACE-GM ``networkx`` graph per region.

    ``transform`` returns a DataFrame indexed by ``region_id`` with a single
    ``nx_graph`` object column (plus ``n_cells``).  The fitted cell-type / marker
    vocabulary is also attached to ``df.attrs`` for convenience, but the paired
    model re-derives it from the training graphs so it never depends on ``attrs``
    surviving pandas operations.
    """

    def __init__(
        self,
        cell_type_col: str = "cell_type",
        coord_cols=("x", "y"),
        near_edge_um: float = 20.0,
        expression_clip: float = 3.0,
    ):
        self.cell_type_col = cell_type_col
        self.coord_cols = tuple(coord_cols)
        self.near_edge_um = float(near_edge_um)
        self.expression_clip = float(expression_clip)

        # Learned on fit():
        self._train_types: list[str] = []
        self.biomarkers: list[str] = []
        self._bm_mean: pd.Series | None = None
        self._bm_std: pd.Series | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def _cell_type_column(self, region: RegionData) -> str:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        raise ValueError(f"Region {region.region_id} has no cell-type column")

    def fit(self, regions: list[RegionData]) -> "SpaceGMGraphBuilder":
        types: set[str] = set()
        marker_sets: list[set[str]] = []
        # Streaming mean/std accumulation per marker (over the shared marker set).
        for region in regions:
            col = self._cell_type_column(region)
            types.update(region.cell_types[col].dropna().astype(str).unique())
            marker_sets.append(set(region.expression.columns))

        self._train_types = sorted(types)
        self.biomarkers = sorted(set.intersection(*marker_sets)) if marker_sets else []

        # Per-marker z-score statistics over all training cells.
        if self.biomarkers:
            sums = pd.Series(0.0, index=self.biomarkers)
            sqs = pd.Series(0.0, index=self.biomarkers)
            count = 0
            for region in regions:
                expr = region.expression[self.biomarkers].to_numpy(float)
                expr = np.nan_to_num(expr, nan=0.0)
                sums += expr.sum(axis=0)
                sqs += (expr ** 2).sum(axis=0)
                count += expr.shape[0]
            count = max(count, 1)
            mean = sums / count
            var = np.maximum(sqs / count - mean ** 2, 0.0)
            std = np.sqrt(var)
            std[std < 1e-8] = 1.0
            self._bm_mean, self._bm_std = mean, std
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def _cell_data_frame(self, region: RegionData) -> pd.DataFrame:
        """Assemble the SPACE-GM ``cell_data`` DataFrame for one region."""
        idx = region.coordinates.index
        coords = coords_in_microns(region, self.coord_cols)

        col = self._cell_type_column(region)
        labels = region.cell_types[col].reindex(idx).astype("object")
        known = set(self._train_types)
        labels = labels.map(lambda v: str(v) if (v == v and str(v) in known) else UNASSIGNED)

        data = {
            "CELL_ID": np.asarray(idx),
            "X": coords[:, 0],
            "Y": coords[:, 1],
            "CELL_TYPE": labels.to_numpy(),
        }
        if self.biomarkers:
            expr = region.expression[self.biomarkers].reindex(idx)
            z = (expr - self._bm_mean) / self._bm_std
            z = z.to_numpy(float)
            z = np.nan_to_num(z, nan=0.0)
            for j, bm in enumerate(self.biomarkers):
                data[f"BM-{bm}"] = z[:, j]
        return pd.DataFrame(data)

    def _build_nx_graph(self, region: RegionData):
        """Delaunay cellular graph with SPACE-GM node/edge attributes."""
        import networkx as nx
        from scipy.spatial import Delaunay, QhullError

        cell_data = self._cell_data_frame(region)
        coords = cell_data[["X", "Y"]].to_numpy(float)
        n = len(coords)

        # Robust Delaunay edges (falls back to a coordinate-ordered chain, mirroring
        # benchmark.features.graph_builder, so a degenerate region never raises).
        if n < 2:
            edges = np.empty((0, 2), dtype=int)
        elif n == 2:
            edges = np.array([[0, 1]], dtype=int)
        else:
            try:
                simplices = Delaunay(coords).simplices
                edge_set = {
                    tuple(sorted((int(s[i]), int(s[(i + 1) % 3]))))
                    for s in simplices for i in range(3)
                }
                edges = np.asarray(sorted(edge_set), dtype=int)
            except QhullError:
                order = np.argsort(coords[:, 0] + coords[:, 1] * 1e-9)
                edges = np.column_stack([order[:-1], order[1:]])

        G = nx.Graph()
        node_to_cell = {}
        for i, cid in enumerate(cell_data["CELL_ID"].tolist()):
            G.add_node(i, voronoi_polygon=None)
            node_to_cell[i] = cid
        for a, b in edges:
            G.add_edge(int(a), int(b))

        # assign_attributes populates cell_type / center_coord / biomarker_expression
        # on nodes and distance / edge_type on edges, exactly as SPACE-GM expects.
        G = sg.graph_build.assign_attributes(
            G, cell_data, node_to_cell,
            edge_kwargs={"neighbor_edge_cutoff": self.near_edge_um, "um_per_pixel": 1.0},
        )
        G.region_id = str(region.region_id)
        return G

    def extract_region(self, region: RegionData) -> dict:
        G = self._build_nx_graph(region)
        return {"nx_graph": G, "n_cells": int(G.number_of_nodes())}

    def transform(self, regions: list[RegionData]) -> pd.DataFrame:
        print("Building graphs for %d regions..." % len(regions))
        df = super().transform(regions)
        # Convenience metadata (the model does not rely on these surviving).
        df.attrs["biomarkers"] = list(self.biomarkers)
        df.attrs["cell_types"] = list(self._train_types)
        df.attrs["node_features"] = list(NODE_FEATURES)
        df.attrs["edge_features"] = list(EDGE_FEATURES)
        return df
