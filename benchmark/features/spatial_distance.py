"""Spatial distance features: nearest-neighbor distances (same-type & cross-type).

Per region, for each cell type and for all cells together:
- For each cell, find distance to its k-th nearest neighbor of the same type and of a different type.
- Aggregate these distances across cells (mean, median, std, min, max, etc.).
- Also compute the proportion of cells whose nearest neighbor is of the same type (or different type).

Features are computed within the tissue foreground if a tissue mask is available, otherwise on all cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class SpatialDistanceFeaturizer(BaseFeatureExtractor):
    """Extract nearest-neighbor distance statistics for same- and cross-type interactions.

    Parameters
    ----------
    cell_type_col : str, default="cell_type"
        Column name in `region.cell_types` to use for cell type labels.
    k : int, default=1
        Order of nearest neighbor (1 = nearest).
    aggregate_functions : tuple, default=("mean", "std", "min", "max", "median")
        Aggregation functions to apply to the per-cell distances.
    use_tissue_mask : bool, default=True
        If True, only cells inside the tissue mask are used; if no tissue mask, falls back to all cells.
    """

    def __init__(
        self,
        cell_type_col: str = "cell_type",
        k: int = 1,
        aggregate_functions: tuple[str, ...] = ("mean", "std", "min", "max", "median"),
        use_tissue_mask: bool = True,
    ) -> None:
        self.cell_type_col = cell_type_col
        self.k = k
        self.agg_funcs = aggregate_functions
        self.use_tissue_mask = use_tissue_mask
        self.cell_types_: list[str] = []

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "SpatialDistanceFeaturizer":
        types: set[str] = set()
        for r in regions:
            col = self._col(r)
            if col is None:
                raise ValueError(f"Region {r.region_id} has no cell-type column")
            types.update(r.cell_types[col].dropna().unique())
        self.cell_types_ = sorted(types)
        return self

    # -- extraction helper ------------------------------------------------
    def _compute_distances(
        self,
        coords: np.ndarray,
        labels: pd.Series,
        tissue_mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute aggregated distances for same and cross types."""
        if len(coords) < 2:
            return {f"{name}": np.nan for name in self.feature_names()}

        # Apply tissue mask if provided
        if tissue_mask is not None:
            valid = tissue_mask
            if not valid.any():
                return {f"{name}": np.nan for name in self.feature_names()}
            coords = coords[valid]
            labels = labels.iloc[valid].reset_index(drop=True)

        n = len(coords)
        k = min(self.k, n - 1)
        if k == 0:
            return {f"{name}": np.nan for name in self.feature_names()}

        tree = KDTree(coords)

        # Query k+1 nearest because first is self
        dists, indices = tree.query(coords, k=k + 1)
        # dists shape (n, k+1); indices shape (n, k+1)
        # Exclude self: take columns 1..k
        dists = dists[:, 1:]  # (n, k)
        indices = indices[:, 1:]

        # For each cell, find distances to same-type neighbors and cross-type neighbors
        same_dists = []
        cross_dists = []
        labels_arr = labels.values
        for i in range(n):
            neigh_labels = labels_arr[indices[i]]
            same_mask = neigh_labels == labels_arr[i]
            same = dists[i][same_mask]
            cross = dists[i][~same_mask]
            # For each cell we take the minimum distance among the k neighbors of the desired type
            # If no such neighbor, we treat as NaN (or maybe inf)
            same_min = same.min() if len(same) > 0 else np.nan
            cross_min = cross.min() if len(cross) > 0 else np.nan
            same_dists.append(same_min)
            cross_dists.append(cross_min)

        # Aggregate
        features = {}
        for agg in self.agg_funcs:
            func = getattr(np, agg, None)
            if func is None:
                continue
            # Same-type distance aggregated across cells
            val_same = func(np.array(same_dists)) if not np.isnan(same_dists).all() else np.nan
            val_cross = func(np.array(cross_dists)) if not np.isnan(cross_dists).all() else np.nan
            features[f"same_neighbor_dist_{agg}"] = float(val_same) if not np.isnan(val_same) else np.nan
            features[f"cross_neighbor_dist_{agg}"] = float(val_cross) if not np.isnan(val_cross) else np.nan

        # Also the proportion of cells whose nearest neighbor (k=1) is same type
        if self.k == 1:
            # Use the first neighbor (index 1 from query)
            neigh_labels = labels_arr[indices[:, 0]]
            same_count = (neigh_labels == labels_arr).sum()
            prop = same_count / n
            features["proportion_same_nearest"] = float(prop)
        else:
            # For k>1, maybe compute average proportion of same-type among k neighbors?
            # Let's add a feature: mean proportion of same-type neighbors across cells
            props = []
            for i in range(n):
                neigh_labels = labels_arr[indices[i]]
                props.append((neigh_labels == labels_arr[i]).mean())
            features["mean_proportion_same_neighbors"] = float(np.mean(props))

        return features

    # -- feature names ----------------------------------------------------
    def feature_names(self) -> list[str]:
        names = []
        for agg in self.agg_funcs:
            names.append(f"same_neighbor_dist_{agg}")
            names.append(f"cross_neighbor_dist_{agg}")
        if self.k == 1:
            names.append("proportion_same_nearest")
        else:
            names.append("mean_proportion_same_neighbors")
        return names

    # -- extraction -------------------------------------------------------
    def extract_region(self, region: RegionData) -> dict[str, float]:
        col = self._col(region)
        if col is None:
            return {name: np.nan for name in self.feature_names()}

        # Get coordinates and labels aligned
        coords = region.coordinates[["x", "y"]].to_numpy(float)
        labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")

        # Determine tissue mask if requested
        tissue_mask = None
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(coords, "tissue")
            if in_tissue is not None:
                tissue_mask = in_tissue

        return self._compute_distances(coords, labels, tissue_mask)