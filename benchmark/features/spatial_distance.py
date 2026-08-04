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
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
                    k (int): Number of nearest neighbors considered.
                    aggregate_functions (tuple[str, ...]): Functions used to summarize distance measurements.
                    use_tissue_mask (bool): Whether to use tissue mask during processing.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.k = k
        self.agg_funcs = aggregate_functions
        self.use_tissue_mask = use_tissue_mask
        self.cell_types_: list[str] = []

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        """Execute the col operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    str | None: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "SpatialDistanceFeaturizer":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'SpatialDistanceFeaturizer': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
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
        """Compute aggregated distances for same and cross types.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array.
            labels (pd.Series): Cell-type or class label assigned to each observation.
            tissue_mask (np.ndarray | None): Spatial mask defining the valid tissue area."""
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

        labels_arr = labels.to_numpy()
        same_dists = np.full(n, np.nan, dtype=float)
        cross_dists = np.full(n, np.nan, dtype=float)

        # Search complete same-type and different-type candidate sets. Looking only
        # inside the k globally nearest cells can leave one category unobserved.
        for cell_type in pd.unique(labels_arr):
            source_idx = np.flatnonzero(labels_arr == cell_type)
            other_idx = np.flatnonzero(labels_arr != cell_type)
            if len(source_idx) > k:
                same_tree = KDTree(coords[source_idx])
                same_query, _ = same_tree.query(coords[source_idx], k=k + 1)
                same_dists[source_idx] = np.asarray(same_query)[:, k]
            if len(other_idx) >= k:
                cross_tree = KDTree(coords[other_idx])
                cross_query, _ = cross_tree.query(coords[source_idx], k=k)
                cross_query = np.asarray(cross_query)
                cross_dists[source_idx] = cross_query if k == 1 else cross_query[:, k - 1]

        # Aggregate
        features = {}
        for agg in self.agg_funcs:
            func = getattr(np, f"nan{agg}", None)
            if func is None:
                continue
            # Same-type distance aggregated across cells
            val_same = func(same_dists) if not np.isnan(same_dists).all() else np.nan
            val_cross = func(cross_dists) if not np.isnan(cross_dists).all() else np.nan
            features[f"same_neighbor_dist_{agg}"] = float(val_same) if not np.isnan(val_same) else np.nan
            features[f"cross_neighbor_dist_{agg}"] = float(val_cross) if not np.isnan(val_cross) else np.nan

        # Also the proportion of cells whose nearest neighbor (k=1) is same type
        if self.k == 1:
            tree = KDTree(coords)
            _, nearest_indices = tree.query(coords, k=2)
            neigh_labels = labels_arr[nearest_indices[:, 1]]
            same_count = (neigh_labels == labels_arr).sum()
            prop = same_count / n
            features["proportion_same_nearest"] = float(prop)
        else:
            # For k>1, maybe compute average proportion of same-type among k neighbors?
            # Let's add a feature: mean proportion of same-type neighbors across cells
            tree = KDTree(coords)
            _, neighbor_indices = tree.query(coords, k=k + 1)
            props = []
            for i in range(n):
                neigh_labels = labels_arr[neighbor_indices[i, 1:]]
                props.append((neigh_labels == labels_arr[i]).mean())
            features["mean_proportion_same_neighbors"] = float(np.mean(props))

        return features

    # -- feature names ----------------------------------------------------
    def feature_names(self) -> list[str]:
        """Execute the feature names operation.

        Returns:
            list[str]: The operation result."""
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
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, float]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        col = self._col(region)
        if col is None:
            return {name: np.nan for name in self.feature_names()}

        # Get coordinates and labels aligned
        raw_coords = region.coordinates[["x", "y"]].to_numpy(float)
        coords = region.coordinates[["x", "y"]].to_numpy(float)
        labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")

        valid_labels = labels.notna().to_numpy()
        raw_coords = raw_coords[valid_labels]
        coords = coords[valid_labels]
        labels = labels.iloc[valid_labels].reset_index(drop=True)

        # Determine tissue mask if requested
        tissue_mask = None
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(raw_coords, "tissue")
            if in_tissue is not None:
                tissue_mask = in_tissue

        return self._compute_distances(coords, labels, tissue_mask)


class MultiKSpatialDistanceFeaturizer(BaseFeatureExtractor):
    """Concatenate spatial-distance summaries for several neighbor orders.

    Each requested ``k`` uses the existing :class:`SpatialDistanceFeaturizer`
    unchanged. Feature names receive a ``k{value}::`` prefix so, for example,
    the first-, second-, and fifth-nearest cross-type distance summaries can be
    supplied to one downstream model simultaneously.
    """

    def __init__(
        self,
        k_values: tuple[int, ...] | list[int] = (1, 2, 5),
        cell_type_col: str = "cell_type",
        aggregate_functions: tuple[str, ...] = ("mean", "std", "min", "max", "median"),
        use_tissue_mask: bool = True,
    ) -> None:
        values = tuple(dict.fromkeys(int(k) for k in k_values))
        if not values or any(k < 1 for k in values):
            raise ValueError("k_values must contain one or more positive integers")
        self.k_values = values
        self.extractors = {
            k: SpatialDistanceFeaturizer(
                cell_type_col=cell_type_col,
                k=k,
                aggregate_functions=aggregate_functions,
                use_tissue_mask=use_tissue_mask,
            )
            for k in values
        }

    def fit(self, regions: list[RegionData]) -> "MultiKSpatialDistanceFeaturizer":
        for extractor in self.extractors.values():
            extractor.fit(regions)
        return self

    def feature_names(self) -> list[str]:
        return [
            f"k{k}::{name}"
            for k, extractor in self.extractors.items()
            for name in extractor.feature_names()
        ]

    def extract_region(self, region: RegionData) -> dict[str, float]:
        features: dict[str, float] = {}
        for k, extractor in self.extractors.items():
            features.update(
                {f"k{k}::{name}": value
                 for name, value in extractor.extract_region(region).items()}
            )
        return features
