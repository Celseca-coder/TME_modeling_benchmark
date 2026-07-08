"""Mixing / infiltration features: local mixing score, Shannon entropy, etc.

Features can be global (based on cell type proportions) or local (based on k-nearest neighbors).
For local features, we compute per-cell diversity (e.g., Gini-Simpson index, entropy) and
aggregate over the region (mean, std, max, min).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class MixingFeaturizer(BaseFeatureExtractor):
    """Extract mixing statistics for cell type spatial distribution.

    Parameters
    ----------
    cell_type_col : str, default="cell_type"
        Column name for cell types.
    k_neighbors : int, default=10
        Number of nearest neighbors to consider for local mixing.
    include_global_entropy : bool, default=True
        Whether to compute Shannon entropy based on global cell type proportions.
    include_local_mixing : bool, default=True
        Whether to compute local mixing scores (diversity of neighbor types).
    use_tissue_mask : bool, default=True
        If True, restrict to cells inside tissue mask.
    """

    def __init__(
        self,
        cell_type_col: str = "cell_type",
        k_neighbors: int = 10,
        include_global_entropy: bool = True,
        include_local_mixing: bool = True,
        use_tissue_mask: bool = True,
    ) -> None:
        self.cell_type_col = cell_type_col
        self.k_neighbors = k_neighbors
        self.include_global_entropy = include_global_entropy
        self.include_local_mixing = include_local_mixing
        self.use_tissue_mask = use_tissue_mask
        self.cell_types_: list[str] = []

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "MixingFeaturizer":
        types: set[str] = set()
        for r in regions:
            col = self._col(r)
            if col is None:
                raise ValueError(f"Region {r.region_id} has no cell-type column")
            types.update(r.cell_types[col].dropna().unique())
        self.cell_types_ = sorted(types)
        return self

    # -- helper ----------------------------------------------------------
    def _compute_global_entropy(self, labels: pd.Series) -> float:
        """Shannon entropy based on cell type proportions."""
        proportions = labels.value_counts(normalize=True)
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        entropy = - (proportions * np.log(proportions)).sum()
        return float(entropy) if not np.isnan(entropy) else np.nan

    def _compute_normalized_entropy(self, labels: pd.Series) -> float:
        """Shannon entropy scaled to [0, 1] for the observed type count."""
        n_types = labels.nunique()
        if n_types < 2:
            return 0.0 if n_types == 1 else np.nan
        return self._compute_global_entropy(labels) / np.log(n_types)

    def _compute_local_mixing(
        self,
        coords: np.ndarray,
        labels: pd.Series,
        k: int,
    ) -> dict[str, float]:
        """Compute local mixing scores and aggregate."""
        n = len(coords)
        if n < 2 or k < 1:
            return {"local_mixing_mean": np.nan, "local_mixing_std": np.nan,
                    "local_mixing_min": np.nan, "local_mixing_max": np.nan}

        k = min(k, n - 1)
        if k == 0:
            return {"local_mixing_mean": np.nan, "local_mixing_std": np.nan,
                    "local_mixing_min": np.nan, "local_mixing_max": np.nan}

        tree = KDTree(coords)
        # Query k+1 neighbors (first is self)
        dists, indices = tree.query(coords, k=k + 1)
        indices = indices[:, 1:]  # exclude self
        labels_arr = labels.values

        # For each cell, compute Gini-Simpson index = 1 - sum(p_i^2)
        # where p_i are proportions of each type among the k neighbors.
        scores = []
        for i in range(n):
            neigh_labels = labels_arr[indices[i]]
            # Simpler: use pandas Series
            counts = pd.Series(neigh_labels).value_counts()
            probs = counts / k
            gini = 1 - (probs**2).sum()
            scores.append(gini)

        same_type = labels_arr[indices] == labels_arr[:, None]
        features = {
            "local_mixing_mean": float(np.mean(scores)),
            "local_mixing_std": float(np.std(scores)),
            "local_mixing_min": float(np.min(scores)),
            "local_mixing_max": float(np.max(scores)),
            "same_type_neighbor_fraction": float(np.mean(same_type)),
        }
        for source_type in self.cell_types_:
            source_mask = labels_arr == source_type
            for neighbor_type in self.cell_types_:
                key = f"neighbor_fraction__{source_type}__to__{neighbor_type}"
                features[key] = (
                    float(np.mean(labels_arr[indices[source_mask]] == neighbor_type))
                    if source_mask.any() else np.nan
                )
        return features

    # -- feature names ------------------------------------------------
    def feature_names(self) -> list[str]:
        names = []
        if self.include_global_entropy:
            names.extend(["shannon_entropy_global", "shannon_entropy_normalized"])
        if self.include_local_mixing:
            names.extend(["local_mixing_mean", "local_mixing_std", "local_mixing_min",
                          "local_mixing_max", "same_type_neighbor_fraction"])
            names.extend(
                f"neighbor_fraction__{source}__to__{neighbor}"
                for source in self.cell_types_ for neighbor in self.cell_types_
            )
        return names

    # -- extraction -------------------------------------------------------
    def extract_region(self, region: RegionData) -> dict[str, float]:
        col = self._col(region)
        if col is None:
            return {name: np.nan for name in self.feature_names()}

        # Get aligned data
        raw_coords = region.coordinates[["x", "y"]].to_numpy(float)
        coords = region.coordinates[["x", "y"]].to_numpy(float)
        labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")

        valid_labels = labels.notna().to_numpy()
        raw_coords = raw_coords[valid_labels]
        coords = coords[valid_labels]
        labels = labels.iloc[valid_labels].reset_index(drop=True)

        # Apply tissue mask if requested
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(raw_coords, "tissue")
            if in_tissue is not None:
                coords = coords[in_tissue]
                labels = labels.iloc[in_tissue].reset_index(drop=True)
                if len(coords) == 0:
                    return {name: np.nan for name in self.feature_names()}

        features = {}
        if self.include_global_entropy:
            features["shannon_entropy_global"] = self._compute_global_entropy(labels)
            features["shannon_entropy_normalized"] = self._compute_normalized_entropy(labels)
        if self.include_local_mixing:
            local = self._compute_local_mixing(coords, labels, self.k_neighbors)
            features.update(local)
        return features
