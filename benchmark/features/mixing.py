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
    """Extract global and local cell-type mixing features from tissue regions.

    The global features describe cell-type diversity without using spatial
    positions. The local features use each cell's nearest neighbors to measure
    how strongly different cell types intermingle.

    Attributes:
        cell_type_col: Preferred column containing cell-type labels.
        k_neighbors: Number of non-self nearest neighbors used per cell.
        include_global_entropy: Whether to emit global Shannon entropy features.
        include_local_mixing: Whether to emit spatial neighbor-mixing features.
        use_tissue_mask: Whether to restrict calculations to the tissue polygon.
        cell_types_: Sorted cell-type vocabulary learned from training regions.
    """

    def __init__(
        self,
        cell_type_col: str = "cell_type",
        k_neighbors: int = 10,
        include_global_entropy: bool = True,
        include_local_mixing: bool = True,
        use_tissue_mask: bool = True,
    ) -> None:
        """Initialize the mixing feature extractor.

        Args:
            cell_type_col: Preferred column containing cell-type labels.
            k_neighbors: Number of nearest neighbors used for each cell.
            include_global_entropy: Whether to calculate global Shannon entropy.
            include_local_mixing: Whether to calculate local spatial mixing.
            use_tissue_mask: Whether to exclude cells outside the tissue polygon.
        """
        self.cell_type_col = cell_type_col
        self.k_neighbors = k_neighbors
        self.include_global_entropy = include_global_entropy
        self.include_local_mixing = include_local_mixing
        self.use_tissue_mask = use_tissue_mask
        self.cell_types_: list[str] = []

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        """Resolve the cell-type column available in a region.

        Args:
            region: Region whose cell-type table is inspected.

        Returns:
            The configured column name, the ``cell_type`` fallback, or ``None``
            when neither column exists.
        """
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "MixingFeaturizer":
        """Learn a common z**cell-type** vocabulary from training regions.

        The vocabulary determines the source-to-neighbor transition feature
        columns and is learned only from the training fold to avoid leakage.

        Args:
            regions: Training regions containing cell-type annotations.

        Returns:
            This fitted feature extractor.

        Raises:
            ValueError: If a training region has no usable cell-type column.
        """
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
        """Calculate Shannon entropy of the region-wide cell-type proportions.

        Args:
            labels: Cell-type label for every included cell.

        Returns:
            Shannon entropy in natural-log units, or ``NaN`` when undefined.
        """
        proportions = labels.value_counts(normalize=True)
        # Avoid log(0)
        proportions = proportions[proportions > 0]
        entropy = - (proportions * np.log(proportions)).sum()
        return float(entropy) if not np.isnan(entropy) else np.nan

    def _compute_normalized_entropy(self, labels: pd.Series) -> float:
        """Calculate Shannon entropy normalized by the observed type count.

        Args:
            labels: Cell-type label for every included cell.

        Returns:
            Entropy divided by ``log(number_of_observed_types)``. A homogeneous
            non-empty region returns ``0`` and an empty region returns ``NaN``.
        """
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
        """Calculate and aggregate nearest-neighbor mixing statistics.

        For every cell, the method finds its ``k`` nearest non-self neighbors
        and calculates the Gini-Simpson diversity
        ``1 - sum(cell_type_proportion ** 2)``. It also calculates the overall
        same-type neighbor fraction and every directed source-type-to-neighbor-
        type fraction.

        Args:
            coords: Two-dimensional coordinates aligned with ``labels``.
            labels: Cell-type labels aligned row-for-row with ``coords``.
            k: Requested number of non-self neighbors per cell.

        Returns:
            Mapping containing the mean, standard deviation, minimum, and
            maximum local diversity; the same-type neighbor fraction; and the
            directed cell-type neighbor fractions. Diversity summaries are
            ``NaN`` when fewer than two cells are available.
        """
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
        """Return feature names in the same schema produced by extraction.

        Returns:
            Ordered names of enabled global and local mixing features.
        """
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
        """Extract enabled mixing features for one tissue region.

        Cell labels are aligned to the coordinate index, missing labels are
        discarded, and the optional tissue polygon is applied before global
        and local statistics are calculated.

        Args:
            region: Tissue region containing coordinates and cell-type labels.

        Returns:
            Flat mapping from feature name to region-level numeric value.
            Missing cell-type columns or an empty tissue selection produce
            ``NaN`` values for the expected feature schema.
        """
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
