"""Patch-based features: cell composition / mean expression within fixed-size windows,
aggregated via MIL-style pooling.

This featurizer divides each region into a grid of square windows, computes a feature
vector for each window (cell-type fractions or mean marker expression), and then aggregates
over all windows to produce a fixed-length region-level vector.

The aggregation mimics multiple instance learning (MIL) pooling: mean, max, std, quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class PatchBasedFeaturizer(BaseFeatureExtractor):
    """Extract patch-level cell composition/expression and aggregate.

    Parameters
    ----------
    window_size_um : float, default=100
        Side length of each square window in microns.
    step_um : float, default=None
        Step size between windows (if None, equals window_size, i.e. non-overlapping).
    feature_type : {"composition", "expression"}, default="composition"
        What to compute per window: cell-type fractions or mean expression of markers.
    cell_type_col : str, default="cell_type"
        Column name for cell types (used only for composition).
    aggregations : tuple, default=("mean", "max", "std")
        Aggregation functions to apply across windows.
    quantiles : tuple, default=(0.25, 0.5, 0.75)
        Quantiles to compute if "quantile" is in aggregations.
    use_tissue_mask : bool, default=True
        If True, only windows with at least one cell inside tissue mask are used.
    min_cells_per_window : int, default=1
        Minimum number of cells required in a window to include it (others ignored).
    expression_vocab_strategy : {"union", "intersection"}, default="union"
        How to learn the marker vocabulary from training regions when
        ``feature_type="expression"``. Composition always keeps its original
        union-of-cell-types behavior.
    """

    def __init__(
        self,
        window_size_um: float = 100,
        step_um: float | None = None,
        feature_type: str = "composition",
        cell_type_col: str = "cell_type",
        aggregations: tuple[str, ...] = ("mean", "max", "std"),
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        use_tissue_mask: bool = True,
        min_cells_per_window: int = 1,
        expression_vocab_strategy: str = "union",
    ) -> None:
        """Initialize the instance.
        
                Args:
                    window_size_um (float): Window size measured in micrometers.
                    step_um (float | None): Step measured in micrometers.
                    feature_type (str): Patch feature family to calculate.
                    cell_type_col (str): Name of the column containing cell type.
                    aggregations (tuple[str, ...]): Summary operations computed for each spatial window.
                    quantiles (tuple[float, ...]): Quantile levels calculated for each feature distribution.
                    use_tissue_mask (bool): Whether to use tissue mask during processing.
                    min_cells_per_window (int): Minimum required cells per window.
        
        Args:
            window_size_um (float): Window size measured in micrometers."""
        self.window_size = window_size_um
        self.step = step_um if step_um is not None else window_size_um
        self.feature_type = feature_type
        self.cell_type_col = cell_type_col
        self.aggregations = aggregations
        self.quantiles = quantiles
        self.use_tissue_mask = use_tissue_mask
        self.min_cells = min_cells_per_window
        if expression_vocab_strategy not in {"union", "intersection"}:
            raise ValueError(
                "expression_vocab_strategy must be 'union' or 'intersection'"
            )
        self.expression_vocab_strategy = expression_vocab_strategy
        self.vocab_: list[str] = []   # cell types or markers

    # -- vocabulary -------------------------------------------------------
    def _get_vocab(self, region: RegionData) -> list[str]:
        """Return vocab.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    list[str]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.feature_type == "composition":
            col = self._col(region)
            if col is None:
                raise ValueError(f"Region {region.region_id} has no cell-type column")
            return sorted(region.cell_types[col].dropna().unique())
        else:  # expression
            return sorted(region.expression.columns.tolist())

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

    def fit(self, regions: list[RegionData]) -> "PatchBasedFeaturizer":
        # Collect vocabulary from all training regions
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'PatchBasedFeaturizer': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        vocab_sets = [set(self._get_vocab(r)) for r in regions]
        if not vocab_sets:
            vocab_set = set()
        elif self.feature_type == "expression" and self.expression_vocab_strategy == "intersection":
            vocab_set = set.intersection(*vocab_sets)
        else:
            # Preserve the original behavior for composition and for expression
            # runs that explicitly request the legacy union vocabulary.
            vocab_set = set.union(*vocab_sets)
        self.vocab_ = sorted(vocab_set)
        return self

    # -- feature names ----------------------------------------------------
    def feature_names(self) -> list[str]:
        """Execute the feature names operation.

        Returns:
            list[str]: The operation result."""
        names = []
        for agg in self.aggregations:
            if agg == "quantile":
                for q in self.quantiles:
                    for v in self.vocab_:
                        names.append(f"patch_{agg}_{q:.2f}_{v}")
            else:
                for v in self.vocab_:
                    names.append(f"patch_{agg}_{v}")
        return names

    # -- window generation ------------------------------------------------
    def _get_windows(self, region: RegionData) -> list[tuple[float, float, float, float]]:
        """Return list of (xmin, ymin, xmax, ymax) for each window.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        coords = region.coordinates[["x", "y"]].to_numpy(float)
        if len(coords) == 0:
            return []
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        # Expand slightly to include edge cells
        margin = self.window_size * 0.1
        xmin -= margin
        xmax += margin
        ymin -= margin
        ymax += margin

        windows = []
        x = xmin
        while x < xmax:
            y = ymin
            while y < ymax:
                x2 = x + self.window_size
                y2 = y + self.window_size
                windows.append((x, y, x2, y2))
                y += self.step
            x += self.step
        return windows

    # -- extract per-window feature vector -------------------------------
    def _window_feature(self, region: RegionData, win: tuple) -> np.ndarray | None:
        """Execute the window feature operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
                    win (tuple): Coordinates or contents of the current spatial window.
        
                Returns:
                    np.ndarray | None: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        xmin, ymin, xmax, ymax = win
        coords = region.coordinates[["x", "y"]].to_numpy(float)
        # Find cells inside window
        inside = (coords[:, 0] >= xmin) & (coords[:, 0] < xmax) & \
                 (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
        if not inside.any():
            return None

        # Possibly restrict to tissue mask
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(coords, "tissue")
            if in_tissue is not None:
                inside = inside & in_tissue
                if not inside.any():
                    return None

        # Count cells inside window
        if self.feature_type == "composition":
            col = self._col(region)
            if col is None:
                return None
            labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")
            cells_in_window = labels.iloc[inside]
            n = len(cells_in_window)
            if n < self.min_cells:
                return None
            # Compute fractions (map to vocab)
            counts = cells_in_window.value_counts()
            vec = np.zeros(len(self.vocab_), dtype=float)
            for i, ct in enumerate(self.vocab_):
                vec[i] = counts.get(ct, 0) / n
            return vec
        else:  # expression
            expr = region.expression.iloc[inside]  # subset of cells
            if len(expr) < self.min_cells:
                return None
            # Mean expression per marker
            mean_expr = expr.mean(axis=0)
            # Map to vocab order (markers present in training)
            vec = np.zeros(len(self.vocab_), dtype=float)
            for i, m in enumerate(self.vocab_):
                vec[i] = mean_expr.get(m, np.nan)  # if marker missing -> NaN
            return vec

    # -- extraction -------------------------------------------------------
    def extract_region(self, region: RegionData) -> dict[str, float]:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, float]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        windows = self._get_windows(region)
        if not windows:
            return {name: np.nan for name in self.feature_names()}

        # Collect window feature vectors (each is a list/array)
        win_vecs = []
        for win in windows:
            vec = self._window_feature(region, win)
            if vec is not None:
                win_vecs.append(vec)

        if len(win_vecs) == 0:
            return {name: np.nan for name in self.feature_names()}

        # Stack into matrix (n_windows x d)
        mat = np.vstack(win_vecs)  # (n_windows, d)

        # Apply aggregations
        feat_dict = {}
        for agg in self.aggregations:
            if agg == "mean":
                vals = mat.mean(axis=0)
            elif agg == "max":
                vals = mat.max(axis=0)
            elif agg == "min":
                vals = mat.min(axis=0)
            elif agg == "std":
                vals = mat.std(axis=0)
            elif agg == "quantile":
                # Compute each quantile separately
                for q in self.quantiles:
                    vals = np.quantile(mat, q, axis=0)
                    for i, v in enumerate(self.vocab_):
                        feat_dict[f"patch_quantile_{q:.2f}_{v}"] = float(vals[i])
                continue  # already added, skip generic loop
            else:
                raise ValueError(f"Unknown aggregation: {agg}")

            for i, v in enumerate(self.vocab_):
                feat_dict[f"patch_{agg}_{v}"] = float(vals[i])

        return feat_dict
