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
from .local_window import VALID_WINDOW_GROUPS, LocalWindowInstanceFeaturizer


class PatchBasedFeaturizer(BaseFeatureExtractor):
    """Extract patch-level features and aggregate with MIL-style pooling.

    Parameters
    ----------
    window_size_um : float, default=100
        Side length of each square window in microns.
    step_um : float, default=None
        Step size between windows (if None, equals window_size, i.e. non-overlapping).
    feature_type : str, default="composition"
        Legacy single-group interface. Ignored when ``feature_groups`` is supplied.
    feature_groups : tuple of local groups, optional
        ``composition``, ``expression``, ``mixing``, ``celltype_density``,
        plus the attention-style ``density`` / ``entropy`` groups.
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
    """

    VALID_GROUPS = VALID_WINDOW_GROUPS

    def __init__(
        self,
        window_size_um: float = 100,
        step_um: float | None = None,
        feature_type: str = "composition",
        feature_groups: tuple[str, ...] | None = None,
        cell_type_col: str = "cell_type",
        aggregations: tuple[str, ...] = ("mean", "max", "std"),
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        use_tissue_mask: bool = True,
        min_cells_per_window: int = 1,
    ) -> None:
        self.window_size = window_size_um
        self.step = step_um if step_um is not None else window_size_um
        self.feature_type = feature_type
        self.feature_groups = tuple(feature_groups or (feature_type,))
        unknown = set(self.feature_groups) - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
        self.cell_type_col = cell_type_col
        self.aggregations = aggregations
        self.quantiles = quantiles
        self.use_tissue_mask = use_tissue_mask
        self.min_cells = min_cells_per_window
        self.cell_types_: list[str] = []
        self.markers_: list[str] = []
        self.vocab_: list[str] = []   # prefixed output dimensions
        self.instances_: LocalWindowInstanceFeaturizer | None = None

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "PatchBasedFeaturizer":
        self.instances_ = LocalWindowInstanceFeaturizer(
            feature_groups=self.feature_groups,
            cell_type_col=self.cell_type_col,
        ).fit(regions)
        self.cell_types_ = self.instances_.cell_types_
        self.markers_ = self.instances_.markers_
        self.vocab_ = self.instances_.feature_names()
        return self

    # -- feature names ----------------------------------------------------
    def feature_names(self) -> list[str]:
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
        """Return list of (xmin, ymin, xmax, ymax) for each window."""
        coords = (region.coordinates[["x", "y"]].to_numpy(float)
                  * float(region.microns_per_pixel))
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
        xmin, ymin, xmax, ymax = win
        raw_coords = region.coordinates[["x", "y"]].to_numpy(float)
        coords = raw_coords * float(region.microns_per_pixel)
        # Find cells inside window
        inside = (coords[:, 0] >= xmin) & (coords[:, 0] < xmax) & \
                 (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
        if not inside.any():
            return None

        # Possibly restrict to tissue mask
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(raw_coords, "tissue")
            if in_tissue is not None:
                inside = inside & in_tissue
                if not inside.any():
                    return None

        if int(inside.sum()) < self.min_cells:
            return None
        if self.instances_ is None:
            raise RuntimeError("PatchBasedFeaturizer must be fitted before extraction")
        return self.instances_.extract_vector(region, inside, self.window_size)

    # -- extraction -------------------------------------------------------
    def extract_region(self, region: RegionData) -> dict[str, float]:
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
