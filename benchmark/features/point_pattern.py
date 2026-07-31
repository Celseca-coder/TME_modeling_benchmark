"""Point pattern statistics: Ripley's K, Besag's L, pair correlation function (pcf), variogram.

Per region, for all cells (or per cell type), evaluate these summary functions
at specified radii. Features are the values of the functions at each radius.

If `by_type` is True, features are computed separately for each cell type's pattern,
otherwise all cells are pooled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, ConvexHull
from scipy.stats import gaussian_kde  # 新增导入

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class PointPatternFeaturizer(BaseFeatureExtractor):
    """Extract point process summary statistics at specified radii.

    Parameters
    ----------
    radii : list[float], default=[10, 20, 50, 100, 200]
        Radii (in microns) at which to evaluate functions.
    metrics : tuple, default=("K", "L")
        Changed default to only "K" and "L" to avoid costly pcf/variogram.
        To include them, pass e.g. metrics=("K","L","pcf","variogram").
    by_type : bool, default=False
        If True, compute separately for each cell type (using `cell_type_col`).
    cell_type_col : str, default="cell_type"
        Column name for cell types (used only if `by_type=True`).
    use_tissue_mask : bool, default=True
        If True, use only cells inside tissue mask; if no mask, fallback to all.
    edge_correction : bool, default=False
        Whether to apply border correction (currently a placeholder).
    pcf_bandwidth : float, default=None
        Bandwidth for pcf estimation (if None, uses min(radii)/2).
    """

    def __init__(
        self,
        radii: list[float] | None = None,
        metrics: tuple[str, ...] = ("K", "L"),  # 默认只计算 K 和 L
        by_type: bool = False,
        cell_type_col: str = "cell_type",
        use_tissue_mask: bool = True,
        edge_correction: bool = False,
        pcf_bandwidth: float | None = None,
    ) -> None:
        """Initialize the instance.
        
                Args:
                    radii (list[float] | None): Value controlling or representing radii.
                    metrics (tuple[str, ...]): Spatial statistics to calculate.
                    by_type (bool): Whether to compute statistics separately for each cell type.
                    cell_type_col (str): Name of the column containing cell type.
                    use_tissue_mask (bool): Whether to use tissue mask during processing.
                    edge_correction (bool): Value controlling or representing edge correction.
                    pcf_bandwidth (float | None): Value controlling or representing pcf bandwidth.
        
        Args:
            radii (list[float] | None): Value controlling or representing radii."""
        self.radii = radii if radii is not None else [10, 20, 50, 100, 200]
        self.metrics = metrics
        self.by_type = by_type
        self.cell_type_col = cell_type_col
        self.use_tissue_mask = use_tissue_mask
        self.edge_correction = edge_correction
        self.pcf_bandwidth = pcf_bandwidth
        self.cell_types_: list[str] = []

    # -- vocabulary (needed only if by_type) ------------------------------
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

    def fit(self, regions: list[RegionData]) -> "PointPatternFeaturizer":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'PointPatternFeaturizer': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        if self.by_type:
            types: set[str] = set()
            for r in regions:
                col = self._col(r)
                if col is None:
                    raise ValueError(f"Region {r.region_id} has no cell-type column")
                types.update(r.cell_types[col].dropna().unique())
            self.cell_types_ = sorted(types)
        else:
            self.cell_types_ = []
        return self

    # -- helper: area ----------------------------------------------------
    def _get_area_um2(self, region: RegionData, coords: np.ndarray) -> float | None:
        """Return area in mm²: use tissue area if available, else convex hull, else bounding box.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed.
            coords (np.ndarray): Two-dimensional cell-coordinate array."""
        if region.tissue_area_mm2 is not None and region.tissue_area_mm2 > 0:
            return region.tissue_area_mm2 * 1e6
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                return float(hull.volume)
            except Exception:
                pass
        if len(coords) > 0:
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
            area_um2 = (xmax - xmin) * (ymax - ymin)
            return float(area_um2)
        return None

    # -- compute Ripley's K and L ----------------------------------------
    def _compute_ripley(
        self,
        coords: np.ndarray,
        area_um2: float,
        radii: list[float],
    ) -> dict[str, float]:
        """Compute K(r) and L(r). Returns dict with keys like 'K_r10'.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array.
            area_um2 (float): Area measured in square micrometers.
            radii (list[float]): Value controlling or representing radii."""
        n = len(coords)
        if n < 2 or area_um2 is None or area_um2 <= 0:
            return {f"{metric}_r{r}": np.nan for metric in self.metrics for r in radii}

        tree = KDTree(coords)
        features = {}
        for r in radii:
            pair_count = 0
            for i in range(n):
                idxs = tree.query_ball_point(coords[i], r)
                pair_count += len(idxs) - 1  # query includes the focal point
            K = (area_um2 / (n * (n - 1))) * pair_count
            if "K" in self.metrics:
                features[f"K_r{r}"] = float(K)
            if "L" in self.metrics:
                L = np.sqrt(K / np.pi) - r
                features[f"L_r{r}"] = float(L)
        return features

    # -- compute pcf (pair correlation function) -------------------------
    def _compute_pcf(
        self,
        coords: np.ndarray,
        area_um2: float,
        radii: list[float],
        bandwidth: float | None = None,
    ) -> dict[str, float]:
        """Estimate g(r) using kernel smoothing of interpoint distances.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array.
            area_um2 (float): Area measured in square micrometers.
            radii (list[float]): Value controlling or representing radii.
            bandwidth (float | None): Kernel bandwidth used by the spatial estimator."""
        n = len(coords)
        if n < 2 or area_um2 is None or area_um2 <= 0:
            return {f"pcf_r{r}": np.nan for r in radii}

        max_r = max(radii)
        # 收集所有成对距离（有序对，每个无序对出现两次）
        all_dists = []
        tree = KDTree(coords)
        for i in range(n):
            idxs = tree.query_ball_point(coords[i], max_r)  # 只取半径 max_r 内的邻居
            for j in idxs:
                if j == i:
                    continue
                # 跳过自身（实际上 query_ball_point 返回的索引不包含自身）
                d = np.linalg.norm(coords[i] - coords[j])
                all_dists.append(d)

        if len(all_dists) == 0:
            return {f"pcf_r{r}": np.nan for r in radii}

        # 核密度估计
        if bandwidth is None:
            bandwidth = max_r / 4.0 if max_r > 0 else 1.0
        kde = gaussian_kde(all_dists, bw_method=bandwidth)

        features = {}
        for r in radii:
            f_r = kde.evaluate([r])[0] if len(all_dists) > 0 else 0.0
            # pcf g(r) = A * f(r) / (2πr)
            if r > 0 and f_r > 0:
                g = (area_um2 * f_r) / (2 * np.pi * r * n * (n - 1))
            else:
                g = np.nan
            features[f"pcf_r{r}"] = float(g) if not np.isnan(g) else np.nan
        return features

    # -- compute variogram -----------------------------------------------
    def _compute_variogram(
        self,
        coords: np.ndarray,
        radii: list[float],
    ) -> dict[str, float]:
        """Compute semi-variogram: 0.5 * mean squared difference in coordinates.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array.
            radii (list[float]): Value controlling or representing radii."""
        n = len(coords)
        if n < 2:
            return {f"variogram_r{r}": np.nan for r in radii}

        # 设置容忍度
        if len(radii) == 0:
            return {}
        tol = min(radii) / 2.0
        features = {}
        tree = KDTree(coords)
        for r in radii:
            sum_sq_diff = 0.0
            count = 0
            # 对每个点，查询半径 r+tol 内的邻居
            for i in range(n):
                idxs = tree.query_ball_point(coords[i], r + tol)
                for j in idxs:
                    if j <= i:  # 只考虑 i<j 避免重复
                        continue
                    d = np.linalg.norm(coords[i] - coords[j])
                    if r - tol <= d <= r + tol:
                        sum_sq_diff += d**2
                        count += 1
            if count > 0:
                gamma = 0.5 * (sum_sq_diff / count)
            else:
                gamma = np.nan
            features[f"variogram_r{r}"] = float(gamma) if not np.isnan(gamma) else np.nan
        return features

    # -- extraction -------------------------------------------------------
    def extract_region(self, region: RegionData) -> dict[str, float]:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, float]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        raw_coords = region.coordinates[["x", "y"]].to_numpy(float)
        coords = region.coordinates_um[["x", "y"]].to_numpy(float)
        if len(coords) == 0:
            return self._empty_features()

        # Apply tissue mask if requested
        in_tissue = None
        if self.use_tissue_mask:
            in_tissue = region.polygon_contains(raw_coords, "tissue")
            if in_tissue is not None:
                coords = coords[in_tissue]
                if len(coords) == 0:
                    return self._empty_features()

        area = self._get_area_um2(region, coords)

        if self.by_type:
            col = self._col(region)
            if col is None:
                return self._empty_features()
            labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")
            if self.use_tissue_mask and in_tissue is not None:
                labels = labels.iloc[in_tissue].reset_index(drop=True)
            features = {}
            for ct in self.cell_types_:
                mask = labels == ct
                if not mask.any():
                    for r in self.radii:
                        for metric in self.metrics:
                            features[f"{ct}_{metric}_r{r}"] = np.nan
                    continue
                sub_coords = coords[mask]
                sub_feats = self._compute_all_metrics(sub_coords, area, self.radii)
                for key, val in sub_feats.items():
                    features[f"{ct}_{key}"] = val
            return features
        else:
            return self._compute_all_metrics(coords, area, self.radii)

    def _compute_all_metrics(
        self,
        coords: np.ndarray,
        area_um2: float,
        radii: list[float],
    ) -> dict[str, float]:
        """Compute all requested metrics and combine into one dict.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array.
            area_um2 (float): Area measured in square micrometers.
            radii (list[float]): Value controlling or representing radii."""
        features = {}
        if "K" in self.metrics or "L" in self.metrics:
            features.update(self._compute_ripley(coords, area_um2, radii))
        if "pcf" in self.metrics:
            features.update(self._compute_pcf(coords, area_um2, radii, self.pcf_bandwidth))
        if "variogram" in self.metrics:
            features.update(self._compute_variogram(coords, radii))
        return features

    def _empty_features(self) -> dict[str, float]:
        """Return NaN for all expected features."""
        features = {}
        for r in self.radii:
            for metric in self.metrics:
                features[f"{metric}_r{r}"] = np.nan
        return features
