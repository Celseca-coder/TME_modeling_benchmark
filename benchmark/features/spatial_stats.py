"""Provide spatial stats functionality."""

# """Spatial statistics features: Ripley's L-function and pair correlation function."""
# from __future__ import annotations

# import numpy as np
# from scipy.spatial import cKDTree
# from scipy.spatial import ConvexHull

# from benchmark.data.dataset import RegionData
# from .base import BaseFeatureExtractor


# def _estimate_area(xy: np.ndarray) -> float:
#     """Estimate region area using convex hull (fallback: bounding box)."""
#     if len(xy) < 4:
#         # Degenerate — use bounding box
#         ranges = xy.max(axis=0) - xy.min(axis=0)
#         return float(max(ranges[0] * ranges[1], 1.0))
#     try:
#         return float(ConvexHull(xy).volume)  # 'volume' is area in 2D
#     except Exception:
#         ranges = xy.max(axis=0) - xy.min(axis=0)
#         return float(max(ranges[0] * ranges[1], 1.0))


# def ripley_k(xy: np.ndarray, r_values: np.ndarray, area: float | None = None) -> np.ndarray:
#     """Estimate Ripley's K(r) for a 2D point pattern.

#     Uses the standard estimator without edge correction (appropriate when
#     the window area is much larger than the interaction range).

#     Returns K(r) values at each of *r_values*.
#     """
#     n = len(xy)
#     if n < 2:
#         return np.full(len(r_values), np.nan)

#     if area is None:
#         area = _estimate_area(xy)

#     tree = cKDTree(xy)
#     K = np.zeros(len(r_values))
#     for i, r in enumerate(r_values):
#         counts = tree.query_ball_point(xy, r=r, return_length=True)
#         # Exclude self-pairs
#         total = float(np.sum(counts)) - n
#         K[i] = (area / (n * (n - 1))) * total

#     return K


# def ripley_l(xy: np.ndarray, r_values: np.ndarray, area: float | None = None) -> np.ndarray:
#     """Besag's L-function: L(r) = sqrt(K(r) / pi) - r.

#     Positive values indicate clustering; negative values indicate inhibition.
#     """
#     K = ripley_k(xy, r_values, area)
#     with np.errstate(invalid="ignore"):
#         L = np.sqrt(np.maximum(K, 0) / np.pi) - r_values
#     return L


# def pair_correlation_function(
#     xy: np.ndarray,
#     r_bins: np.ndarray,
#     area: float | None = None,
# ) -> np.ndarray:
#     """Pair correlation function (PCF) g(r) estimated via kernel smoothing.

#     g(r) = K'(r) / (2 * pi * r), approximated by finite differences of K.

#     Returns g(r) at bin midpoints (length = len(r_bins) - 1).
#     """
#     n = len(xy)
#     if n < 2:
#         return np.full(len(r_bins) - 1, np.nan)

#     if area is None:
#         area = _estimate_area(xy)

#     r_vals = r_bins
#     K = ripley_k(xy, r_vals, area)
#     dK = np.diff(K)
#     dr = np.diff(r_vals)
#     r_mid = 0.5 * (r_vals[:-1] + r_vals[1:])

#     with np.errstate(divide="ignore", invalid="ignore"):
#         g = dK / (2 * np.pi * r_mid * dr)
#     return g


# class RipleyLFeatures(BaseFeatureExtractor):
#     """Ripley's L(r) - r at a set of radii, per cell type or all cells.

#     Parameters
#     ----------
#     r_values : list[float]
#         Radii (same units as coordinates) at which to evaluate L.
#     cell_types : list[str] | None
#         Compute L for each of these cell types separately.  None → all cells.
#     cell_type_col : str
#     sample_n : int | None
#         Sub-sample to at most *sample_n* cells per type for speed.
#     """

#     def __init__(
#         self,
#         r_values: list[float] | None = None,
#         cell_types: list[str] | None = None,
#         cell_type_col: str = "cell_type",
#         sample_n: int | None = 1000,
#     ) -> None:
#         self.r_values = np.asarray(r_values or [20, 50, 100, 200])
#         self.cell_types = cell_types
#         self.cell_type_col = cell_type_col
#         self.sample_n = sample_n
#         self._fitted_types: list[str] = []

#     def fit(self, regions: list[RegionData]) -> "RipleyLFeatures":
#         if self.cell_types is not None:
#             self._fitted_types = self.cell_types
#         else:
#             ct_set: set[str] = set()
#             for r in regions:
#                 col = self.cell_type_col
#                 if col in r.cell_types.columns:
#                     ct_set.update(r.cell_types[col].dropna().unique())
#             self._fitted_types = sorted(ct_set)
#         return self

#     def extract_region(self, region: RegionData) -> dict[str, float]:
#         col = self.cell_type_col
#         all_xy = region.coordinates[["x", "y"]].values
#         area = _estimate_area(all_xy)

#         feats: dict[str, float] = {}

#         # All-cells L-function
#         L_all = ripley_l(all_xy, self.r_values, area)
#         for r, val in zip(self.r_values, L_all):
#             feats[f"L_all_r{int(r)}"] = float(val)

#         # Per-type L-function
#         if col in region.cell_types.columns:
#             ct_series = region.cell_types[col]
#             coords = region.coordinates[["x", "y"]]
#             merged = coords.join(ct_series, how="inner")

#             for ct in self._fitted_types:
#                 ct_xy = merged.loc[merged[col] == ct, ["x", "y"]].values
#                 if len(ct_xy) < 4:
#                     for r in self.r_values:
#                         feats[f"L_{ct}_r{int(r)}"] = np.nan
#                     continue
#                 if self.sample_n and len(ct_xy) > self.sample_n:
#                     rng = np.random.default_rng(42)
#                     ct_xy = ct_xy[rng.choice(len(ct_xy), self.sample_n, replace=False)]
#                 L = ripley_l(ct_xy, self.r_values, area)
#                 for r, val in zip(self.r_values, L):
#                     feats[f"L_{ct}_r{int(r)}"] = float(val)

#         return feats


# class PCFFeatures(BaseFeatureExtractor):
#     """Pair correlation function g(r) at binned radii.

#     Parameters
#     ----------
#     r_bins : list[float]
#         Bin edges.  Features are computed at bin midpoints.
#     cell_types : list[str] | None
#         Compute PCF for each cell type separately.  None → all cells only.
#     cell_type_col : str
#     sample_n : int | None
#     """

#     def __init__(
#         self,
#         r_bins: list[float] | None = None,
#         cell_types: list[str] | None = None,
#         cell_type_col: str = "cell_type",
#         sample_n: int | None = 1000,
#     ) -> None:
#         self.r_bins = np.asarray(r_bins or [0, 20, 50, 100, 200])
#         self.cell_types = cell_types
#         self.cell_type_col = cell_type_col
#         self.sample_n = sample_n
#         self._fitted_types: list[str] = []
#         self._r_mid = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])

#     def fit(self, regions: list[RegionData]) -> "PCFFeatures":
#         if self.cell_types is not None:
#             self._fitted_types = self.cell_types
#         else:
#             ct_set: set[str] = set()
#             for r in regions:
#                 col = self.cell_type_col
#                 if col in r.cell_types.columns:
#                     ct_set.update(r.cell_types[col].dropna().unique())
#             self._fitted_types = sorted(ct_set)
#         return self

#     def extract_region(self, region: RegionData) -> dict[str, float]:
#         col = self.cell_type_col
#         all_xy = region.coordinates[["x", "y"]].values
#         area = _estimate_area(all_xy)

#         feats: dict[str, float] = {}

#         def _add(tag: str, xy: np.ndarray) -> None:
#             g = pair_correlation_function(xy, self.r_bins, area)
#             for r, val in zip(self._r_mid, g):
#                 feats[f"PCF_{tag}_r{int(r)}"] = float(val)

#         _add("all", all_xy)

#         if col in region.cell_types.columns:
#             ct_series = region.cell_types[col]
#             coords = region.coordinates[["x", "y"]]
#             merged = coords.join(ct_series, how="inner")

#             for ct in self._fitted_types:
#                 ct_xy = merged.loc[merged[col] == ct, ["x", "y"]].values
#                 if len(ct_xy) < 4:
#                     for r in self._r_mid:
#                         feats[f"PCF_{ct}_r{int(r)}"] = np.nan
#                     continue
#                 if self.sample_n and len(ct_xy) > self.sample_n:
#                     rng = np.random.default_rng(42)
#                     ct_xy = ct_xy[rng.choice(len(ct_xy), self.sample_n, replace=False)]
#                 _add(ct, ct_xy)

#         return feats
