"""Provide distance feats functionality."""

# """Distance-based features: min/mean spatial distance between cell-type pairs."""
# from __future__ import annotations

# from itertools import product

# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree

# from benchmark.data.dataset import RegionData
# from .base import BaseFeatureExtractor


# class CrossTypeDistanceFeatures(BaseFeatureExtractor):
#     """Min and mean nearest-neighbour distance between pairs of cell types.

#     For each ordered pair (A, B) of cell types:
#       - ``min_dist_A_to_B``  : min over all A-cells of the distance to the nearest B-cell
#       - ``mean_dist_A_to_B`` : mean over all A-cells of the distance to the nearest B-cell

#     Parameters
#     ----------
#     cell_type_pairs : list[tuple[str, str]] | None
#         Ordered (source, target) pairs.  If None, all ordered pairs of known
#         cell types are used (can be large).
#     cell_type_col : str
#         Column name for cell type labels.
#     max_dist : float
#         Cap distances at this value (µm) to handle isolated cells.  NaN if no
#         target cell is present.
#     k_neighbours : int
#         Number of nearest neighbours to query (default 1 → nearest neighbour).
#     sample_n : int | None
#         If set, sub-sample each source type to at most *sample_n* cells to
#         keep computation manageable for very large regions.
#     """

#     def __init__(
#         self,
#         cell_type_pairs: list[tuple[str, str]] | None = None,
#         cell_type_col: str = "cell_type",
#         max_dist: float = 500.0,
#         k_neighbours: int = 1,
#         sample_n: int | None = 500,
#     ) -> None:
#         self.cell_type_pairs = cell_type_pairs
#         self.cell_type_col = cell_type_col
#         self.max_dist = max_dist
#         self.k_neighbours = k_neighbours
#         self.sample_n = sample_n
#         self._cell_types: list[str] = []

#     def fit(self, regions: list[RegionData]) -> "CrossTypeDistanceFeatures":
#         ct_set: set[str] = set()
#         for r in regions:
#             col = self.cell_type_col
#             if col in r.cell_types.columns:
#                 ct_set.update(r.cell_types[col].dropna().unique())
#         self._cell_types = sorted(ct_set)
#         if self.cell_type_pairs is None:
#             self.cell_type_pairs = list(product(self._cell_types, self._cell_types))
#         return self

#     def extract_region(self, region: RegionData) -> dict[str, float]:
#         col = self.cell_type_col
#         if col not in region.cell_types.columns:
#             return {}

#         # Build lookup: cell_type → xy coordinates
#         ct_series = region.cell_types[col]
#         coords = region.coordinates[["x", "y"]]
#         merged = coords.join(ct_series, how="inner")

#         feats: dict[str, float] = {}
#         pairs = self.cell_type_pairs or list(product(self._cell_types, self._cell_types))

#         # Cache KDTree per target type
#         trees: dict[str, cKDTree | None] = {}

#         for src_type, tgt_type in pairs:
#             src_xy = merged.loc[merged[col] == src_type, ["x", "y"]].values
#             if len(src_xy) == 0:
#                 feats[f"min_dist_{src_type}_to_{tgt_type}"] = np.nan
#                 feats[f"mean_dist_{src_type}_to_{tgt_type}"] = np.nan
#                 continue

#             if tgt_type not in trees:
#                 tgt_xy = merged.loc[merged[col] == tgt_type, ["x", "y"]].values
#                 trees[tgt_type] = cKDTree(tgt_xy) if len(tgt_xy) > 0 else None

#             tree = trees[tgt_type]
#             if tree is None:
#                 feats[f"min_dist_{src_type}_to_{tgt_type}"] = np.nan
#                 feats[f"mean_dist_{src_type}_to_{tgt_type}"] = np.nan
#                 continue

#             if self.sample_n and len(src_xy) > self.sample_n:
#                 rng = np.random.default_rng(42)
#                 src_xy = src_xy[rng.choice(len(src_xy), self.sample_n, replace=False)]

#             dists, _ = tree.query(src_xy, k=self.k_neighbours, workers=1)
#             if self.k_neighbours > 1:
#                 dists = dists.min(axis=1)
#             dists = np.minimum(dists, self.max_dist)

#             feats[f"min_dist_{src_type}_to_{tgt_type}"] = float(dists.min())
#             feats[f"mean_dist_{src_type}_to_{tgt_type}"] = float(dists.mean())

#         return feats
