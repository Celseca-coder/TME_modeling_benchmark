# """Proximity features: cell-cell co-localisation within a spatial radius."""
# from __future__ import annotations

# from itertools import product

# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree

# from benchmark.data.dataset import RegionData
# from .base import BaseFeatureExtractor


# class ProximityFeatures(BaseFeatureExtractor):
#     """Count cell-type neighbours within *radius* µm, normalised by cell counts.

#     For each ordered pair (A, B):
#       ``prox_{A}_{B}_r{radius}`` = (# B-cells within radius of any A-cell) / n_cells_A

#     This is an asymmetric, unnormalised estimate of the conditional density
#     of B around A.  Divide by n_cells_B / area to get a formal cross-type
#     pair correlation estimate; for benchmarking the raw density is sufficient.

#     Parameters
#     ----------
#     radii : list[float]
#         One or more search radii (µm or pixels, matching coordinate units).
#     cell_type_pairs : list[tuple[str, str]] | None
#         If None, all ordered pairs of fitted cell types are used.
#     cell_type_col : str
#     min_cells : int
#         Minimum cells of a type in a region to compute the feature (else NaN).
#     """

#     def __init__(
#         self,
#         radii: list[float] | None = None,
#         cell_type_pairs: list[tuple[str, str]] | None = None,
#         cell_type_col: str = "cell_type",
#         min_cells: int = 3,
#     ) -> None:
#         self.radii = radii or [20.0, 50.0, 100.0]
#         self.cell_type_pairs = cell_type_pairs
#         self.cell_type_col = cell_type_col
#         self.min_cells = min_cells
#         self._cell_types: list[str] = []

#     def fit(self, regions: list[RegionData]) -> "ProximityFeatures":
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

#         ct_series = region.cell_types[col]
#         coords = region.coordinates[["x", "y"]]
#         merged = coords.join(ct_series, how="inner")

#         # Build per-type KDTrees over all cells (for target queries)
#         all_xy = merged[["x", "y"]].values
#         all_types = merged[col].values

#         type_xy: dict[str, np.ndarray] = {}
#         for ct in self._cell_types:
#             mask = all_types == ct
#             type_xy[ct] = all_xy[mask]

#         # Build target trees
#         trees: dict[str, cKDTree | None] = {
#             ct: (cKDTree(xy) if len(xy) > 0 else None)
#             for ct, xy in type_xy.items()
#         }

#         feats: dict[str, float] = {}
#         pairs = self.cell_type_pairs or list(product(self._cell_types, self._cell_types))

#         for radius in self.radii:
#             r_tag = f"r{int(radius)}"
#             for src_type, tgt_type in pairs:
#                 key = f"prox_{src_type}_{tgt_type}_{r_tag}"
#                 src_xy = type_xy.get(src_type, np.empty((0, 2)))
#                 if len(src_xy) < self.min_cells:
#                     feats[key] = np.nan
#                     continue

#                 tree = trees.get(tgt_type)
#                 if tree is None:
#                     feats[key] = 0.0
#                     continue

#                 # Count B-cells within radius for each A-cell
#                 counts = tree.query_ball_point(src_xy, r=radius, return_length=True)
#                 # Subtract 1 if src == tgt (exclude self)
#                 if src_type == tgt_type:
#                     counts = np.maximum(0, np.asarray(counts) - 1)
#                 feats[key] = float(np.mean(counts))

#         return feats
