"""Cellular Neighborhood (CN) unsupervised features.

This implements the common CN workflow used in spatial single-cell analyses:
for every cell, summarize the cell-type composition of its local neighbourhood,
cluster those neighbourhood profiles with KMeans on the training fold, then
represent each tissue region by the abundance and cell-type content of the
learned neighbourhood states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class CellularNeighborhoodFeaturizer(BaseFeatureExtractor):
    def __init__(
        self,
        cell_type_col: str = "cell_type",
        coord_cols=("x", "y"),
        n_neighborhoods: int = 10,
        k_neighbors: int = 10,
        radius_um: float | None = None,
        max_cells_per_fit_region: int = 2000,
        random_state: int = 0,
    ) -> None:
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.n_neighborhoods = n_neighborhoods
        self.k_neighbors = k_neighbors
        self.radius_um = radius_um
        self.max_cells_per_fit_region = max_cells_per_fit_region
        self.random_state = random_state
        self.cell_types_: list[str] = []
        self.kmeans_: KMeans | None = None

    def _col(self, region: RegionData) -> str:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        raise ValueError(f"Region {region.region_id} has no cell-type column")

    def _coords_um(self, region: RegionData) -> np.ndarray:
        coords_um = getattr(region, "coordinates_um", None)
        if coords_um is not None:
            return coords_um[self.coord_cols].to_numpy(float)
        coords = region.coordinates[self.coord_cols].to_numpy(float)
        return coords * float(getattr(region, "microns_per_pixel", 1.0))

    def _labels(self, region: RegionData) -> pd.Series:
        return region.cell_types[self._col(region)].reindex(region.coordinates.index).astype("object")

    def _type_ids(self, labels: pd.Series) -> np.ndarray:
        type_map = {ct: i for i, ct in enumerate(self.cell_types_)}
        return np.array([type_map.get(str(x), -1) for x in labels], dtype=int)

    def _neighbour_indices(self, coords: np.ndarray) -> list[np.ndarray]:
        n = len(coords)
        if n == 0:
            return []
        tree = cKDTree(coords)
        if self.radius_um is not None:
            return [np.asarray(ix, dtype=int) for ix in tree.query_ball_point(coords, self.radius_um)]
        k = min(self.k_neighbors + 1, n)
        neighbours = tree.query(coords, k=k)[1]
        if k == 1:
            neighbours = neighbours[:, None]
        return [np.asarray(row, dtype=int) for row in neighbours]

    def _profiles(self, region: RegionData) -> np.ndarray:
        coords = self._coords_um(region)
        type_ids = self._type_ids(self._labels(region))
        neighbours = self._neighbour_indices(coords)
        profiles = np.zeros((len(neighbours), len(self.cell_types_)), dtype=float)
        for i, ix in enumerate(neighbours):
            valid = type_ids[ix]
            valid = valid[valid >= 0]
            if len(valid) == 0:
                continue
            counts = np.bincount(valid, minlength=len(self.cell_types_))
            profiles[i] = counts / counts.sum()
        return profiles

    def fit(self, regions: list[RegionData]) -> "CellularNeighborhoodFeaturizer":
        types: set[str] = set()
        for region in regions:
            types.update(self._labels(region).dropna().astype(str).unique())
        self.cell_types_ = sorted(types)
        if not self.cell_types_:
            raise ValueError("No cell-type labels available to fit Cellular Neighborhoods")

        sampled_profiles = []
        for region in regions:
            profiles = self._profiles(region)
            if len(profiles) == 0:
                continue
            if len(profiles) > self.max_cells_per_fit_region:
                stable = sum(ord(ch) for ch in str(region.region_id))
                rng = np.random.default_rng(self.random_state + stable)
                keep = rng.choice(len(profiles), self.max_cells_per_fit_region, replace=False)
                profiles = profiles[np.sort(keep)]
            sampled_profiles.append(profiles)

        if not sampled_profiles:
            raise ValueError("No cells available to fit Cellular Neighborhoods")
        X = np.vstack(sampled_profiles)
        n_clusters = min(self.n_neighborhoods, max(1, len(X)))
        self.kmeans_ = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.kmeans_.fit(X)
        return self

    def feature_names(self) -> list[str]:
        assert self.kmeans_ is not None, "call fit() before feature_names()"
        names = [f"cn_fraction::{i}" for i in range(self.kmeans_.n_clusters)]
        for i in range(self.kmeans_.n_clusters):
            for ct in self.cell_types_:
                names.append(f"cn_celltype_fraction::{i}::{ct}")
        return names

    def extract_region(self, region: RegionData) -> dict[str, float]:
        assert self.kmeans_ is not None, "call fit() before extract_region()"
        profiles = self._profiles(region)
        out = {name: 0.0 for name in self.feature_names()}
        if len(profiles) == 0:
            return out

        cn = self.kmeans_.predict(profiles)
        n_cells = float(len(cn))
        for i in range(self.kmeans_.n_clusters):
            mask = cn == i
            out[f"cn_fraction::{i}"] = float(mask.sum() / n_cells)
            if mask.any():
                mean_profile = profiles[mask].mean(axis=0)
                for j, ct in enumerate(self.cell_types_):
                    out[f"cn_celltype_fraction::{i}::{ct}"] = float(mean_profile[j])
        return out
