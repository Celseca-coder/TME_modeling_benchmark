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
    """Represent cellular neighborhood featurizer."""
    def __init__(
        self,
        cell_type_col: str = "cell_type",
        coord_cols=("x", "y"),
        n_neighborhoods: int = 10,
        k_neighbors: int = 10,
        radius_um: float | None = None,
        max_cells_per_fit_region: int = 2000,
        random_state: int = 0,
        include_cn_celltype_content: bool = True,
    ) -> None:
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
                    coord_cols (Any): Names of columns containing coord.
                    n_neighborhoods (int): Number of neighborhoods.
                    k_neighbors (int): Value controlling or representing k neighbors.
                    radius_um (float | None): Radius measured in micrometers.
                    max_cells_per_fit_region (int): Maximum allowed cells per fit region.
                    random_state (int): Random state used for reproducibility.
                    include_cn_celltype_content (bool): Whether to include the
                        within-CN cell-type composition features. When False,
                        only CN abundance features are returned.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.coord_cols = list(coord_cols)
        self.n_neighborhoods = n_neighborhoods
        self.k_neighbors = k_neighbors
        self.radius_um = radius_um
        self.max_cells_per_fit_region = max_cells_per_fit_region
        self.random_state = random_state
        self.include_cn_celltype_content = include_cn_celltype_content
        self.cell_types_: list[str] = []
        self.kmeans_: KMeans | None = None

    def _col(self, region: RegionData) -> str:
        """Execute the col operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    str: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        raise ValueError(f"Region {region.region_id} has no cell-type column")

    def _coords_um(self, region: RegionData) -> np.ndarray:
        """Execute the coords(use um as unit) operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        coords_um = getattr(region, "coordinates_um", None)
        if coords_um is not None:
            return coords_um[self.coord_cols].to_numpy(float)
        coords = region.coordinates[self.coord_cols].to_numpy(float)
        return coords * float(getattr(region, "microns_per_pixel", 1.0))

    def _labels(self, region: RegionData) -> pd.Series:
        """Read the labels of each cell.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    pd.Series: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        return region.cell_types[self._col(region)].reindex(region.coordinates.index).astype("object")

    def _type_ids(self, labels: pd.Series) -> np.ndarray:
        """Turn cell type in string format into integer IDs. 
        
                Args:
                    labels (pd.Series): Cell-type or class label assigned to each observation.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            labels (pd.Series): Cell-type or class label assigned to each observation."""
        type_map = {ct: i for i, ct in enumerate(self.cell_types_)}
        return np.array([type_map.get(str(x), -1) for x in labels], dtype=int)

    def _neighbour_indices(self, coords: np.ndarray) -> list[np.ndarray]:
        """Find the neighbor of each cell
        
                Args:
                    coords (np.ndarray): Two-dimensional cell-coordinate array.
        
                Returns:
                    list[np.ndarray]: The operation result.
        
        Args:
            coords (np.ndarray): Two-dimensional cell-coordinate array."""
        n = len(coords)
        # 取得当前区域的细胞数量
        if n == 0:
            return []
        tree = cKDTree(coords)
        #cKDTree:一种根据二维坐标建立空间索引的东西
        # 建立索引后，可以快速回答两类问题：1. 距离每个细胞最近的 K 个细胞是谁？2. 每个细胞指定半径范围内有哪些细胞？
        #如果有半径，就使用query_ball_point方法来获取每个细胞在指定半径内的邻居索引
        if self.radius_um is not None:
            return [np.asarray(ix, dtype=int) for ix in tree.query_ball_point(coords, self.radius_um)]
        # 如果没有指定半径，则使用 k_neighbors 参数来确定邻居数量
        k = min(self.k_neighbors + 1, n)
        neighbours = tree.query(coords, k=k)[1]
        if k == 1:
            neighbours = neighbours[:, None]
        return [np.asarray(row, dtype=int) for row in neighbours]

    def _profiles(self, region: RegionData) -> np.ndarray:
        """
        Put out the neighborhood matrix
        Eg:
        profiles = np.array([
            [0.333, 0.000, 0.667],  # 细胞 0 的邻域
            [0.333, 0.000, 0.667],  # 细胞 1 的邻域
            [0.333, 0.000, 0.667],  # 细胞 2 的邻域
            [0.500, 0.500, 0.000],  # 细胞 3 的邻域
            [0.500, 0.500, 0.000],  # 细胞 4 的邻域
        ])
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed.
        """
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
        """Fit.
                Collect all the cell types appeared in the training regions, combine all the neighborhood vector,then fit KMeans to the neighborhood profiles.
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'CellularNeighborhoodFeaturizer': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
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
            # 如果区域内细胞数目过多，就随机保留
            if len(profiles) > self.max_cells_per_fit_region:
                stable = sum(ord(ch) for ch in str(region.region_id))
                rng = np.random.default_rng(self.random_state + stable)
                keep = rng.choice(len(profiles), self.max_cells_per_fit_region, replace=False)
                profiles = profiles[np.sort(keep)]
            sampled_profiles.append(profiles)

        if not sampled_profiles:
            raise ValueError("No cells available to fit Cellular Neighborhoods")
        '''
        数据堆叠,eg:
        profiles_A = np.array([
            [0.2, 0.1, 0.7],
            [0.3, 0.0, 0.7],
            [0.1, 0.2, 0.7],
        ])

        profiles_B = np.array([
            [0.6, 0.4, 0.0],
            [0.5, 0.5, 0.0],
        ])
        结果是:
        X = np.array([
            [0.2, 0.1, 0.7],  # Region A，细胞 0
            [0.3, 0.0, 0.7],  # Region A，细胞 1
            [0.1, 0.2, 0.7],  # Region A，细胞 2
            [0.6, 0.4, 0.0],  # Region B，细胞 0
            [0.5, 0.5, 0.0],  # Region B，细胞 1
        ])
        '''
        X = np.vstack(sampled_profiles)
        n_clusters = min(self.n_neighborhoods, max(1, len(X)))
        self.kmeans_ = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.kmeans_.fit(X)
        return self

    def feature_names(self) -> list[str]:
        """
        Execute the feature names operation.
        生成最终输出特征的名称。
        输出包含两类特征。
        第一类是每种 CN 在整个区域中的占比：
        cn_fraction::0
        cn_fraction::1
        第二类是某种 CN 内部的平均细胞类型组成：
        cn_celltype_fraction::0::T_cell
        cn_celltype_fraction::0::B_cell
        调用这个方法前必须先调用 fit()，因为特征数量取决于实际拟合出的聚类数量。
        Returns:
            list[str]: The operation result."""
        assert self.kmeans_ is not None, "call fit() before feature_names()"
        names = [f"cn_fraction::{i}" for i in range(self.kmeans_.n_clusters)]
        if self.include_cn_celltype_content:
            for i in range(self.kmeans_.n_clusters):
                for ct in self.cell_types_:
                    names.append(f"cn_celltype_fraction::{i}::{ct}")
        return names

    def extract_region(self, region: RegionData) -> dict[str, float]:
        """
        Extract region.
        计算区域中每个细胞的邻域组成向量。
        使用训练好的 KMeans 判断每个细胞属于哪种 CN。
        计算每种 CN 在该区域中的细胞占比。
        计算属于每种 CN 的细胞，其邻域组成向量的平均值。
        返回 {特征名: 数值} 字典。
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, float]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
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
            # 后面可能导致错误
            if self.include_cn_celltype_content and mask.any():
                mean_profile = profiles[mask].mean(axis=0)
                for j, ct in enumerate(self.cell_types_):
                    out[f"cn_celltype_fraction::{i}::{ct}"] = float(mean_profile[j])
        return out
