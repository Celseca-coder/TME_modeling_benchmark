"""Global (region-level) features: cell-type composition and mean expression."""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor


class CompositionFeaturizer(BaseFeatureExtractor):
    """Cell-type composition: the fraction of each cell type across the region.

    One column per cell type. The vocabulary is learned from the regions passed to
    :meth:`fit` (the training split), so cell types absent there are ignored and
    types absent from a given region are 0. Pass ``cell_type_col`` to use a
    cross-cohort-harmonised column (e.g. ``cell_type_uniform``).
    """

    def __init__(self, cell_type_col: str = "cell_type") -> None:
        self.cell_type_col = cell_type_col
        self.cell_types_: list[str] = []

    def _col(self, region: RegionData) -> str | None:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "CompositionFeaturizer":
        types: set[str] = set()
        for r in regions:
            col = self._col(r)
            if col is not None:
                types.update(r.cell_types[col].dropna().unique())
            else:
                raise ValueError(f"Region {r.region_id} has no cell-type column")
        self.cell_types_ = sorted(types)
        return self

    def extract_region(self, region: RegionData) -> dict[str, float]:
        col = self._col(region)
        counts = region.cell_types[col].dropna().value_counts() if col else pd.Series(dtype=float)
        n = float(counts.sum())
        return {ct: (float(counts.get(ct, 0)) / n if n else 0.0) for ct in self.cell_types_}


class MeanExpressionFeaturizer(BaseFeatureExtractor):
    """Mean expression of each marker, averaged over all cells in the region.

    One column per marker. The marker vocabulary is learned from the regions passed
    to :meth:`fit`; markers absent from a region are NaN (imputed model-side).
    """

    def __init__(self) -> None:
        self.markers_: list[str] = []

    def fit(self, regions: list[RegionData]) -> "MeanExpressionFeaturizer":
        # expression is indexed by cell_id, so its columns are exactly the markers
        marker_sets = [set(r.expression.columns) for r in regions]
        shared_markers = set.intersection(*marker_sets) if marker_sets else set()
        self.markers_ = sorted(shared_markers)
        return self

    def extract_region(self, region: RegionData) -> dict[str, float]:
        mu = region.expression.mean()
        return {m: float(mu.get(m, np.nan)) for m in self.markers_}

