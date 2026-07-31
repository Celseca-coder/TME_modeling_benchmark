"""Abstract base for all feature extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from benchmark.data.dataset import RegionData


class BaseFeatureExtractor(ABC):
    """Extract a flat feature vector from a RegionData object,
    with optional specification of cell IDs.

    Subclasses implement :meth:`extract_region`.  The ``fit`` / ``transform``
    API mirrors sklearn's pattern so extractors can be used inside pipelines.
    """

    def fit(self, regions: list[RegionData]) -> "BaseFeatureExtractor":
        """Discover domain from data (e.g. all cell-type names). Override if needed.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        return self

    @abstractmethod
    def extract_region(self, region: RegionData) -> dict[str, float]:
        """Return a flat dict mapping feature name → value for one region.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""

    def transform(self, regions: list[RegionData]) -> pd.DataFrame:
        """Extract features for all regions; returns DataFrame indexed by region_id.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        rows = []
        ids = []
        for r in regions:
            rows.append(self.extract_region(r))
            ids.append(r.region_id)
        df = pd.DataFrame(rows, index=ids)
        df.index.name = "region_id"
        return df

    def fit_transform(self, regions: list[RegionData]) -> pd.DataFrame:
        """Fit transform.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    pd.DataFrame: The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        return self.fit(regions).transform(regions)
