"""Define the common interface for region-level feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from benchmark.data.dataset import RegionData


class BaseFeatureExtractor(ABC):
    """Convert spatial regions into flat, region-level feature vectors."""

    def fit(self, regions: list[RegionData]) -> "BaseFeatureExtractor":
        """Learn any feature schema required by an extractor.

        Args:
            regions: Training regions used to learn vocabularies or parameters.

        Returns:
            This fitted extractor.
        """
        return self

    @abstractmethod
    def extract_region(self, region: RegionData) -> dict[str, float]:
        """Extract one flat feature vector.

        Args:
            region: Region to summarize.

        Returns:
            Mapping from feature name to numeric value.
        """

    def transform(self, regions: list[RegionData]) -> pd.DataFrame:
        """Extract a feature table for multiple regions.

        Args:
            regions: Regions to summarize independently.

        Returns:
            Region-by-feature table indexed by region identifier.
        """
        rows = []
        ids = []
        for region in regions:
            rows.append(self.extract_region(region))
            ids.append(region.region_id)
        table = pd.DataFrame(rows, index=ids)
        table.index.name = "region_id"
        return table

    def fit_transform(self, regions: list[RegionData]) -> pd.DataFrame:
        """Fit the extractor and transform the same regions.

        Args:
            regions: Regions used to learn the schema and produce output rows.

        Returns:
            Region-by-feature table indexed by region identifier.
        """
        return self.fit(regions).transform(regions)
