"""Composition of multiple region-level feature extractors."""
from __future__ import annotations

from collections.abc import Mapping

from benchmark.data.dataset import RegionData

from .base import BaseFeatureExtractor


class CombinedFeaturizer(BaseFeatureExtractor):
    """Fit several extractors on the same split and concatenate their outputs.

    Feature names are prefixed with the group name, which prevents collisions and
    makes fitted model inputs auditable. ``fixed_markers`` is used only for the
    expression extractor in cross-cohort tests, where the marker vocabulary must
    be the intersection of the train and test panels.
    """

    def __init__(
        self,
        extractors: Mapping[str, BaseFeatureExtractor],
        fixed_markers: list[str] | None = None,
    ) -> None:
        self.extractors = dict(extractors)
        self.fixed_markers = fixed_markers

    def fit(self, regions: list[RegionData]) -> "CombinedFeaturizer":
        for name, extractor in self.extractors.items():
            if name == "expression" and self.fixed_markers is not None:
                extractor.markers_ = list(self.fixed_markers)
            else:
                extractor.fit(regions)
        return self

    def extract_region(self, region: RegionData) -> dict[str, float]:
        features: dict[str, float] = {}
        for name, extractor in self.extractors.items():
            features.update(
                {f"{name}::{key}": value for key, value in extractor.extract_region(region).items()}
            )
        return features
