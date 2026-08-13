"""Simple column-wise composition and expression feature concatenation."""
from __future__ import annotations

from benchmark.data.dataset import RegionData

from .base import BaseFeatureExtractor
from .basic_feats import CompositionFeaturizer, MeanExpressionFeaturizer


class CombinedCompositionExpressionFeaturizer(BaseFeatureExtractor):
    """Concatenate global-style composition and mean-expression features.

    The same fitted extractor can be applied to a complete region or to a local
    ``RegionData`` subset.  This keeps global, naive-MIL, and attention-MIL
    definitions identical; only the spatial unit supplied to ``extract_region``
    differs.
    """

    VALID_GROUPS = {"composition", "expression"}

    def __init__(
        self,
        cell_type_col: str = "cell_type",
        feature_groups: tuple[str, ...] = ("composition", "expression"),
    ) -> None:
        unknown = set(feature_groups) - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
        if not feature_groups:
            raise ValueError("At least one feature group is required")
        self.feature_groups = tuple(feature_groups)
        self.composition = CompositionFeaturizer(cell_type_col=cell_type_col)
        self.expression = MeanExpressionFeaturizer()

    def fit(
        self, regions: list[RegionData]
    ) -> "CombinedCompositionExpressionFeaturizer":
        if "composition" in self.feature_groups:
            self.composition.fit(regions)
        if "expression" in self.feature_groups:
            self.expression.fit(regions)
        if not self.feature_names():
            raise ValueError("No composition or expression features are available")
        return self

    @property
    def cell_types_(self) -> list[str]:
        return self.composition.cell_types_

    @property
    def markers_(self) -> list[str]:
        return self.expression.markers_

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if "composition" in self.feature_groups:
            names.extend(f"composition__{x}" for x in self.cell_types_)
        if "expression" in self.feature_groups:
            names.extend(f"expression__{x}" for x in self.markers_)
        return names

    def extract_region(self, region: RegionData) -> dict[str, float]:
        values: dict[str, float] = {}
        if "composition" in self.feature_groups:
            values.update(
                {
                    f"composition__{name}": value
                    for name, value in self.composition.extract_region(region).items()
                }
            )
        if "expression" in self.feature_groups:
            values.update(
                {
                    f"expression__{name}": value
                    for name, value in self.expression.extract_region(region).items()
                }
            )
        return values

    def extract_cells(
        self, region: RegionData, cell_ids
    ) -> dict[str, float]:
        """Extract the same features from a selected set of region cells.

        ``reindex`` is intentional: some external-cohort regions have coordinates
        and expression but no cell-type rows.  Global composition treats such a
        region as zero composition, so local MIL windows must do the same instead
        of failing strict ``.loc`` selection.
        """
        cell_ids = region.coordinates.index.intersection(cell_ids, sort=False)
        local_region = RegionData(
            region_id=f"{region.region_id}___local",
            coordinates=region.coordinates.loc[cell_ids].copy(),
            expression=region.expression.reindex(cell_ids).copy(),
            cell_types=region.cell_types.reindex(cell_ids).copy(),
            microns_per_pixel=region.microns_per_pixel,
            polygons=None,
        )
        return self.extract_region(local_region)
