"""Shared per-window instance features for naive-mean and attention MIL."""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData

from .combined import CombinedCompositionExpressionFeaturizer
from .mixing import MixingFeaturizer

VALID_WINDOW_GROUPS = {
    "composition",
    "expression",
    "density",
    "entropy",
    "mixing",
    "celltype_density",
}

MOTIF_MIL_GROUPS = ("composition", "mixing", "celltype_density")
MOTIF_MIL_COMBOS = (
    ("composition",),
    ("mixing",),
    ("celltype_density",),
    ("composition", "mixing"),
    ("composition", "celltype_density"),
    ("mixing", "celltype_density"),
)


def _subset_region(region: RegionData, cell_ids) -> RegionData:
    cell_ids = region.coordinates.index.intersection(cell_ids, sort=False)
    return RegionData(
        region_id=f"{region.region_id}___local",
        coordinates=region.coordinates.loc[cell_ids].copy(),
        expression=region.expression.reindex(cell_ids).copy(),
        cell_types=region.cell_types.reindex(cell_ids).copy(),
        microns_per_pixel=region.microns_per_pixel,
        polygons=None,
    )


def _discover_cell_types(regions: list[RegionData], cell_type_col: str) -> list[str]:
    types: set[str] = set()
    for region in regions:
        col = cell_type_col if cell_type_col in region.cell_types.columns else (
            "cell_type" if "cell_type" in region.cell_types.columns else None
        )
        if col is None:
            continue
        types.update(region.cell_types[col].dropna().astype(str).unique())
    return sorted(types)


class LocalWindowInstanceFeaturizer:
    """One feature vector per spatial window; vocabulary is fold-local."""

    VALID_GROUPS = VALID_WINDOW_GROUPS

    def __init__(
        self,
        feature_groups: tuple[str, ...] = MOTIF_MIL_GROUPS,
        cell_type_col: str = "cell_type",
        max_markers: int | None = None,
        mixing_k: int = 10,
    ) -> None:
        unknown = set(feature_groups) - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
        if not feature_groups:
            raise ValueError("At least one feature group is required")
        self.feature_groups = tuple(feature_groups)
        self.cell_type_col = cell_type_col
        self.max_markers = max_markers
        self.mixing_k = mixing_k
        self.cell_types_: list[str] = []
        self.markers_: list[str] = []
        self.local_features_: CombinedCompositionExpressionFeaturizer | None = None
        self.mixing_: MixingFeaturizer | None = None

    def fit(self, regions: list[RegionData]) -> "LocalWindowInstanceFeaturizer":
        simple = tuple(
            group for group in ("composition", "expression")
            if group in self.feature_groups
        )
        if simple:
            self.local_features_ = CombinedCompositionExpressionFeaturizer(
                cell_type_col=self.cell_type_col,
                feature_groups=simple,
            ).fit(regions)
            self.cell_types_ = list(self.local_features_.cell_types_)
            self.markers_ = list(self.local_features_.markers_)
        else:
            self.local_features_ = None
            self.markers_ = []
            self.cell_types_ = _discover_cell_types(regions, self.cell_type_col)

        if self.max_markers is not None:
            self.markers_ = self.markers_[: self.max_markers]
            if self.local_features_ is not None:
                self.local_features_.expression.markers_ = self.markers_

        if "mixing" in self.feature_groups:
            self.mixing_ = MixingFeaturizer(
                cell_type_col=self.cell_type_col,
                k_neighbors=self.mixing_k,
                use_tissue_mask=False,
            ).fit(regions)
            if not self.cell_types_:
                self.cell_types_ = list(self.mixing_.cell_types_)
            else:
                self.mixing_.cell_types_ = list(self.cell_types_)
        else:
            self.mixing_ = None

        if not self.feature_names():
            raise ValueError("No local window features are available")
        return self

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group in self.feature_groups:
            if group == "composition":
                names.extend(f"composition__{x}" for x in self.cell_types_)
            elif group == "density":
                names.append("local_cell_density_per_mm2")
            elif group == "celltype_density":
                names.extend(f"celltype_density__{x}" for x in self.cell_types_)
            elif group == "expression":
                names.extend(f"expression__{x}" for x in self.markers_)
            elif group == "entropy":
                names.append("local_celltype_entropy")
            elif group == "mixing":
                if self.mixing_ is None:
                    continue
                names.extend(f"mixing__{name}" for name in self.mixing_.feature_names())
        return names

    def extract_vector(
        self,
        region: RegionData,
        inside: np.ndarray,
        window_size_um: float,
    ) -> np.ndarray:
        names = self.feature_names()
        values: dict[str, float] = {}
        cell_ids = region.coordinates.index[inside]
        n = int(inside.sum())
        area_mm2 = (float(window_size_um) * float(window_size_um)) / 1_000_000.0

        simple_values: dict[str, float] = {}
        if self.local_features_ is not None:
            simple_values = self.local_features_.extract_cells(region, cell_ids)

        col = (
            self.cell_type_col if self.cell_type_col in region.cell_types.columns
            else ("cell_type" if "cell_type" in region.cell_types.columns else None)
        )
        labels = None if col is None else (
            region.cell_types[col].reindex(region.coordinates.index).astype("object")
        )
        counts = pd.Series(dtype=float)
        if labels is not None:
            counts = labels.iloc[inside].dropna().astype(str).value_counts()

        if "composition" in self.feature_groups:
            for cell_type in self.cell_types_:
                values[f"composition__{cell_type}"] = simple_values[f"composition__{cell_type}"]
        if "density" in self.feature_groups:
            values["local_cell_density_per_mm2"] = float(n / area_mm2) if area_mm2 else np.nan
        if "celltype_density" in self.feature_groups:
            for cell_type in self.cell_types_:
                values[f"celltype_density__{cell_type}"] = (
                    float(counts.get(cell_type, 0.0) / area_mm2) if area_mm2 else np.nan
                )
        if "expression" in self.feature_groups:
            for marker in self.markers_:
                values[f"expression__{marker}"] = simple_values[f"expression__{marker}"]
        if "entropy" in self.feature_groups:
            p = counts.to_numpy(float)
            p = p / p.sum() if p.sum() else p
            values["local_celltype_entropy"] = float(-(p * np.log(p + 1e-12)).sum()) if len(p) else np.nan
        if "mixing" in self.feature_groups and self.mixing_ is not None:
            mixed = self.mixing_.extract_region(_subset_region(region, cell_ids))
            for name, value in mixed.items():
                values[f"mixing__{name}"] = value

        return np.asarray([values[name] for name in names], dtype=float)
