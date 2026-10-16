"""Shared per-window instance features for naive-mean and attention MIL."""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData

from .basic_feats import CompositionFeaturizer, MeanExpressionFeaturizer
from .mixing import MixingFeaturizer
from .point_pattern import PointPatternFeaturizer

VALID_WINDOW_GROUPS = {
    "composition",
    "expression",
    "density",
    "entropy",
    "mixing",
    "celltype_density",
    "point_pattern",
}

# Window-scale analog of the global tabular panel.
# Global "density" is per-type tissue/tumor density; a 100 µm window has no
# tissue mask, so the family is total density + per-type counts / window area.
WINDOW_PANEL_FAMILIES = ("composition", "density", "mixing", "point_pattern")
PANEL_FAMILY_TO_GROUPS = {
    "composition": ("composition",),
    "density": ("density", "celltype_density"),
    "mixing": ("mixing",),
    "point_pattern": ("point_pattern",),
}
FAMILY_COLUMN_PREFIXES = {
    "composition": ("composition__",),
    "density": ("local_cell_density_per_mm2", "celltype_density__"),
    "mixing": ("mixing__",),
    "point_pattern": ("point_pattern__",),
}
WINDOW_PANEL_COMBOS = (
    ("composition",),
    ("density",),
    ("mixing",),
    ("point_pattern",),
    ("composition", "density"),
    ("composition", "mixing"),
    ("composition", "point_pattern"),
    ("density", "mixing"),
    ("density", "point_pattern"),
    ("mixing", "point_pattern"),
    ("composition", "density", "mixing", "point_pattern"),
)
WINDOW_POINT_PATTERN_RADII = (10, 20, 50)

MOTIF_MIL_GROUPS = ("composition", "mixing", "celltype_density")
MOTIF_MIL_COMBOS = (
    ("composition",),
    ("mixing",),
    ("celltype_density",),
    ("composition", "mixing"),
    ("composition", "celltype_density"),
    ("mixing", "celltype_density"),
)


def expand_panel_groups(families: tuple[str, ...]) -> tuple[str, ...]:
    """Map panel family names (density, point_pattern, …) to extractor groups."""
    out: list[str] = []
    for family in families:
        if family not in PANEL_FAMILY_TO_GROUPS:
            raise ValueError(f"Unknown panel family: {family}")
        out.extend(PANEL_FAMILY_TO_GROUPS[family])
    return tuple(dict.fromkeys(out))


def combo_name(families: tuple[str, ...]) -> str:
    return "+".join(families)


def columns_for_families(columns: list[str], families: tuple[str, ...]) -> list[str]:
    prefixes: list[str] = []
    for family in families:
        if family not in FAMILY_COLUMN_PREFIXES:
            raise ValueError(f"Unknown panel family: {family}")
        prefixes.extend(FAMILY_COLUMN_PREFIXES[family])
    keep = []
    for col in columns:
        if any(col == prefix or col.startswith(prefix) for prefix in prefixes):
            keep.append(col)
    return keep


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
        point_pattern_radii: tuple[float, ...] = WINDOW_POINT_PATTERN_RADII,
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
        self.point_pattern_radii = tuple(point_pattern_radii)
        self.cell_types_: list[str] = []
        self.markers_: list[str] = []
        self.composition_: CompositionFeaturizer | None = None
        self.expression_: MeanExpressionFeaturizer | None = None
        self.mixing_: MixingFeaturizer | None = None
        self.point_pattern_: PointPatternFeaturizer | None = None

    def fit(self, regions: list[RegionData]) -> "LocalWindowInstanceFeaturizer":
        if "composition" in self.feature_groups:
            self.composition_ = CompositionFeaturizer(cell_type_col=self.cell_type_col).fit(regions)
            self.cell_types_ = list(self.composition_.cell_types_)
        else:
            self.composition_ = None
            self.cell_types_ = _discover_cell_types(regions, self.cell_type_col)

        if "expression" in self.feature_groups:
            self.expression_ = MeanExpressionFeaturizer().fit(regions)
            self.markers_ = list(self.expression_.markers_)
        else:
            self.expression_ = None
            self.markers_ = []

        if self.max_markers is not None:
            self.markers_ = self.markers_[: self.max_markers]
            if self.expression_ is not None:
                self.expression_.markers_ = self.markers_

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

        if "point_pattern" in self.feature_groups:
            self.point_pattern_ = PointPatternFeaturizer(
                radii=list(self.point_pattern_radii),
                metrics=("K", "L"),
                by_type=True,
                cell_type_col=self.cell_type_col,
                use_tissue_mask=False,
            ).fit(regions)
            if not self.cell_types_:
                self.cell_types_ = list(self.point_pattern_.cell_types_)
            else:
                self.point_pattern_.cell_types_ = list(self.cell_types_)
        else:
            self.point_pattern_ = None

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
            elif group == "point_pattern":
                if self.point_pattern_ is None:
                    continue
                for cell_type in self.point_pattern_.cell_types_:
                    for radius in self.point_pattern_.radii:
                        for metric in self.point_pattern_.metrics:
                            names.append(f"point_pattern__{cell_type}_{metric}_r{radius}")
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

        subset = _subset_region(region, cell_ids)
        simple_values: dict[str, float] = {}
        if self.composition_ is not None:
            for name, value in self.composition_.extract_region(subset).items():
                simple_values[f"composition__{name}"] = value
        if self.expression_ is not None:
            for name, value in self.expression_.extract_region(subset).items():
                simple_values[f"expression__{name}"] = value

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
            mixed = self.mixing_.extract_region(subset)
            for name, value in mixed.items():
                values[f"mixing__{name}"] = value
        if "point_pattern" in self.feature_groups and self.point_pattern_ is not None:
            xy = subset.coordinates[["x", "y"]].to_numpy(float)
            pp = self.point_pattern_
            type_col = (
                self.cell_type_col if self.cell_type_col in subset.cell_types.columns
                else ("cell_type" if "cell_type" in subset.cell_types.columns else None)
            )
            labels_pp = (
                None if type_col is None
                else subset.cell_types[type_col].astype("object").to_numpy()
            )
            for cell_type in pp.cell_types_:
                if labels_pp is None:
                    sub_xy = np.empty((0, 2), dtype=float)
                else:
                    sub_xy = xy[labels_pp.astype(str) == str(cell_type)]
                feats = pp._compute_all_metrics(sub_xy, area_mm2, pp.radii)
                for key, value in feats.items():
                    values[f"point_pattern__{cell_type}_{key}"] = value

        return np.asarray([values.get(name, np.nan) for name in names], dtype=float)

    def extract_table(
        self,
        region: RegionData,
        windows,
        window_size_um: float,
    ) -> pd.DataFrame:
        """One row per window; index is window_id."""
        names = self.feature_names()
        rows = []
        for win in windows:
            vec = self.extract_vector(region, win.inside, window_size_um)
            rec = {"window_id": int(win.window_id),
                "cx_um": float(win.center_um[0]),
                "cy_um": float(win.center_um[1])}
            rec.update(zip(names, vec.tolist()))
            rows.append(rec)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.set_index("window_id")
