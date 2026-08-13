"""Interpretable local-window bags for attention multiple-instance learning.

Each region is represented by a variable-length bag.  An instance is a square
spatial window and its features are deliberately simple: cell-type composition,
cell density, mean marker expression, and cell-type entropy.  The returned
``DataFrame`` contains object-valued bags so it can pass through the unchanged
benchmark cross-validation code to ``AttentionMILModel``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor
from .combined import CombinedCompositionExpressionFeaturizer


class HandcraftedAttentionMILFeaturizer(BaseFeatureExtractor):
    """Build one variable-length local-feature bag per region.

    Vocabulary discovery happens only in ``fit`` (inside each training fold), so
    validation/test regions cannot leak cell types or markers into the model.
    """

    VALID_GROUPS = {"composition", "density", "expression", "entropy"}

    def __init__(
        self,
        window_size_um: float = 100.0,
        step_um: float | None = 50.0,
        feature_groups: tuple[str, ...] = (
            "composition", "density", "expression", "entropy"
        ),
        cell_type_col: str = "cell_type",
        min_cells_per_window: int = 10,
        use_tissue_mask: bool = True,
        max_markers: int | None = None,
    ) -> None:
        unknown = set(feature_groups) - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
        self.window_size = float(window_size_um)
        self.step = float(step_um if step_um is not None else window_size_um)
        self.feature_groups = tuple(feature_groups)
        self.cell_type_col = cell_type_col
        self.min_cells = int(min_cells_per_window)
        self.use_tissue_mask = use_tissue_mask
        self.max_markers = max_markers
        self.cell_types_: list[str] = []
        self.markers_: list[str] = []
        self.local_features_: CombinedCompositionExpressionFeaturizer | None = None

    def _col(self, region: RegionData) -> str | None:
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "HandcraftedAttentionMILFeaturizer":
        simple_groups = tuple(
            group
            for group in ("composition", "expression")
            if group in self.feature_groups
        )
        if simple_groups:
            self.local_features_ = CombinedCompositionExpressionFeaturizer(
                cell_type_col=self.cell_type_col,
                feature_groups=simple_groups,
            ).fit(regions)
            self.cell_types_ = self.local_features_.cell_types_
            self.markers_ = self.local_features_.markers_
        else:
            self.local_features_ = None
            self.cell_types_ = []
            self.markers_ = []
        if self.max_markers is not None:
            self.markers_ = self.markers_[: self.max_markers]
            if self.local_features_ is not None:
                self.local_features_.expression.markers_ = self.markers_
        if not self.feature_names():
            raise ValueError("No local handcrafted features are available")
        return self

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if "composition" in self.feature_groups:
            names.extend(f"composition__{x}" for x in self.cell_types_)
        if "density" in self.feature_groups:
            names.append("local_cell_density_per_mm2")
        if "expression" in self.feature_groups:
            names.extend(f"expression__{x}" for x in self.markers_)
        if "entropy" in self.feature_groups:
            names.append("local_celltype_entropy")
        return names

    def _windows(self, region: RegionData):
        xy = (region.coordinates[["x", "y"]].to_numpy(float)
              * float(region.microns_per_pixel))
        if len(xy) == 0:
            return []
        margin = self.window_size * 0.1
        xmin, ymin = xy.min(axis=0) - margin
        xmax, ymax = xy.max(axis=0) + margin
        out = []
        x = xmin
        while x < xmax:
            y = ymin
            while y < ymax:
                out.append((x, y, x + self.window_size, y + self.window_size))
                y += self.step
            x += self.step
        return out

    def extract_region(self, region: RegionData) -> dict:
        raw_xy = region.coordinates[["x", "y"]].to_numpy(float)
        xy = raw_xy * float(region.microns_per_pixel)
        # Exported polygons live in raw coordinate/pixel space.
        tissue = region.polygon_contains(raw_xy, "tissue") if self.use_tissue_mask else None
        col = self._col(region)
        labels = None if col is None else (
            region.cell_types[col].reindex(region.coordinates.index).astype("object")
        )
        rows, centers = [], []
        for xmin, ymin, xmax, ymax in self._windows(region):
            inside = ((xy[:, 0] >= xmin) & (xy[:, 0] < xmax) &
                      (xy[:, 1] >= ymin) & (xy[:, 1] < ymax))
            if tissue is not None:
                inside &= tissue
            n = int(inside.sum())
            if n < self.min_cells:
                continue

            values: list[float] = []
            simple_values: dict[str, float] = {}
            if self.local_features_ is not None:
                cell_ids = region.coordinates.index[inside]
                simple_values = self.local_features_.extract_cells(region, cell_ids)
            counts = pd.Series(dtype=float)
            if labels is not None:
                counts = labels.iloc[inside].dropna().astype(str).value_counts()
            if "composition" in self.feature_groups:
                values.extend(
                    simple_values[f"composition__{cell_type}"]
                    for cell_type in self.cell_types_
                )
            if "density" in self.feature_groups:
                # coordinates are interpreted in microns, as elsewhere in benchmark
                area_mm2 = (self.window_size * self.window_size) / 1_000_000.0
                values.append(float(n / area_mm2))
            if "expression" in self.feature_groups:
                values.extend(
                    simple_values[f"expression__{marker}"]
                    for marker in self.markers_
                )
            if "entropy" in self.feature_groups:
                p = counts.to_numpy(float)
                p = p / p.sum() if p.sum() else p
                values.append(float(-(p * np.log(p + 1e-12)).sum()))
            rows.append(values)
            centers.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))

        d = len(self.feature_names())
        bag = np.asarray(rows, dtype=np.float32).reshape(-1, d)
        center_array = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        return {"bag": bag, "instance_centers": center_array}
