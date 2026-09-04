"""Interpretable local-window bags for attention multiple-instance learning.

Each region is represented by a variable-length bag.  An instance is a square
spatial window and its features are deliberately simple: cell-type composition,
cell density, mean marker expression, and cell-type entropy.  The returned
``DataFrame`` contains object-valued bags so it can pass through the unchanged
benchmark cross-validation code to ``AttentionMILModel``.
"""
from __future__ import annotations

import numpy as np

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor
from .local_window import VALID_WINDOW_GROUPS, LocalWindowInstanceFeaturizer


class HandcraftedAttentionMILFeaturizer(BaseFeatureExtractor):
    """Build one variable-length local-feature bag per region.

    Vocabulary discovery happens only in ``fit`` (inside each training fold), so
    validation/test regions cannot leak cell types or markers into the model.
    """

    VALID_GROUPS = VALID_WINDOW_GROUPS

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
        self.instances_: LocalWindowInstanceFeaturizer | None = None

    def fit(self, regions: list[RegionData]) -> "HandcraftedAttentionMILFeaturizer":
        self.instances_ = LocalWindowInstanceFeaturizer(
            feature_groups=self.feature_groups,
            cell_type_col=self.cell_type_col,
            max_markers=self.max_markers,
        ).fit(regions)
        self.cell_types_ = self.instances_.cell_types_
        self.markers_ = self.instances_.markers_
        if not self.feature_names():
            raise ValueError("No local handcrafted features are available")
        return self

    def feature_names(self) -> list[str]:
        if self.instances_ is None:
            return []
        return self.instances_.feature_names()

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
        if self.instances_ is None:
            raise RuntimeError("HandcraftedAttentionMILFeaturizer must be fitted first")
        rows, centers = [], []
        for xmin, ymin, xmax, ymax in self._windows(region):
            inside = ((xy[:, 0] >= xmin) & (xy[:, 0] < xmax) &
                      (xy[:, 1] >= ymin) & (xy[:, 1] < ymax))
            if tissue is not None:
                inside &= tissue
            n = int(inside.sum())
            if n < self.min_cells:
                continue
            rows.append(self.instances_.extract_vector(region, inside, self.window_size))
            centers.append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))

        d = len(self.feature_names())
        bag = np.asarray(rows, dtype=np.float32).reshape(-1, d)
        center_array = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        return {"bag": bag, "instance_centers": center_array}
