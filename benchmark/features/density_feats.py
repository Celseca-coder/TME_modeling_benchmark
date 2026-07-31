"""Cell-type density features (counts per mm^2) using the tissue & tumour masks.

Unlike :class:`~benchmark.features.basic_feats.CompositionFeaturizer` (which gives
*fractions*, unitless), this featurizer produces **physical densities** — cells per mm^2 —
by dividing cell counts by the mask areas carried on :class:`RegionData` (the exported
``tissue`` / ``tumour`` polygons; see the tissue_area_estimation notebook).

Per region it emits three groups of features:

* ``tissue_density::<cell_type>`` — density of each cell type in the **tissue** foreground
  (cells inside the tissue mask, divided by tissue area);
* ``tumor_area_ratio`` — tumour area / tissue area (a compartment-size summary);
* ``tumor_density::<cell_type>`` — density of each cell type in the **tumour** compartment
  (only cells inside the tumour mask, divided by tumour area).

Requires regions loaded with their polygons (``TMEDataset.load_region`` loads them
automatically when the GeoJSON is present). Regions without a tissue mask yield all-NaN
rows (imputed model-side); regions with a tissue mask but no tumour bulk get
``tumor_area_ratio = 0`` and zero tumour densities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from .base import BaseFeatureExtractor

TISSUE = "tissue"
TUMOR = "tumour"


class CellTypeDensityFeaturizer(BaseFeatureExtractor):
    """Per-cell-type densities (per mm^2) in the tissue and tumour masks.

    The cell-type vocabulary is learned from the regions passed to :meth:`fit` (the
    training split), so types absent there are ignored and types absent from a region are
    0. Pass ``cell_type_col`` to use a cross-cohort-harmonised column
    (e.g. ``cell_type_uniform``).
    """

    def __init__(self, cell_type_col: str = "cell_type") -> None:
        """Initialize the instance.
        
                Args:
                    cell_type_col (str): Name of the column containing cell type.
        
        Args:
            cell_type_col (str): Name of the column containing cell type."""
        self.cell_type_col = cell_type_col
        self.cell_types_: list[str] = []

    # -- vocabulary -------------------------------------------------------
    def _col(self, region: RegionData) -> str | None:
        """Execute the col operation.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    str | None: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        if self.cell_type_col in region.cell_types.columns:
            return self.cell_type_col
        if "cell_type" in region.cell_types.columns:
            return "cell_type"
        return None

    def fit(self, regions: list[RegionData]) -> "CellTypeDensityFeaturizer":
        """Fit.
        
                Args:
                    regions (list[RegionData]): Tissue regions used for fitting or feature extraction.
        
                Returns:
                    'CellTypeDensityFeaturizer': The operation result.
        
        Args:
            regions (list[RegionData]): Tissue regions used for fitting or feature extraction."""
        types: set[str] = set()
        for r in regions:
            col = self._col(r)
            if col is None:
                raise ValueError(f"Region {r.region_id} has no cell-type column")
            types.update(r.cell_types[col].dropna().unique())
        self.cell_types_ = sorted(types)
        return self

    # -- feature names ----------------------------------------------------
    def feature_names(self) -> list[str]:
        """Execute the feature names operation.

        Returns:
            list[str]: The operation result."""
        names = [f"tissue_density::{ct}" for ct in self.cell_types_]
        names.append("tumor_area_ratio")
        names += [f"tumor_density::{ct}" for ct in self.cell_types_]
        return names

    # -- extraction -------------------------------------------------------
    def _thousand_per_mm2(self, labels: pd.Series, inside: np.ndarray,
                        area_mm2: float | None) -> dict[str, float]:
        """{cell_type: density} for cells with ``inside`` True, over the full vocabulary.
        
        Args:
            labels (pd.Series): Cell-type or class label assigned to each observation.
            inside (np.ndarray): Boolean mask indicating points inside the target area.
            area_mm2 (float | None): Area measured in square millimeters."""
        if area_mm2 is None or area_mm2 <= 0 or inside is None:
            return {ct: np.nan for ct in self.cell_types_}
        counts = labels[inside].value_counts()
        return {ct: (float(counts.get(ct, 0)) / 1000) / area_mm2 for ct in self.cell_types_}

    def extract_region(self, region: RegionData) -> dict[str, float]:
        """Extract region.
        
                Args:
                    region (RegionData): Region whose cells and spatial measurements are processed.
        
                Returns:
                    dict[str, float]: The operation result.
        
        Args:
            region (RegionData): Region whose cells and spatial measurements are processed."""
        col = self._col(region)
        # labels aligned to the coordinate/mask order
        labels = region.cell_types[col].reindex(region.coordinates.index).astype("object")
        xy = region.coordinates[["x", "y"]].to_numpy(float)

        tissue_area = region.tissue_area_mm2
        tumor_area = region.tumor_area_mm2
        in_tissue = region.polygon_contains(xy, TISSUE)  # None if no tissue mask
        in_tumor = region.polygon_contains(xy, TUMOR)    # None if no tumour mask

        feats: dict[str, float] = {}

        # (1) tissue densities — NaN throughout if there is no usable tissue mask
        tissue_dens = self._thousand_per_mm2(labels, in_tissue, tissue_area)
        for ct in self.cell_types_:
            feats[f"tissue_density::{ct}"] = tissue_dens[ct]

        tissue_ok = tissue_area is not None and tissue_area > 0 and in_tissue is not None

        # (2) tumour area ratio & (3) tumour densities
        if not tissue_ok:
            tumor_area_ratio = np.nan
            tumor_dens = {ct: np.nan for ct in self.cell_types_}
        elif tumor_area is None:
            # tissue but no tumour bulk
            tumor_area_ratio = 0.0
            tumor_dens = {ct: np.nan for ct in self.cell_types_}
        else:
            tumor_area_ratio = tumor_area / tissue_area
            tumor_dens = self._thousand_per_mm2(labels, in_tumor, tumor_area)

        feats["tumor_area_ratio"] = tumor_area_ratio
        for ct in self.cell_types_:
            feats[f"tumor_density::{ct}"] = tumor_dens[ct]

        return feats
