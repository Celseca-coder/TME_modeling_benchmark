"""Neighborhood multiple-instance learning (MIL) region-level model.

A thin wrapper that turns *any* region featurizer + region classifier into a
neighborhood-MIL model. Bags are **regions**; instances are **local neighborhoods**
sampled around focal cells via :meth:`RegionData.neighborhood` (which reuses
:meth:`RegionData.subset`). The wrapper only does the MIL-specific plumbing —
sample neighborhoods, label each with its region, pool the per-instance predictions
— and delegates all the actual learning to the model instance you pass in:

    NeighborhoodMILModel(
        featurizer=CompositionFeaturizer(cell_type_col="cell_type"),
        model=GradientBoostingModel(seed=0),   # or RandomForestModel 
        n_samples=16,
        radius_um=100.0,
    )

Both ``featurizer`` and ``model`` are plain instances (not factories). ``fit`` fits
the featurizer's vocabulary on the training regions, builds the neighborhood-instance
table, and calls ``model.fit``; ``predict`` builds the instance table for each region,
calls ``model.predict``, and mean-pools the instance probabilities back to one row per
region. Classification only (binary / multiclass).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark.data.dataset import RegionData
from .base import RegionModel


class NeighborhoodMILModel(RegionModel):
    """Sample-neighborhoods-and-pool MIL wrapper around a region model.

    Parameters
    ----------
    featurizer : BaseFeatureExtractor instance
        Featurizes one neighborhood (a ``RegionData``) via ``extract_region``. Its
        vocabulary is (re)fit on the training regions in :meth:`fit` — no leakage.
    model : RegionModel instance
        The instance-level classifier, e.g. ``GradientBoostingModel(seed=0)``,
        ``RandomForestModel(...)`` or ``LinearClassifier(...)``. Used as-is.
    n_samples : int
        Neighborhoods sampled per region (focal cells drawn at random without
        replacement, capped at the region's cell count).
    sample_ratio : float | None
        If set, sample ``ceil(sample_ratio * n_cells)`` neighborhoods per region
        instead of a fixed ``n_samples``.
    radius_um : float
        Neighborhood radius in microns (passed to ``neighborhood``).
    min_cells : int
        Skip neighborhoods with fewer than this many cells.
    agg : {"mean", "max"}
        How per-instance probabilities are pooled into the region probability.
    seed : int
        Seeds the focal-cell sampling.
    """

    def __init__(self, featurizer, model: RegionModel, *,
                 n_samples: float = 16, radius_um: float = 100.0,
                 min_cells: int = 10, agg: str = "mean", seed: int = 0) -> None:
        self.featurizer = featurizer
        self.model = model
        self.n_samples = n_samples
        self.radius_um = radius_um
        self.min_cells = min_cells
        self.agg = agg
        self.seed = seed
        self.task_type = model.task_type

    # ------------------------------------------------------------------
    def _centers(self, region: RegionData, rng, n_samples=None) -> list:
        ids = region.coordinates.index.to_numpy()
        n = len(ids)
        if n == 0:
            return []
        if n_samples is None:
            n_samples = self.n_samples

        if n_samples <= 1:
            k = int(np.ceil(self.n_samples * n))
        else:
            k = int(self.n_samples)
        k = max(1, min(k, n))

        if k >= n:
            return ids.tolist()
        return rng.choice(ids, size=k, replace=False).tolist()

    def _instances(self, regions: list[RegionData], rng, n_samples=None):
        """Featurize sampled neighborhoods -> (instance feature table, region-id index)."""
        rows, idx, rid = [], [], []
        for r in regions:
            for j, c in enumerate(self._centers(r, rng, n_samples=n_samples)):
                nb = r.neighborhood(c, radius_um=self.radius_um)
                if nb.n_cells < self.min_cells:
                    continue
                rows.append(self.featurizer.extract_region(nb))
                idx.append(f"{r.region_id}#inst{j}")
                rid.append(r.region_id)
        X = pd.DataFrame(rows, index=idx)
        return X, pd.Index(rid, name="region_id")

    # ------------------------------------------------------------------
    # RegionModel API  (note: consumes list[RegionData], not a feature table)
    # ------------------------------------------------------------------
    def fit(self, regions: list[RegionData], target: pd.Series):
        rng = np.random.default_rng(self.seed)
        keep = [r for r in regions if r.region_id in target.index]
        X, rid = self._instances(keep, rng)
        if X.empty:
            raise ValueError("no neighborhoods could be sampled from the training regions")
        y = pd.Series(target.loc[rid].to_numpy(), index=X.index)
        self.model.fit(X, y)                               # delegate to the wrapped model
        self.classes_ = getattr(self.model, "classes_", None)
        return self

    def predict(self, regions: list[RegionData], n_samples: float | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1)
        X, rid = self._instances(regions, rng, n_samples=n_samples)
        n_classes = len(self.classes_)
        order = [r.region_id for r in regions]
        if X.empty:
            return np.full((len(regions), n_classes), 1.0 / n_classes)

        proba = pd.DataFrame(np.asarray(self.model.predict(X)), index=rid)
        pooled = proba.groupby(level=0).max() if self.agg == "max" else proba.groupby(level=0).mean()
        if self.agg == "max":                              # renormalise after max-pool
            pooled = pooled.div(pooled.sum(axis=1), axis=0)
        out = pooled.reindex(order)
        out = out.fillna(1.0 / n_classes)                  # regions with no valid instances
        return out.to_numpy()
