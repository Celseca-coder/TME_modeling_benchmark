"""Region-level model interface + a shared tabular-preprocessing mixin."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RegionModel(ABC):
    """Maps a region-level feature table to predictions.

    ``fit(features, target)``
      * ``features`` : DataFrame indexed by ``region_id`` (regions × features).
      * ``target``   : region-indexed Series of 0/1 or class labels
        (classification), or a DataFrame with ``time`` / ``event`` (survival).

    ``predict(features)`` returns, aligned to ``features`` rows:
      * classification — a probability matrix ``(n, n_classes)`` ordered by
        ``classes_``;
      * survival       — a 1-D risk score (higher = worse prognosis).
    """

    task_type: str = "base"   # 'binary' | 'multiclass' | 'survival'
    classes_ = None

    @abstractmethod
    def fit(self, features: pd.DataFrame, target) -> "RegionModel":
        ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        ...


class _TabularModel(RegionModel):
    """Shared preprocessing for tabular models: median-impute NaNs, drop
    zero-variance columns, and standardise on the training statistics."""

    def _prep_fit(self, X: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        self._median = X.median()
        Xf = X.fillna(self._median)
        std = Xf.std()
        self._keep = std[std > 0].index.tolist()
        if not self._keep:
            raise ValueError("no non-constant features")
        self._scaler = StandardScaler().fit(Xf[self._keep])
        return self._scaler.transform(Xf[self._keep])

    def _prep_pred(self, X: pd.DataFrame) -> np.ndarray:
        Xp = X[self._keep].fillna(self._median[self._keep])
        return self._scaler.transform(Xp)
