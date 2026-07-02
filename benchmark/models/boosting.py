"""Gradient-boosting region-level model (the XGBoost-equivalent)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _TabularModel


class GradientBoostingModel(_TabularModel):
    """Histogram gradient-boosting classifier (binary + multiclass).

    Uses :class:`~sklearn.ensemble.HistGradientBoostingClassifier` — the same
    histogram-based gradient boosting as XGBoost/LightGBM, with no extra dependency
    — so it is a drop-in "XGBoost-equivalent" alongside ``RandomForestModel``.
    Reuses the shared tabular preprocessing (median-impute + drop zero-variance);
    standardisation is harmless for trees. Survival is not supported.

    ``class_weight="balanced"`` re-weights samples inversely to class frequency
    (HistGB has no native ``class_weight``, so it is applied via ``sample_weight``).
    """

    def __init__(self, seed: int = 0, n_estimators: int = 300,
                 learning_rate: float = 0.1, max_depth: int | None = None,
                 class_weight: str | None = "balanced") -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "GradientBoostingModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xs = self._prep_fit(features.loc[list(target.index)])
        sw = None
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", target.values)
        self._clf = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_iter=self.n_estimators,
            max_depth=self.max_depth, random_state=self.seed,
        )
        self._clf.fit(Xs, target.values, sample_weight=sw)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(self._prep_pred(features))
