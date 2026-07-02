"""Random-forest region-level model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _TabularModel


class RandomForestModel(_TabularModel):
    """Basic random-forest classifier (binary + multiclass).

    Reuses the shared tabular preprocessing (median-impute + drop zero-variance);
    standardisation is harmless for trees. Survival is not supported.
    """

    def __init__(self, seed: int = 0, n_estimators: int = 300,
                 max_features: str | float = "sqrt",
                 class_weight: str | None = "balanced") -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.class_weight = class_weight
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestClassifier
        Xs = self._prep_fit(features.loc[list(target.index)])
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_features=self.max_features,
            class_weight=self.class_weight, random_state=self.seed, n_jobs=-1,
        )
        self._clf.fit(Xs, target.values)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(self._prep_pred(features))
