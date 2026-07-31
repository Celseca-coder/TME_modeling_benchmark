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
        """Initialize the instance.
        
                Args:
                    seed (int): Random seed used for reproducibility.
                    n_estimators (int): Number of estimators.
                    max_features (str | float): Maximum allowed features.
                    class_weight (str | None): Weight applied to class.
        
        Args:
            seed (int): Random seed used to make results reproducible."""
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.class_weight = class_weight
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RandomForestModel":
        """Fit.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
                    target (pd.Series): Target labels or outcomes associated with the samples.
        
                Returns:
                    'RandomForestModel': The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
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
        """Predict.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        return self._clf.predict_proba(self._prep_pred(features))
