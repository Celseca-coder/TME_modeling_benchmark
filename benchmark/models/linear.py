"""Linear region-level models: regularised logistic regression and Cox regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _TabularModel


class LinearClassifier(_TabularModel):
    """Logistic regression with L1 / L2 / elastic-net regularisation.

    ``l1_ratio`` mixes the penalty: 0.0 = pure L2 (ridge), 1.0 = pure L1 (lasso),
    in between = elastic net. ``C`` is the inverse regularisation strength.
    Handles binary and multiclass targets.
    """

    def __init__(self, seed: int = 0, C: float = 1.0, l1_ratio: float = 0.0,
                 class_weight: str | None = "balanced", max_iter: int = 5000) -> None:
        self.seed = seed
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LinearClassifier":
        from sklearn.linear_model import LogisticRegression
        Xs = self._prep_fit(features.loc[list(target.index)])
        if self.l1_ratio <= 0.0:                       # pure L2: fast lbfgs solver
            kw = dict(solver="lbfgs", penalty="l2")
        elif self.l1_ratio >= 1.0:                     # pure L1: saga + l1
            kw = dict(solver="saga", penalty="l1")
        else:                                          # elastic net
            kw = dict(solver="saga", penalty="elasticnet", l1_ratio=self.l1_ratio)
        self._clf = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
            random_state=self.seed, **kw,
        )
        self._clf.fit(Xs, target.values)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(self._prep_pred(features))


class LinearCox(_TabularModel):
    """Ridge-penalised Cox proportional-hazards regression (lifelines)."""

    task_type = "survival"

    def __init__(self, seed: int = 0, penalizer: float = 0.1) -> None:
        self.seed = seed
        self.penalizer = penalizer

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "LinearCox":
        from lifelines import CoxPHFitter
        Xs = self._prep_fit(features.loc[list(target.index)])
        data = pd.DataFrame(Xs, columns=self._keep, index=target.index)
        data["time"] = target["time"].values
        data["event"] = target["event"].values
        self._model = CoxPHFitter(penalizer=self.penalizer)
        self._model.fit(data, duration_col="time", event_col="event")
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        Xs = pd.DataFrame(self._prep_pred(features), columns=self._keep, index=features.index)
        return self._model.predict_partial_hazard(Xs).values
