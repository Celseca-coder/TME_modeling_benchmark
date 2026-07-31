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
        """Initialize the regularized logistic-regression model.

        Args:
            seed: Random seed used by stochastic solvers.
            C: Inverse regularization strength.
            l1_ratio: Elastic-net mixing value; zero selects pure L2.
            class_weight: Class-weight strategy passed to scikit-learn.
            max_iter: Maximum solver iterations.
        """
        self.seed = seed
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LinearClassifier":
        """Fit logistic regression to region-level features.

        Args:
            features: Region-by-feature training table.
            target: Region-indexed class labels.

        Returns:
            This fitted classifier.
        """
        from sklearn.linear_model import LogisticRegression
        Xs = self._prep_fit(features.loc[list(target.index)])
        if self.l1_ratio == 0.0:                       # pure L2: fast lbfgs solver
            kw = dict(solver="lbfgs")
        else:                                          # L1 / elastic net: saga solver
            kw = dict(solver="saga", l1_ratio=self.l1_ratio)
        self._clf = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
            random_state=self.seed, **kw,
        )
        self._clf.fit(Xs, target.values)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for regions.

        Args:
            features: Region-by-feature table to score.

        Returns:
            Probability matrix whose columns follow ``classes_``.
        """
        return self._clf.predict_proba(self._prep_pred(features))


class LinearCox(_TabularModel):
    """Ridge-penalised Cox proportional-hazards regression (lifelines)."""

    task_type = "survival"

    def __init__(self, seed: int = 0, penalizer: float = 0.1) -> None:
        """Initialize the penalized Cox proportional-hazards model.

        Args:
            seed: Reproducibility seed retained for the common model interface.
            penalizer: L2 penalty applied by ``lifelines.CoxPHFitter``.
        """
        self.seed = seed
        self.penalizer = penalizer

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "LinearCox":
        """Fit Cox regression to features and censored survival outcomes.

        Args:
            features: Region-by-feature training table.
            target: Region-indexed table with ``time`` and ``event`` columns.

        Returns:
            This fitted survival model.
        """
        from lifelines import CoxPHFitter
        Xs = self._prep_fit(features.loc[list(target.index)])
        data = pd.DataFrame(Xs, columns=self._keep, index=target.index)
        data["time"] = target["time"].values
        data["event"] = target["event"].values
        self._model = CoxPHFitter(penalizer=self.penalizer)
        self._model.fit(data, duration_col="time", event_col="event")
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict relative hazard for each region.

        Args:
            features: Region-by-feature table to score.

        Returns:
            One-dimensional relative-risk array; larger values indicate worse
            predicted prognosis.
        """
        Xs = pd.DataFrame(self._prep_pred(features), columns=self._keep, index=features.index)
        return self._model.predict_partial_hazard(Xs).values
