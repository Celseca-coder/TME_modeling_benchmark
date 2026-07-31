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
        """Initialize the instance.
        
                Args:
                    seed (int): Random seed used for reproducibility.
                    C (float): Inverse regularization strength used by the estimator.
                    l1_ratio (float): Ratio controlling l1.
                    class_weight (str | None): Weight applied to class.
                    max_iter (int): Maximum allowed iter.
        
        Args:
            seed (int): Random seed used to make results reproducible."""
        self.seed = seed
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.task_type = "binary"
        self.classes_ = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LinearClassifier":
        """Fit.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
                    target (pd.Series): Target labels or outcomes associated with the samples.
        
                Returns:
                    'LinearClassifier': The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
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
        """Predict.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        return self._clf.predict_proba(self._prep_pred(features))


class LinearCox(_TabularModel):
    """Ridge-penalised Cox proportional-hazards regression (lifelines)."""

    task_type = "survival"

    def __init__(self, seed: int = 0, penalizer: float = 0.1) -> None:
        """Initialize the instance.
        
                Args:
                    seed (int): Random seed used for reproducibility.
                    penalizer (float): Value controlling or representing penalizer.
        
        Args:
            seed (int): Random seed used to make results reproducible."""
        self.seed = seed
        self.penalizer = penalizer

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> "LinearCox":
        """
        Fit.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
                    target (pd.DataFrame): Target labels or outcomes associated with the samples.
        
                Returns:
                    'LinearCox': The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model.
        Basic Fomula
        Cox 模型学习的是：
        \[
        h(t \mid x)=h_0(t)\exp(\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p)
        \]这里：
        \(x_1,x_2,\dots,x_p\) 是 CN 特征；
        \(\beta_1,\beta_2,\dots,\beta_p\) 是模型学习的系数；
        \(h(t\mid x)\) 是患者在时间 \(t\) 附近发生事件的相对风险；
        \(h_0(t)\) 是基线风险。
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
        """Predict.
        
                Args:
                    features (pd.DataFrame): Feature values used to fit or evaluate the model.
        
                Returns:
                    np.ndarray: The operation result.
        
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model."""
        Xs = pd.DataFrame(self._prep_pred(features), columns=self._keep, index=features.index)
        return self._model.predict_partial_hazard(Xs).values
