"""Window scores from random-forest mean decrease in impurity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.models.forest import RandomForestModel


def mdi_window_scores(model: RandomForestModel, features: pd.DataFrame) -> np.ndarray:
    """Score each window by ``s_w = sum_j I_j x_{w,j}``.

    ``I`` is ``feature_importances_`` (mean decrease in Gini impurity), aligned
    to the variance-filtered columns kept at fit time. ``x`` is the same
    median-imputed value the forest saw, before standardisation. Scaling a
    column does not change ``I``, but it would change this dot product.
    """
    X = features[model._keep].fillna(model._median[model._keep])
    importance = np.asarray(model._clf.feature_importances_, dtype=float)
    return np.asarray(X, dtype=float) @ importance
