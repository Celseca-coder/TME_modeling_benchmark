"""Window scores from a random forest trained on window pseudo-labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.models.forest import RandomForestModel


def rf_window_scores(model: RandomForestModel, features: pd.DataFrame) -> np.ndarray:
    """``s_w = P(y=1 | x_w)``, the mean positive probability across trees."""
    proba = np.asarray(model.predict(features), dtype=float)
    classes = list(model.classes_)
    if 1 in classes:
        column = classes.index(1)
    elif 1.0 in classes:
        column = classes.index(1.0)
    else:
        column = len(classes) - 1
    return proba[:, column]
