"""Random-forest window score is the positive-class probability."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.models.forest import RandomForestModel
from benchmark.motifs.rf_mdi import mdi_window_scores


class RfMdiTest(unittest.TestCase):
    def test_score_is_positive_probability(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 2))
        y = pd.Series((x[:, 0] > 0).astype(float))
        features = pd.DataFrame(x, columns=["a", "b"])
        model = RandomForestModel(
            seed=0, n_estimators=20, class_weight=None,
        ).fit(features, y)
        scores = rf_window_scores(model, features)
        proba = model.predict(features)
        column = list(model.classes_).index(1.0)
        np.testing.assert_allclose(scores, proba[:, column])
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        self.assertGreater(float(scores[x[:, 0] > 0].mean()), float(scores[x[:, 0] <= 0].mean()))


if __name__ == "__main__":
    unittest.main()
