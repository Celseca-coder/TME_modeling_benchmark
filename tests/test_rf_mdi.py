"""Random-forest impurity importance as a window score."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.models.forest import RandomForestModel
from benchmark.motifs.rf_mdi import mdi_window_scores


class RfMdiTest(unittest.TestCase):
    def test_score_is_importance_weighted_feature_sum(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 2))
        y = pd.Series((x[:, 0] > 0).astype(float))
        features = pd.DataFrame(x, columns=["a", "b"])
        model = RandomForestModel(
            seed=0, n_estimators=20, class_weight=None,
        ).fit(features, y)
        scores = mdi_window_scores(model, features)
        raw = features[model._keep].fillna(model._median[model._keep]).to_numpy(float)
        expected = raw @ model._clf.feature_importances_
        np.testing.assert_allclose(scores, expected)
        self.assertAlmostEqual(float(model._clf.feature_importances_.sum()), 1.0)
        self.assertGreater(
            model._clf.feature_importances_[model._keep.index("a")],
            model._clf.feature_importances_[model._keep.index("b")],
        )


if __name__ == "__main__":
    unittest.main()
