"""Additive MIL window contributions sum to the bag logit."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.models.additive_mil import AdditiveMILModel


def _fitted() -> tuple[AdditiveMILModel, np.ndarray]:
    rng = np.random.default_rng(0)
    bags = [rng.normal(size=(6, 4)).astype(np.float32) for _ in range(8)]
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    feat = pd.DataFrame({"bag": bags}, index=y.index)
    model = AdditiveMILModel(
        epochs=2, patience=0, max_instances=None, seed=0,
        hidden_dim=8, attention_dim=4,
    ).fit(feat, y)
    return model, bags[0]


class AdditiveMilTest(unittest.TestCase):
    def test_contributions_sum_to_positive_logit(self):
        model, bag = _fitted()
        scores = model.window_contributions(bag)
        self.assertEqual(scores.shape, (len(bag),))
        self.assertTrue(np.isfinite(scores).all())
        logit = model.positive_logit(bag)
        gap = abs(float(scores.sum()) + model.positive_bias() - logit)
        self.assertLess(gap, 1e-4)


if __name__ == "__main__":
    unittest.main()
