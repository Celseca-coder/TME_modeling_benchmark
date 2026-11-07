"""Additive MIL attention weights form a distribution over windows."""
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
    def test_attention_weights_sum_to_one(self):
        model, bag = _fitted()
        weights = model.attention_weights(pd.DataFrame({"bag": [bag]}))[0]
        self.assertEqual(weights.shape, (len(bag),))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights >= -1e-6))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
