"""Mean-pooled L1 MIL window scores are the per-window sigmoids."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from benchmark.models.l1_mil import L1MILModel


class L1MilTest(unittest.TestCase):
    def test_bag_probability_is_mean_of_window_scores(self):
        rng = np.random.default_rng(0)
        bags = [rng.normal(size=(5, 3)).astype(np.float32) for _ in range(8)]
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        feat = pd.DataFrame({"bag": bags}, index=y.index)
        model = L1MILModel(
            epochs=2, patience=0, max_instances=None, seed=0, l1_lambda=1e-5,
        ).fit(feat, y)
        scores = model.window_probabilities(bags[0])
        self.assertEqual(scores.shape, (len(bags[0]),))
        self.assertTrue(np.all((scores > 0.0) & (scores < 1.0)))
        model._net.eval()
        with torch.no_grad():
            pooled, _ = model._net(model._bag_tensor(bags[0]))
        self.assertAlmostEqual(float(scores.mean()), float(pooled.cpu()), places=5)


if __name__ == "__main__":
    unittest.main()
