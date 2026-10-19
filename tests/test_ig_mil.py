"""Integrated Gradients window scores for gated-attention MIL."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.models.attention_mil import AttentionMILModel
from benchmark.motifs.ig_mil import ig_completeness_error, ig_window_importance


def _fitted_model() -> tuple[AttentionMILModel, np.ndarray]:
    rng = np.random.default_rng(0)
    bags = [rng.normal(size=(5, 3)).astype(np.float32) for _ in range(8)]
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    feat = pd.DataFrame({"bag": bags}, index=y.index)
    model = AttentionMILModel(
        epochs=1, patience=0, max_instances=None, seed=0,
        hidden_dim=8, attention_dim=4,
    ).fit(feat, y)
    return model, bags[0]


class IgMilTest(unittest.TestCase):
    def test_scores_match_window_count_and_completeness(self):
        model, bag = _fitted_model()
        coarse = ig_window_importance(model, bag, steps=32)
        fine = ig_window_importance(model, bag, steps=256)
        self.assertEqual(fine.shape, (len(bag),))
        self.assertTrue(np.isfinite(fine).all())
        err_coarse = ig_completeness_error(model, bag, coarse)
        err_fine = ig_completeness_error(model, bag, fine)
        self.assertLess(err_fine, err_coarse)
        self.assertLess(err_fine, 0.05)

    def test_empty_bag(self):
        model, _ = _fitted_model()
        empty = np.zeros((0, 3), dtype=np.float32)
        scores = ig_window_importance(model, empty, steps=4)
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(ig_completeness_error(model, empty, scores), 0.0)

    def test_requires_full_bag(self):
        model, bag = _fitted_model()
        model.max_instances = 2
        with self.assertRaises(ValueError):
            ig_window_importance(model, bag, steps=2)


if __name__ == "__main__":
    unittest.main()
