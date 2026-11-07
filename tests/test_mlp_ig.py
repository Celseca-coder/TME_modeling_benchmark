"""Integrated Gradients of a mean-pooled window MLP."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from benchmark.motifs.mlp_ig import (
    WindowMLP,
    global_feature_importance,
    integrated_gradients,
    mean_feature_rank,
    recovery_at_k,
)


def _fitted() -> tuple[WindowMLP, np.ndarray]:
    rng = np.random.default_rng(0)
    bags = []
    labels = []
    for _ in range(8):
        x = rng.normal(size=(5, 3)).astype(np.float32)
        bags.append(x)
        labels.append(int(x[:, 0].mean() > 0))
    features = pd.DataFrame({"bag": bags})
    y = pd.Series(labels)
    model = WindowMLP(
        seed=0, epochs=8, patience=3, hidden_dim=8, dropout=0.0,
    ).fit(features, y, columns=["a", "b", "c"])
    return model, bags[0]


class MlpIgTest(unittest.TestCase):
    def test_window_score_is_absolute_ig(self):
        model, bag = _fitted()
        ig, scores = integrated_gradients(model, bag, steps=32)
        self.assertEqual(ig.shape, (len(bag), len(model._columns)))
        np.testing.assert_allclose(scores, np.abs(ig).sum(axis=1))
        importance = global_feature_importance(ig)
        self.assertEqual(importance.shape, (len(model._columns),))
        np.testing.assert_allclose(importance, np.abs(ig).mean(axis=0))

    def test_completeness_improves_with_steps(self):
        model, bag = _fitted()
        model._net.eval()
        x = model._tensor(bag)

        def gap_of(steps: int) -> float:
            ig, _ = integrated_gradients(model, bag, steps=steps)
            with torch.inference_mode():
                fx = float(torch.sigmoid(model._net(x).mean()).cpu())
                f0 = float(torch.sigmoid(model._net(torch.zeros_like(x)).mean()).cpu())
            gap = fx - f0
            return abs(float(ig.sum()) - gap) / (abs(gap) + 1e-6)

        self.assertLess(gap_of(64), gap_of(8))

    def test_mean_rank_and_hit(self):
        ranking = mean_feature_rank([
            ["composition__tumor", "mixing__local_mixing_mean", "other"],
            ["composition__tumor", "other", "mixing__local_mixing_mean"],
        ])
        self.assertEqual(ranking.iloc[0]["feature"], "composition__tumor")
        rec = recovery_at_k(ranking["feature"].tolist(), "tumor_high", k=5)
        self.assertEqual(rec["hit_at_k"], 1)
        self.assertGreater(rec["precision_at_k"], 0)


if __name__ == "__main__":
    unittest.main()
