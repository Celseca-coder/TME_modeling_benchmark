"""Integrated Gradients of a window MLP."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.motifs.mlp_ig import (
    WindowMLP,
    global_feature_importance,
    integrated_gradients,
    mean_feature_rank,
    recovery_at_k,
)


def _fitted() -> tuple[WindowMLP, pd.DataFrame]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 3))
    y = pd.Series((x[:, 0] > 0).astype(float))
    features = pd.DataFrame(x, columns=["a", "b", "c"])
    model = WindowMLP(seed=0, epochs=8, patience=3, batch_size=16, hidden_dim=8).fit(features, y)
    return model, features


class MlpIgTest(unittest.TestCase):
    def test_window_score_sums_feature_ig(self):
        model, features = _fitted()
        ig, scores = integrated_gradients(model, features, steps=64, chunk=16)
        self.assertEqual(ig.shape, (len(features), len(model._columns)))
        np.testing.assert_allclose(scores, ig.sum(axis=1))
        importance = global_feature_importance(ig)
        self.assertEqual(importance.shape, (len(model._columns),))
        np.testing.assert_allclose(importance, np.abs(ig).mean(axis=0))

    def test_completeness_improves_with_steps(self):
        model, features = _fitted()
        x = model._scale_matrix(features)
        pos = model._positive_index()
        model._net.eval()
        import torch
        xt = torch.as_tensor(x, dtype=torch.float32, device=model.device_)
        with torch.inference_mode():
            gap = (
                model._net(xt)[:, pos] - model._net(torch.zeros_like(xt))[:, pos]
            ).detach().cpu().numpy()
        coarse, _ = integrated_gradients(model, features, steps=16, chunk=16)
        fine, _ = integrated_gradients(model, features, steps=128, chunk=16)
        err = lambda ig: np.mean(np.abs(ig.sum(axis=1) - gap) / (np.abs(gap) + 1e-6))
        self.assertLess(err(fine), err(coarse))
        self.assertLess(err(fine), 0.05)

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
