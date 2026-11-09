"""Leave-one-out window scores match the drop-one formulas."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from benchmark.models.additive_mil import AdditiveMILModel
from benchmark.models.attention_mil import _GatedAttentionNet
from benchmark.models.l1_mil import L1MILModel
from benchmark.motifs.mlp_ig import WindowMLP


def _bags(n: int = 8, windows: int = 5, width: int = 3):
    rng = np.random.default_rng(0)
    bags = [rng.normal(size=(windows, width)).astype(np.float32) for _ in range(n)]
    y = pd.Series([0, 1] * (n // 2))
    return bags, pd.DataFrame({"bag": bags}, index=y.index), y


class LeaveOneOutTest(unittest.TestCase):
    def test_l1_matches_mean_drop(self):
        bags, feat, y = _bags()
        model = L1MILModel(
            epochs=2, patience=0, max_instances=None, seed=0, l1_lambda=1e-5,
        ).fit(feat, y)
        p = model.window_probabilities(bags[0])
        got = model.leave_one_out_scores(bags[0])
        expected = p.mean() - (p.sum() - p) / (len(p) - 1)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        single = model.leave_one_out_scores(bags[0][:1])
        self.assertTrue(np.isnan(single).all())

    def test_mlp_matches_sigmoid_mean_drop(self):
        bags, feat, y = _bags()
        model = WindowMLP(
            seed=0, epochs=2, patience=0, hidden_dim=8, dropout=0.0,
        ).fit(feat, y.astype(float), columns=["a", "b", "c"])
        logits = model.window_logits(bags[0])
        got = model.leave_one_out_scores(bags[0])
        full = 1.0 / (1.0 + np.exp(-logits.mean()))
        held = (logits.sum() - logits) / (len(logits) - 1)
        expected = full - 1.0 / (1.0 + np.exp(-held))
        np.testing.assert_allclose(got, expected, atol=1e-5)
        self.assertEqual(model.l1_lambda, 0.0)

    def test_gated_matches_renormalized_attention(self):
        torch.manual_seed(0)
        net = _GatedAttentionNet(4, 8, 4, 1, dropout=0.5)
        bag = torch.randn(6, 4)
        net.eval()
        with torch.no_grad():
            got = net.leave_one_out(bag)
            h = net.encoder(bag)
            energy = net.attn_w(
                torch.tanh(net.attn_v(h)) * torch.sigmoid(net.attn_u(h))
            ).squeeze(-1)
            full_w = torch.softmax(energy, dim=0)
            full = torch.sigmoid(net.head((full_w[:, None] * h).sum(0)).reshape(-1)[0])
            expected = []
            for j in range(bag.shape[0]):
                keep = torch.ones(bag.shape[0], dtype=torch.bool)
                keep[j] = False
                weight = torch.softmax(energy[keep], dim=0)
                pooled = (weight[:, None] * h[keep]).sum(0)
                partial = torch.sigmoid(net.head(pooled).reshape(-1)[0])
                expected.append(full - partial)
        torch.testing.assert_close(got, torch.stack(expected), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isnan(net.leave_one_out(bag[:1])).all())

    def test_additive_penalty_is_zero_and_loo_is_finite(self):
        bags, feat, y = _bags(width=4, windows=6)
        model = AdditiveMILModel(
            epochs=2, patience=0, max_instances=None, seed=0,
            hidden_dim=8, attention_dim=4, dropout=0.5,
        ).fit(feat, y)
        self.assertEqual(float(model._penalty().detach().cpu()), 0.0)
        scores = model.leave_one_out_scores(bags[0])
        self.assertEqual(scores.shape, (len(bags[0]),))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
