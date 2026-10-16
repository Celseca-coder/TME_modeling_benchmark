from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from benchmark.motifs.window_extractor_eval import (
    join_features_and_labels,
    topk_labeled_positions,
    within_region_auroc,
    within_region_precision_at_frac,
)


class WindowExtractorEvalTest(unittest.TestCase):
    def test_join_uses_window_label_not_region(self):
        feats = pd.DataFrame({
            "region_id": ["r1", "r1"],
            "window_id": [0, 1],
            "composition__CD8": [0.1, 0.8],
        })
        wins = pd.DataFrame({
            "region_id": ["r1", "r1"],
            "window_id": [0, 1],
            "motif_id": ["cd8_high", "cd8_high"],
            "label": [0.0, 1.0],
        })
        tab = join_features_and_labels(feats, wins, "cd8_high")
        self.assertEqual(list(tab["y_w"]), [0.0, 1.0])

    def test_within_region_not_pooled(self):
        y = pd.Series([0, 1, 0, 1, 0, 1] * 2, dtype=float)
        s = pd.Series([0.1, 0.9, 0.2, 0.8, 0.15, 0.85] * 2)
        r = pd.Series(["a"] * 6 + ["b"] * 6)
        rec = within_region_auroc(y, s, r)
        self.assertGreater(rec["auroc_window_mean"], 0.9)
        self.assertTrue(rec["pass_window"])

    def test_skips_unlabeled_and_single_class(self):
        y = pd.Series([np.nan, np.nan, 1, 1], dtype=float)
        s = pd.Series([0.1, 0.9, 0.2, 0.8])
        r = pd.Series(["a", "a", "b", "b"])
        rec = within_region_auroc(y, s, r)
        self.assertEqual(rec["n_regions_scored"], 0)

    def test_precision_at_20_counts_top_fraction(self):
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=float)
        s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.15, 0.9])
        r = pd.Series(["a"] * 10)
        rec = within_region_precision_at_frac(y, s, r, frac=0.20)
        self.assertEqual(rec["n_regions_scored"], 1)
        self.assertEqual(rec["precision_at_20_mean"], 1.0)
        top = topk_labeled_positions(y.to_numpy(float), s.to_numpy(float), frac=0.20)
        self.assertEqual(set(top.tolist()), {7, 9})