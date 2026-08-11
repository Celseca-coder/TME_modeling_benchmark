import argparse
import unittest

import numpy as np
import pandas as pd

from scripts.run_shap_analysis import (
    _linear_classification_shap,
    _normalise_tree_output,
    _summarise_features,
)


class ShapAnalysisTest(unittest.TestCase):
    def test_logistic_shap_recovers_signal_direction(self):
        rng = np.random.default_rng(4)
        signal = np.r_[-np.ones(20), np.ones(20)]
        train = np.column_stack([signal, rng.normal(size=40)])
        target = pd.Series(np.r_[np.zeros(20), np.ones(20)].astype(int))
        val = np.array([[-1.0, 0.0], [1.0, 0.0]])
        args = argparse.Namespace(
            logistic_penalty="l1",
            l1_ratio=0.5,
            C=10.0,
            no_class_weight=False,
            max_iter=5000,
            model_seed=0,
        )

        classes, values = _linear_classification_shap(train, val, target, args)

        self.assertEqual(classes, ["1"])
        self.assertLess(values[0][0, 0], 0)
        self.assertGreater(values[0][1, 0], 0)

    def test_tree_output_normalises_feature_and_class_axes(self):
        feature_last = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        arrays = _normalise_tree_output(feature_last, n_features=3, n_classes=4)
        self.assertEqual(len(arrays), 4)
        self.assertEqual(arrays[0].shape, (2, 3))

        class_middle = np.arange(2 * 4 * 3).reshape(2, 4, 3)
        arrays = _normalise_tree_output(class_middle, n_features=3, n_classes=4)
        self.assertEqual(len(arrays), 4)
        self.assertEqual(arrays[0].shape, (2, 3))

    def test_feature_summary_reports_fold_stability(self):
        fold = pd.DataFrame({
            "dataset": ["d", "d"],
            "task": ["t", "t"],
            "scheme": ["cv", "cv"],
            "model": ["logistic", "logistic"],
            "class": ["1", "1"],
            "seed": [0, 0],
            "fold": [0, 1],
            "feature": ["signal", "signal"],
            "mean_abs_shap": [2.0, 4.0],
            "mean_shap": [0.2, 0.4],
            "value_shap_spearman": [0.8, 0.6],
            "top_feature": [True, True],
            "n_explained_regions": [10, 10],
        })

        summary = _summarise_features(fold).iloc[0]

        self.assertEqual(summary["mean_abs_shap"], 3.0)
        self.assertEqual(summary["top_fold_frequency"], 1.0)
        self.assertEqual(summary["direction_consistency"], 1.0)
        self.assertEqual(summary["direction"], "higher value increases prediction")


if __name__ == "__main__":
    unittest.main()
