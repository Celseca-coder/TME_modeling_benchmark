import unittest

import numpy as np
import pandas as pd

from benchmark.validation.stability_lasso import stability_lasso_cv


class _SyntheticDataset:
    name = "synthetic"
    validation_config = {"n_folds": 3, "patient_col": "patient_id"}

    def __init__(self):
        ids = [f"r{i}" for i in range(30)]
        labels = np.array([0, 1] * 15)
        self._meta = pd.DataFrame({
            "region_id": ids,
            "patient_id": [f"p{i}" for i in range(30)],
            "label": labels,
        })
        self._target = self._meta.set_index("region_id")["label"]

    def get_task_config(self, task_id):
        return {"type": "binary", "label_col": "label"}

    def get_task_metadata(self, task_id):
        return self._meta.copy()

    def build_target(self, region_ids, task_id):
        return self._target.loc[region_ids].copy()


class StabilityLassoTest(unittest.TestCase):
    def test_precomputed_features_produce_all_reports(self):
        dataset = _SyntheticDataset()
        rng = np.random.default_rng(4)
        target = dataset._target.to_numpy()
        features = pd.DataFrame(
            {
                "signal": target + rng.normal(0, 0.15, len(target)),
                "noise": rng.normal(size=len(target)),
            },
            index=dataset._target.index,
        )

        result = stability_lasso_cv(
            dataset,
            "outcome",
            features=features,
            seeds=[0, 1],
            n_bootstrap=3,
            lambda_value=0.2,
        )

        self.assertEqual(len(result.fold_coefficients), 2 * 3 * 2)
        self.assertEqual(len(result.seed_summary), 2 * 2)
        self.assertEqual(len(result.feature_summary), 2)
        self.assertIn("seed_selection_frequency", result.feature_summary)
        self.assertIn("ci_low", result.feature_summary)
        self.assertFalse(result.bootstrap_coefficients.empty)

        summary = result.feature_summary.set_index("feature")
        self.assertGreater(summary.loc["signal", "coefficient_mean"], 0)


if __name__ == "__main__":
    unittest.main()
