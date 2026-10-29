"""Wide-table layout for the Extreme20 window-importance driver."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.eval_extreme20_window_importance import _label_paths, write_wide


class Extreme20WindowImportanceTest(unittest.TestCase):
    def test_clean_and_noisy_filenames(self):
        clean_w, clean_b = _label_paths("hnc_wu2022", "clean")
        noisy_w, noisy_b = _label_paths("hnc_wu2022", "noisy")
        self.assertTrue(clean_w.name.endswith("_windows.csv"))
        self.assertTrue(clean_b.name.endswith("_v2.csv"))
        self.assertTrue(noisy_w.name.endswith("_windows_noisy.csv"))
        self.assertTrue(noisy_b.name.endswith("_v2_noisy.csv"))
        self.assertTrue(clean_w.exists())
        self.assertTrue(clean_b.exists())

    def test_wide_header_matches_method_order(self):
        frame = pd.DataFrame([
            dict(dataset="hnc_wu2022", motif="cd8_clustering", method="lasso",
                 n_windows=10, n_labeled=8, auroc_window_mean=0.81, precision_at_20=0.7),
            dict(dataset="hnc_wu2022", motif="cd8_clustering", method="additive_mil",
                 n_windows=10, n_labeled=8, auroc_window_mean=0.55, precision_at_20=0.9),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "wide.csv"
            write_wide(frame, dest)
            lines = dest.read_text().splitlines()
        self.assertIn("window L1 MIL", lines[0])
        self.assertIn("Additive MIL", lines[0])
        self.assertTrue(lines[1].startswith("task,n_windows,n_labeled,Lasso (AUROC),Lasso (precision_at_20)"))
        self.assertIn("Additive MIL (precision_at_20),Highest Method (AUROC)", lines[1])
        self.assertTrue(lines[2].endswith("Lasso,Additive MIL"))


if __name__ == "__main__":
    unittest.main()
