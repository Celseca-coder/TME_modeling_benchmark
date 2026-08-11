import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_cached_model_features import (
    _region_lookup,
    _select_npz,
    _validate_table,
    aggregate_cytocommunity,
    aggregate_embeddings,
    aggregate_utag_message_passing,
)


class CachedModelFeatureExportTest(unittest.TestCase):
    def test_embedding_npz_files_become_region_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "r1.npz", feature=np.array([1.0, 2.0]))
            np.savez_compressed(root / "r2.npz", feature=np.array([3.0, 4.0]))

            paths = _select_npz(root, _region_lookup(["r1", "r2"]))
            table = _validate_table(aggregate_embeddings(paths, "kronos"))

            self.assertEqual(list(table.index), ["r1", "r2"])
            self.assertEqual(list(table.columns), ["kronos_0000", "kronos_0001"])
            self.assertEqual(float(table.loc["r2", "kronos_0001"]), 4.0)

    def test_utag_message_passing_uses_marker_mean_and_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(
                root / "r1.npz",
                X=np.array([[1.0, 2.0], [3.0, 6.0]]),
                markers=np.array(["CD3", "CD8"]),
            )

            paths = _select_npz(root, _region_lookup(["r1"]))
            table = _validate_table(aggregate_utag_message_passing(paths))

            self.assertEqual(
                list(table.columns),
                [
                    "utag_mean__CD3",
                    "utag_mean__CD8",
                    "utag_std__CD3",
                    "utag_std__CD8",
                ],
            )
            self.assertEqual(float(table.loc["r1", "utag_mean__CD8"]), 4.0)
            self.assertEqual(float(table.loc["r1", "utag_std__CD3"]), 1.0)

    def test_cytocommunity_labels_become_compositions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "Step4_Output_demo__r1"
            output.mkdir()
            pd.DataFrame({"TCN_Label": [1, 1, 2, 2]}).to_csv(
                output / "ResultTable_demo__r1.csv", index=False
            )

            table = _validate_table(
                aggregate_cytocommunity(root, _region_lookup(["r1"]))
            )

            self.assertEqual(
                list(table.columns), ["tcn_fraction__1", "tcn_fraction__2"]
            )
            self.assertEqual(float(table.loc["r1"].sum()), 1.0)

    def test_duplicate_cache_variants_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            np.savez_compressed(root / "a" / "r1.npz", feature=np.array([1.0]))
            np.savez_compressed(root / "b" / "r1.npz", feature=np.array([2.0]))

            with self.assertRaisesRegex(ValueError, "Multiple cache variants"):
                _select_npz(root, _region_lookup(["r1"]))


if __name__ == "__main__":
    unittest.main()
