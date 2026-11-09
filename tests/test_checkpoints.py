"""Fold checkpoints are written only when a directory is given."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from benchmark.motifs.checkpoints import save_fold
from scripts.eval_extreme20_window_importance import _checkpoints_ready, checkpoint_dir


class CheckpointTest(unittest.TestCase):
    def test_save_and_reload_state(self):
        weight = torch.tensor([1.0, 2.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_fold(tmp, 2, {"kind": "l1_mil", "state_dict": {"score.weight": weight}})
            self.assertEqual(path.name, "fold2.pt")
            loaded = torch.load(path, weights_only=False)
        self.assertEqual(loaded["kind"], "l1_mil")
        torch.testing.assert_close(loaded["state_dict"]["score.weight"], weight)

    def test_none_directory_skips_the_file(self):
        self.assertIsNone(save_fold(None, 0, {"kind": "attention"}))

    def test_finished_row_without_files_is_not_ready(self):
        rows = [{"dataset": "hnc", "motif": "cd8", "method": "lasso", "n_folds": 5}]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "lasso.csv"
            self.assertFalse(_checkpoints_ready(rows, dest, "hnc", "cd8", "lasso"))
            folder = checkpoint_dir(dest, "hnc", "cd8", "lasso")
            folder.mkdir(parents=True)
            for fold in range(5):
                torch.save({"fold": fold}, folder / f"fold{fold}.pt")
            self.assertTrue(_checkpoints_ready(rows, dest, "hnc", "cd8", "lasso"))


if __name__ == "__main__":
    unittest.main()
