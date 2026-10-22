"""Window mean / attention pooling keeps one column per global feature."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from benchmark.features.window_pool import (
    attention_weighted_mean,
    extract_bags,
    feature_names,
    make_featurizers,
    mean_pool,
    mean_pool_table,
)


def _region() -> RegionData:
    index = [1, 2, 3, 4]
    coordinates = pd.DataFrame(
        {"x": [10.0, 20.0, 160.0, 170.0], "y": [10.0, 20.0, 10.0, 20.0]},
        index=index,
    )
    expression = pd.DataFrame({"CD8": [0.0, 0.0, 1.0, 1.0]}, index=index)
    cell_types = pd.DataFrame(
        {"cell_type": ["Tumor", "Tumor", "CD8 T cell", "CD8 T cell"]},
        index=index,
    )
    return RegionData(
        region_id="r1",
        coordinates=coordinates,
        expression=expression,
        cell_types=cell_types,
        microns_per_pixel=1.0,
        polygons=None,
    )


class WindowPoolTest(unittest.TestCase):
    def test_mean_ignores_nan_per_column(self):
        bag = np.array([[1.0, np.nan], [3.0, 4.0]])
        np.testing.assert_allclose(mean_pool(bag), [2.0, 4.0])

    def test_attention_renormalizes_nan_columns(self):
        bag = np.array([[1.0, np.nan], [3.0, 4.0]])
        pooled = attention_weighted_mean(bag, np.array([0.25, 0.75]))
        np.testing.assert_allclose(pooled, [2.5, 4.0])

    def test_square_windows_average_composition(self):
        region = _region()
        featurizers = make_featurizers("cell_type")
        for feat in featurizers.values():
            feat.fit([region])
        bags = extract_bags([region], featurizers, window_size_um=100.0, step_um=100.0, min_cells=2)
        self.assertEqual(len(bags["r1"]), 2)
        names = feature_names(featurizers)
        self.assertIn("composition::Tumor", names)
        self.assertIn("point-pattern::CD8 T cell_L_r10", names)
        self.assertNotIn("point-pattern::CD8 T cell_L_r100", names)
        table = mean_pool_table(bags, names)
        self.assertAlmostEqual(table.loc["r1", "composition::Tumor"], 0.5)
        self.assertAlmostEqual(table.loc["r1", "composition::CD8 T cell"], 0.5)


if __name__ == "__main__":
    unittest.main()
