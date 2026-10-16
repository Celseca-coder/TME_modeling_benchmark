from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from benchmark.data.dataset import RegionData
from benchmark.features.local_window import (
    LocalWindowInstanceFeaturizer,
    WINDOW_PANEL_FAMILIES,
    columns_for_families,
    expand_panel_groups,
)
from benchmark.features.window_grid import Window


def _toy_region() -> RegionData:
    coords = pd.DataFrame(
        {"x": [0.0, 10.0, 20.0, 10.0, 30.0], "y": [0.0, 0.0, 5.0, 15.0, 8.0]},
        index=[f"c{i}" for i in range(5)],
    )
    expr = pd.DataFrame({"m1": [1, 1, 2, 2, 1]}, index=coords.index)
    types = pd.DataFrame(
        {"cell_type": ["T cell", "T cell", "Tumor", "Tumor", "T cell"]},
        index=coords.index,
    )
    return RegionData("r1", coords, expr, types, 1.0)


class LocalWindowPanelTest(unittest.TestCase):
    def test_panel_groups_include_point_pattern_and_density(self):
        groups = expand_panel_groups(WINDOW_PANEL_FAMILIES)
        self.assertEqual(
            groups,
            ("composition", "density", "celltype_density", "mixing", "point_pattern"),
        )

    def test_point_pattern_vector_matches_names(self):
        region = _toy_region()
        feat = LocalWindowInstanceFeaturizer(feature_groups=("point_pattern",)).fit([region])
        names = feat.feature_names()
        self.assertTrue(any(n.startswith("point_pattern__T cell_K_r") for n in names))
        inside = np.ones(len(region.coordinates), dtype=bool)
        vec = feat.extract_vector(region, inside, 100.0)
        self.assertEqual(len(vec), len(names))
        self.assertTrue(np.isfinite(vec).any())

    def test_extract_table_and_family_columns(self):
        region = _toy_region()
        feat = LocalWindowInstanceFeaturizer(
            feature_groups=expand_panel_groups(WINDOW_PANEL_FAMILIES),
        ).fit([region])
        win = Window(
            window_id=0,
            center_um=np.array([50.0, 50.0]),
            origin_um=np.array([0.0, 0.0]),
            inside=np.ones(len(region.coordinates), dtype=bool),
        )
        table = feat.extract_table(region, [win], 100.0)
        cols = columns_for_families(list(table.columns), ("density", "point_pattern"))
        self.assertIn("local_cell_density_per_mm2", cols)
        self.assertTrue(any(c.startswith("celltype_density__") for c in cols))
        self.assertTrue(any(c.startswith("point_pattern__") for c in cols))
        self.assertFalse(any(c.startswith("composition__") for c in cols))
        self.assertFalse(any(c.startswith("mixing__") for c in cols))

    def test_one_cell_window_keeps_mixing_keys(self):
        region = _toy_region()
        feat = LocalWindowInstanceFeaturizer(feature_groups=("mixing",)).fit([region])
        inside = np.zeros(len(region.coordinates), dtype=bool)
        inside[0] = True
        vec = feat.extract_vector(region, inside, 100.0)
        names = feat.feature_names()
        self.assertIn("mixing__same_type_neighbor_fraction", names)
        self.assertEqual(len(vec), len(names))
        self.assertTrue(np.isnan(vec[names.index("mixing__same_type_neighbor_fraction")]))
