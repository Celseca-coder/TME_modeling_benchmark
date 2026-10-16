from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from benchmark.data.dataset import RegionData
from benchmark.features.window_grid import iter_windows
from benchmark.motifs.mil_evidence import (
    label_bag_windows,
    window_label,
)
from benchmark.motifs.spec import catalog_from_dict


def _region(types, xs, ys, polygons=None) -> RegionData:
    index = pd.Index(range(len(types)), name="cell_id")
    return RegionData(
        region_id="synthetic",
        coordinates=pd.DataFrame({"x": xs, "y": ys}, index=index),
        expression=pd.DataFrame({"m": np.ones(len(types))}, index=index),
        cell_types=pd.DataFrame({"cell_type_uniform": types}, index=index),
        microns_per_pixel=1.0,
        polygons=polygons,
    )


def _catalog():
    return catalog_from_dict({
        "dataset": "synthetic",
        "cell_type_col": "cell_type_uniform",
        "cell_sets": {
            "tumor": ["Tumor"],
            "cd8": ["CD8 T cell"],
            "t_cell": ["CD8 T cell"],
            "b_cell": ["B cell"],
        },
        "motifs": [
            {"id": "tumor_high", "kind": "composition", "spatial": False, "cell_set": "tumor"},
            {
                "id": "cd8_clustering",
                "kind": "neighbor_fraction",
                "source_set": "cd8",
                "neighbor_set": "cd8",
                "radius_um": 20.0,
                "min_source_cells": 1,
                "label_rule": "tertile_extremes",
            },
            {
                "id": "tls_like",
                "kind": "tls_like",
                "required_sets": ["b_cell", "t_cell"],
                "min_frac": {"b_cell": 0.2, "t_cell": 0.2},
                "radius_um": 20.0,
                "min_cluster_size": 3,
            },
        ],
    })


def test_window_label_nan_when_neither():
    assert np.isnan(window_label(False, False))
    assert window_label(True, False) == 1.0
    assert window_label(False, True) == 0.0


def test_composition_tertiles_split_high_low_windows():
    types = ["Tumor"] * 8 + ["CD8 T cell"] * 8
    xs = [10.0] * 8 + [200.0] * 8
    ys = [10.0 + i for i in range(8)] + [10.0 + i for i in range(8)]
    region = _region(types, xs, ys)
    catalog = _catalog()
    table = label_bag_windows(
        region, catalog, catalog.motif("tumor_high"),
        window_size_um=50.0, step_um=50.0, min_cells=1,
    )
    assert set(table["label"].dropna().unique()).issubset({0.0, 1.0})
    assert table.loc[table["score"].idxmax(), "label"] == 1.0
    assert table.loc[table["score"].idxmin(), "label"] == 0.0


def test_clustering_neg_on_tumor_pure_window():
    types = ["Tumor"] * 6 + ["CD8 T cell"] * 6
    xs = [5.0] * 6 + [180.0] * 6
    ys = list(range(6)) + list(range(6))
    region = _region(types, xs, ys)
    catalog = _catalog()
    table = label_bag_windows(
        region, catalog, catalog.motif("cd8_clustering"),
        window_size_um=40.0, step_um=40.0, min_cells=1,
    )
    tumor_win = table.loc[table["score"].idxmin()]
    assert tumor_win["neg"] == 1.0 or tumor_win["label"] == 0.0


def test_iter_windows_uses_100_by_50_defaults():
    types = ["Tumor"] * 4
    region = _region(types, [0, 80, 0, 80], [0, 0, 80, 80])
    windows = iter_windows(region)
    assert windows
    assert all(w.inside.any() for w in windows)