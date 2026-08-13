from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from benchmark.features.attention_mil import HandcraftedAttentionMILFeaturizer
from benchmark.features.combined import CombinedCompositionExpressionFeaturizer
from benchmark.features.patch_feats import PatchBasedFeaturizer


def _region() -> RegionData:
    index = pd.Index([1, 2, 3], name="cell_id")
    return RegionData(
        region_id="region",
        coordinates=pd.DataFrame(
            {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0]}, index=index
        ),
        expression=pd.DataFrame(
            {"marker_a": [1.0, 3.0, 5.0], "marker_b": [2.0, 4.0, 6.0]},
            index=index,
        ),
        cell_types=pd.DataFrame({"cell_type": ["A", "B", "A"]}, index=index),
        microns_per_pixel=1.0,
    )


def test_naive_mil_single_window_matches_global_simple_concatenation():
    region = _region()
    global_features = CombinedCompositionExpressionFeaturizer().fit([region])
    expected = global_features.extract_region(region)

    naive = PatchBasedFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("composition", "expression"),
        aggregations=("mean",),
        min_cells_per_window=1,
    ).fit([region])
    actual = naive.extract_region(region)

    assert list(actual) == [f"patch_mean_{name}" for name in expected]
    np.testing.assert_allclose(list(actual.values()), list(expected.values()))


def test_attention_mil_single_window_matches_global_simple_concatenation():
    region = _region()
    global_features = CombinedCompositionExpressionFeaturizer().fit([region])
    expected = global_features.extract_region(region)

    attention = HandcraftedAttentionMILFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("composition", "expression"),
        min_cells_per_window=1,
    ).fit([region])
    actual = attention.extract_region(region)["bag"]

    assert attention.feature_names() == list(expected)
    assert actual.shape == (1, len(expected))
    np.testing.assert_allclose(actual[0], list(expected.values()))


def test_mil_windows_allow_external_regions_without_cell_type_rows():
    train = _region()
    external = _region()
    external.cell_types = external.cell_types.iloc[0:0]

    naive = PatchBasedFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("composition", "expression"),
        aggregations=("mean",),
        min_cells_per_window=1,
    ).fit([train])
    naive_values = naive.extract_region(external)
    assert naive_values["patch_mean_composition__A"] == 0.0
    assert naive_values["patch_mean_composition__B"] == 0.0
    assert naive_values["patch_mean_expression__marker_a"] == 3.0

    attention = HandcraftedAttentionMILFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("composition", "expression"),
        min_cells_per_window=1,
    ).fit([train])
    bag = attention.extract_region(external)["bag"]
    np.testing.assert_allclose(bag[0], [0.0, 0.0, 3.0, 4.0])
