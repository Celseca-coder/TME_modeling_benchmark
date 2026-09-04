from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from benchmark.features.attention_mil import HandcraftedAttentionMILFeaturizer
from benchmark.features.combined import CombinedCompositionExpressionFeaturizer
from benchmark.features.mixing import MixingFeaturizer
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


def test_window_celltype_density_and_mixing_groups():
    region = _region()
    naive = PatchBasedFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("composition", "celltype_density", "mixing"),
        aggregations=("mean",),
        min_cells_per_window=1,
    ).fit([region])
    values = naive.extract_region(region)
    area_mm2 = 100.0 * 100.0 / 1_000_000.0
    assert values["patch_mean_celltype_density__A"] == 2.0 / area_mm2
    assert values["patch_mean_celltype_density__B"] == 1.0 / area_mm2
    assert "patch_mean_mixing__local_mixing_mean" in values

    attention = HandcraftedAttentionMILFeaturizer(
        window_size_um=100.0,
        step_um=100.0,
        feature_groups=("mixing", "celltype_density"),
        min_cells_per_window=1,
    ).fit([region])
    names = attention.feature_names()
    assert any(name.startswith("mixing__") for name in names)
    assert "celltype_density__A" in names
    bag = attention.extract_region(region)["bag"]
    assert bag.shape[0] == 1
    assert bag.shape[1] == len(names)


def test_local_mixing_gini_matches_per_cell_value_counts():
    rng = np.random.default_rng(0)
    n, k = 40, 6
    xy = rng.normal(size=(n, 2))
    labels = pd.Series(rng.choice(["A", "B", "C"], size=n))
    feat = MixingFeaturizer(k_neighbors=k)
    feat.cell_types_ = ["A", "B", "C"]
    got = feat._compute_local_mixing(xy, labels, k)

    from scipy.spatial import KDTree
    _, indices = KDTree(xy).query(xy, k=k + 1)
    indices = np.asarray(indices)[:, 1:]
    labels_arr = labels.to_numpy()
    scores = []
    for i in range(n):
        counts = pd.Series(labels_arr[indices[i]]).value_counts()
        scores.append(float(1 - ((counts / k) ** 2).sum()))
    np.testing.assert_allclose(got["local_mixing_mean"], np.mean(scores), rtol=1e-12)
    np.testing.assert_allclose(got["local_mixing_std"], np.std(scores), rtol=1e-12)
    same = labels_arr[indices] == labels_arr[:, None]
    np.testing.assert_allclose(got["same_type_neighbor_fraction"], np.mean(same), rtol=1e-12)
    np.testing.assert_allclose(
        got["neighbor_fraction__A__to__B"],
        np.mean(labels_arr[indices[labels_arr == "A"]] == "B"),
        rtol=1e-12,
    )
