from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import box

from benchmark.data.dataset import RegionData
from benchmark.motifs.detect import score_motif, score_region, shuffle_coordinates
from benchmark.motifs.labels import add_bootstrap_labels, add_pseudo_labels, assign_labels
from benchmark.motifs.overlay import attach_pseudo_labels, motif_task_ids
from benchmark.motifs.spec import MotifCatalog, catalog_from_dict, load_motif_catalog


def _grid_region(types: list[str], xs: list[float], ys: list[float], polygons=None) -> RegionData:
    index = pd.Index(range(len(types)), name="cell_id")
    return RegionData(
        region_id="synthetic",
        coordinates=pd.DataFrame({"x": xs, "y": ys}, index=index),
        expression=pd.DataFrame({"m": np.ones(len(types))}, index=index),
        cell_types=pd.DataFrame({"cell_type_uniform": types}, index=index),
        microns_per_pixel=1.0,
        polygons=polygons,
    )


def _hnc_catalog(**overrides) -> MotifCatalog:
    raw = {
        "dataset": "synthetic",
        "cell_type_col": "cell_type_uniform",
        "cell_sets": {
            "tumor": ["Tumor", "Tumor (Proliferating)"],
            "t_cell": ["CD4 T cell", "CD8 T cell"],
            "cd8": ["CD8 T cell"],
            "b_cell": ["B cell"],
            "immune": ["CD4 T cell", "CD8 T cell", "B cell"],
        },
        "motifs": [
            {"id": "tumor_high", "kind": "composition", "spatial": False, "cell_set": "tumor"},
            {
                "id": "t_tumor_mixing",
                "kind": "neighbor_fraction",
                "source_set": "t_cell",
                "neighbor_set": "tumor",
                "radius_um": 1.6,
                "min_source_cells": 2,
                "label_rule": "tertile_extremes",
            },
            {
                "id": "immune_exclusion",
                "kind": "immune_exclusion",
                "immune_set": "cd8",
                "tumor_set": "tumor",
                "radius_um": 1.6,
                "min_source_cells": 2,
                "residual_on": ["tumor", "cd8"],
                "label_rule": "tertile_extremes",
            },
            {
                "id": "tls_like",
                "kind": "tls_like",
                "required_sets": ["b_cell", "t_cell"],
                "min_frac": {"b_cell": 0.2, "t_cell": 0.2},
                "radius_um": 1.6,
                "min_cluster_size": 3,
            },
            {
                "id": "interface_immune",
                "kind": "interface",
                "immune_set": "immune",
                "tumor_set": "tumor",
                "radius_um": 1.2,
                "min_source_cells": 2,
            },
        ],
    }
    raw.update(overrides)
    return catalog_from_dict(raw)


def test_hnc_yaml_loads_expanded_catalog():
    catalog = load_motif_catalog(dataset="hnc_wu2022")
    ids = [m.id for m in catalog.motifs]
    assert catalog.dataset == "hnc_wu2022"
    assert ids[:2] == ["tumor_high", "cd8_high"]
    assert "immune_exclusion" in ids
    assert "tls_like" in ids
    assert "cd8_clustering" in ids
    assert "macrophage_tumor_niche" in ids
    assert len(ids) >= 12
    spatial = [m for m in catalog.motifs if m.spatial]
    assert spatial
    assert all(m.residual_on for m in spatial)
    assert catalog.resolve_set("tumor") == {"Tumor", "Tumor (Proliferating)"}
    assert "Vessel" in catalog.resolve_set("vessel")


def test_composition_score_is_tumor_fraction():
    catalog = _hnc_catalog()
    region = _grid_region(
        ["Tumor"] * 8 + ["CD8 T cell"] * 2,
        list(range(10)),
        [0.0] * 10,
    )
    spec = catalog.motif("tumor_high")
    assert score_motif(region, catalog, spec) == 0.8
    assert score_region(region, catalog)["frac__cd8"] == 0.2


def test_mixing_high_when_interleaved_low_when_segregated():
    catalog = _hnc_catalog()
    spec = catalog.motif("t_tumor_mixing")
    mixed = _grid_region(
        ["Tumor", "CD8 T cell"] * 6,
        list(range(12)),
        [0.0] * 12,
    )
    segregated = _grid_region(
        ["Tumor"] * 6 + ["CD8 T cell"] * 6,
        list(range(12)),
        [0.0] * 12,
    )
    mixed_score = score_motif(mixed, catalog, spec)
    segregated_score = score_motif(segregated, catalog, spec)
    assert mixed_score > 0.4
    assert segregated_score < mixed_score


def test_exclusion_uses_tumor_polygon():
    catalog = _hnc_catalog()
    spec = catalog.motif("immune_exclusion")
    types = ["Tumor"] * 6 + ["CD8 T cell"] * 6
    xs = [0.2] * 6 + [3.2] * 6
    ys = list(range(6)) + list(range(6))
    polygons = {"tumour": box(0.0, -0.5, 1.0, 6.5)}
    excluded = _grid_region(types, xs, ys, polygons=polygons)
    infiltrated = _grid_region(
        types,
        [0.2] * 6 + [0.4] * 6,
        list(range(6)) + list(range(6)),
        polygons=polygons,
    )
    assert score_motif(excluded, catalog, spec) == 1.0
    assert score_motif(infiltrated, catalog, spec) == 0.0


def test_tls_cluster_beats_scattered_b_and_t():
    catalog = _hnc_catalog()
    spec = catalog.motif("tls_like")
    clustered = _grid_region(
        ["B cell", "CD4 T cell", "CD8 T cell", "B cell", "CD4 T cell"] + ["Tumor"] * 8,
        [0.0, 0.4, 0.8, 0.2, 0.6] + list(np.linspace(8, 16, 8)),
        [0.0, 0.0, 0.0, 0.4, 0.4] + [10.0] * 8,
    )
    scattered = _grid_region(
        ["B cell", "Tumor", "CD4 T cell", "Tumor", "CD8 T cell"] + ["Tumor"] * 8,
        list(range(13)),
        [0.0] * 13,
    )
    assert score_motif(clustered, catalog, spec) > score_motif(scattered, catalog, spec)


def test_shuffle_changes_spatial_score_not_composition():
    catalog = _hnc_catalog()
    region = _grid_region(
        ["Tumor"] * 6 + ["CD8 T cell"] * 6,
        list(range(12)),
        [0.0] * 12,
    )
    rng = np.random.default_rng(0)
    shuffled = shuffle_coordinates(region, rng)
    assert score_motif(region, catalog, catalog.motif("tumor_high")) == score_motif(
        shuffled, catalog, catalog.motif("tumor_high")
    )
    # Segregated line becomes mixed after a coordinate permutation.
    assert score_motif(shuffled, catalog, catalog.motif("t_tumor_mixing")) > score_motif(
        region, catalog, catalog.motif("t_tumor_mixing")
    )


def test_tertile_extremes_drops_middle():
    values = pd.Series(np.arange(9, dtype=float))
    labels = assign_labels(values, "tertile_extremes")
    assert int(labels.isna().sum()) == 3
    assert set(labels.dropna().unique()) == {0.0, 1.0}


def test_add_pseudo_labels_and_overlay_register_tasks():
    catalog = _hnc_catalog()
    rows = []
    for i in range(12):
        rows.append({
            "region_id": f"r{i}",
            "tumor_high_score": i / 11,
            "t_tumor_mixing_score": (11 - i) / 11,
            "immune_exclusion_score": abs(i - 5) / 11,
            "tls_like_score": (i % 4) / 3,
            "interface_immune_score": i / 11,
            "frac__tumor": i / 11,
            "frac__cd8": (11 - i) / 11,
            "frac__t_cell": 0.2,
            "frac__b_cell": 0.1,
            "frac__immune": 0.3,
        })
    table = pd.DataFrame(rows)
    labeled = add_pseudo_labels(table, catalog)
    assert labeled["tumor_high_label"].notna().all()
    assert labeled["t_tumor_mixing_label"].isna().any()

    class _Dummy:
        def __init__(self):
            self.config = {"tasks": [{"id": "clinical", "type": "binary_classification"}]}
            self._metadata = pd.DataFrame({
                "region_id": [f"r{i}" for i in range(12)],
                "patient_id": [f"p{i}" for i in range(12)],
            })

        def get_metadata(self):
            return self._metadata

    dummy = _Dummy()
    attach_pseudo_labels(dummy, labeled, catalog)
    assert "motif_tumor_high" in motif_task_ids(dummy)
    assert "tumor_high_label" in dummy.get_metadata().columns
    assert dummy.config["tasks"][0]["id"] == "clinical"


def test_bootstrap_labels_keep_stable_extremes():
    catalog = _hnc_catalog()
    rows = []
    for i in range(30):
        rows.append({
            "region_id": f"r{i}",
            "patient_id": f"p{i // 3}",
            "t_tumor_mixing_score": float(i),
            "t_tumor_mixing_score_used": float(i),
            "t_tumor_mixing_label": 0.0 if i < 10 else (1.0 if i >= 20 else float("nan")),
            "tumor_high_score": 0.5,
            "tumor_high_score_used": 0.5,
            "tumor_high_label": 1.0,
            "immune_exclusion_score": 0.0,
            "tls_like_score": 0.0,
            "interface_immune_score": 0.0,
            "frac__tumor": 0.4,
            "frac__cd8": 0.1,
            "frac__t_cell": 0.2,
            "frac__b_cell": 0.1,
            "frac__immune": 0.3,
        })
    table = pd.DataFrame(rows)
    labeled = add_pseudo_labels(table, catalog)
    refined = add_bootstrap_labels(
        labeled, catalog, n_boot=50, confidence=0.80, seed=0,
        motifs=["t_tumor_mixing"],
    )
    v2 = refined["t_tumor_mixing_label_v2"]
    assert (v2.iloc[:5] == 0.0).all()
    assert (v2.iloc[-5:] == 1.0).all()
    assert v2.isna().any()
    assert int(v2.notna().sum()) <= int(labeled["t_tumor_mixing_label"].notna().sum())
    assert "t_tumor_mixing_p_low" in refined.columns


def test_overlay_rejects_stale_label_table():
    catalog = _hnc_catalog()
    stale = pd.DataFrame({"region_id": ["r0"], "tumor_high_label": [1.0]})

    class _Dummy:
        def __init__(self):
            self.config = {"tasks": []}
            self._metadata = pd.DataFrame({"region_id": ["r0"]})

        def get_metadata(self):
            return self._metadata

    try:
        attach_pseudo_labels(_Dummy(), stale, catalog)
    except ValueError as exc:
        assert "cd8_tumor_contact_label" not in str(exc)
        assert "t_tumor_mixing_label" in str(exc)
        assert "generate_pseudo_labels.py" in str(exc)
    else:
        raise AssertionError("expected stale labels to raise")
