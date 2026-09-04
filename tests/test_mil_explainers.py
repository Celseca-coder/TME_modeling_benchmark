from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from benchmark.data.dataset import RegionData
from benchmark.models.attention_mil import AttentionMILModel
from benchmark.models.mil_explainers import (
    aupc,
    explain_bag,
    faithfulness_metrics,
    one_removed_scores,
    perturbation_curve,
)
from benchmark.motifs.mil_evidence import evidence_for_bag, window_inside_mask
from benchmark.motifs.mil_run import localization_metrics, summarize_mil
from benchmark.motifs.spec import MotifCatalog, MotifSpec


def _region() -> RegionData:
    index = pd.Index(["a", "b", "c", "d"], name="cell_id")
    return RegionData(
        region_id="r1",
        coordinates=pd.DataFrame(
            {"x": [0.0, 1.0, 50.0, 51.0], "y": [0.0, 1.0, 50.0, 51.0]},
            index=index,
        ),
        expression=pd.DataFrame({"m": [1.0, 1.0, 1.0, 1.0]}, index=index),
        cell_types=pd.DataFrame(
            {"cell_type": ["Tumor", "Tumor", "CD8 T cell", "CD8 T cell"]},
            index=index,
        ),
        microns_per_pixel=1.0,
    )


def _catalog() -> MotifCatalog:
    return MotifCatalog(
        dataset="toy",
        cell_type_col="cell_type",
        cell_sets={
            "tumor": ("Tumor",),
            "cd8": ("CD8 T cell",),
        },
        motifs=(
            MotifSpec(id="tumor_high", kind="composition", spatial=False, cell_set="tumor"),
            MotifSpec(
                id="cd8_clustering",
                kind="neighbor_fraction",
                source_set="cd8",
                neighbor_set="cd8",
                radius_um=5.0,
                min_source_cells=1,
            ),
        ),
    )


def _trained_model() -> tuple[AttentionMILModel, np.ndarray]:
    torch.manual_seed(0)
    key = np.array([[8.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    bg = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.0]], dtype=np.float32)
    bags = pd.DataFrame({
        "bag": [key] * 6 + [bg] * 6,
        "instance_centers": [np.zeros((3, 2))] * 12,
    }, index=[f"p{i}" for i in range(6)] + [f"n{i}" for i in range(6)])
    y = pd.Series([1] * 6 + [0] * 6, index=bags.index)
    model = AttentionMILModel(
        seed=0, epochs=250, patience=80, dropout=0.0, device="cpu",
        max_instances=16, hidden_dim=16, lr=5e-3,
    ).fit(bags, y)
    return model, key


def test_window_inside_mask_splits_two_clusters():
    region = _region()
    xy = region.coordinates[["x", "y"]].to_numpy(float)
    tumor = window_inside_mask(xy, np.array([0.5, 0.5]), 10.0)
    cd8 = window_inside_mask(xy, np.array([50.5, 50.5]), 10.0)
    assert tumor.sum() == 2
    assert cd8.sum() == 2
    assert not np.array_equal(tumor, cd8)


def test_composition_evidence_ranks_tumor_window_higher():
    region = _region()
    catalog = _catalog()
    spec = catalog.motif("tumor_high")
    centers = np.array([[0.5, 0.5], [50.5, 50.5]], dtype=float)
    ev = evidence_for_bag(region, catalog, spec, centers, window_size_um=10.0)
    assert ev["score"][0] > ev["score"][1]
    assert ev["pos"][0] == 1.0
    assert ev["pos"][1] == 0.0


def test_localization_enrichment_detects_aligned_scores():
    pos = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([3, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    rec = localization_metrics(score, pos, 1 - pos)
    assert rec["loc_passed"]
    assert rec["loc_enrichment"] >= 1.25


def test_one_removed_marks_the_key_instance():
    model, bag = _trained_model()
    scores = one_removed_scores(model, bag)
    assert scores.argmax() == 0
    assert scores[0] > scores[1]


def test_morf_drops_faster_than_lerf_when_ranking_is_right():
    model, bag = _trained_model()
    scores = explain_bag(model, bag, "one_removed")
    morf = perturbation_curve(model, bag, scores, mode="morf", step_frac=0.34)
    lerf = perturbation_curve(model, bag, scores, mode="lerf", step_frac=0.34)
    assert aupc(morf) <= aupc(lerf) + 1e-6
    faith = faithfulness_metrics(model, bag, scores, np.random.default_rng(0).standard_normal(len(bag)))
    assert "aupc_morf" in faith


def test_summarize_mil_blocks_spatial_when_controls_fail():
    fold = pd.DataFrame([
        dict(task="motif_tumor_high", kind="control", explainer="attention",
             auc=0.5, auc_passed=False, loc_auprc=0.2, loc_auprc2=0.2,
             loc_enrichment=1.0, loc_passed=False, aupc_morf=0.4, aupc_lerf=0.4,
             delta_lerf_morf=0.0, faith_passed=False, fold_passed=False, selected=True),
        dict(task="motif_cd8_clustering", kind="spatial", explainer="attention",
             auc=0.7, auc_passed=True, loc_auprc=0.8, loc_auprc2=0.7,
             loc_enrichment=2.0, loc_passed=True, aupc_morf=0.2, aupc_lerf=0.6,
             delta_lerf_morf=0.4, faith_passed=True, fold_passed=True, selected=True),
    ])
    summary = summarize_mil(fold)
    spatial = summary[summary.task == "motif_cd8_clustering"].iloc[0]
    assert spatial["verdict"] == "fail_control_block"
