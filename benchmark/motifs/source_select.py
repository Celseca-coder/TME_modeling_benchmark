"""Pick one source dataset per motif from a frozen candidate list.

Jackson / METABRIC have no CD8 vs CD4 split; ``cd8_*`` there is a T-cell proxy
and is penalised relative to HNC / TNBC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

CANDIDATE_DATASETS = (
    "hnc_wu2022",
    "bc_jackson2020",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)

TRUE_CD8_DATASETS = {"hnc_wu2022", "tnbc_wang2023"}
CD8_MOTIFS = {
    "cd8_high",
    "cd8_clustering",
    "cd8_tumor_contact",
    "immune_exclusion",
}

TASK_TYPE = {
    "tumor_high": "composition_control",
    "cd8_high": "composition_control",
    "cd8_clustering": "spatial_arrangement",
    "tumor_stroma_mixing": "spatial_mixing",
    "t_tumor_mixing": "spatial_mixing",
    "cd8_tumor_contact": "spatial_mixing",
    "macrophage_tumor_niche": "spatial_mixing",
    "granulocyte_tumor_niche": "spatial_mixing",
    "apc_t_contact": "spatial_mixing",
    "interface_immune": "spatial_compartment",
    "immune_exclusion": "spatial_compartment",
    "vessel_immune": "spatial_compartment",
    "tls_like": "spatial_aggregate",
}

DEFINITIONS = {
    "tumor_high": "Tumor-cell fraction at or above the discovery-cohort median.",
    "cd8_high": "CD8 (or T-cell proxy) fraction at or above the discovery-cohort median.",
    "cd8_clustering": "After residualizing on CD8%, CD8 cells have unusually many CD8 neighbors within 50 µm.",
    "tumor_stroma_mixing": "After residualizing on tumor% and stroma%, tumor cells have unusually many stromal neighbors within 50 µm.",
    "interface_immune": "After residualizing on immune% and tumor%, immune cells are concentrated within 50 µm of the tumor boundary.",
    "immune_exclusion": "After residualizing on tumor% and CD8%, CD8 cells lie outside the tumor polygon more than abundance predicts.",
    "t_tumor_mixing": "After residualizing on tumor% and T-cell%, T cells have unusually many tumor neighbors within 50 µm.",
    "cd8_tumor_contact": "After residualizing on tumor% and CD8%, CD8 cells have unusually many tumor neighbors within 50 µm.",
    "macrophage_tumor_niche": "After residualizing on tumor% and macrophage%, macrophages have unusually many tumor neighbors within 50 µm.",
    "granulocyte_tumor_niche": "After residualizing on tumor% and granulocyte%, granulocytes have unusually many tumor neighbors within 50 µm.",
    "apc_t_contact": "After residualizing on APC% and T-cell%, APCs have unusually many T-cell neighbors within 50 µm.",
    "tls_like": "After residualizing on B% and T%, cells sit in 50 µm neighborhoods jointly enriched for B and T that form clusters of size ≥5.",
    "vessel_immune": "After residualizing on immune% and vessel%, immune cells have unusually many vessel neighbors within 50 µm.",
}


@dataclass(frozen=True)
class MotifSource:
    motif_id: str
    source_dataset: str | None
    eligible: bool
    score: float | None
    reason: str
    n_labeled: int
    n_0: int
    n_1: int
    auc_composition: float | None
    auc_expression: float | None
    auc_density: float | None
    expression_keep: bool
    density_keep: bool
    cd8_is_proxy: bool
    task_type: str
    definition: str


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if pd.notna(x) else None


def _eligible_control(expr_row: pd.Series) -> bool:
    return (
        str(expr_row.get("gate1_composition")) == "pass_control"
        and str(expr_row.get("gate2_expression")) == "pass_control"
    )


def _eligible_spatial(expr_row: pd.Series, dens_row: pd.Series | None) -> tuple[bool, bool]:
    expr_keep = bool(expr_row.get("in_formal_set")) and str(expr_row.get("decision")) in {
        "keep_spatial",
        "keep_spatial_caveat",
    }
    dens_keep = False
    if dens_row is not None:
        dens_keep = bool(dens_row.get("in_formal_set")) and str(dens_row.get("decision")) in {
            "keep_spatial",
            "keep_spatial_caveat",
        }
    return expr_keep, dens_keep


def suitability_score(
    motif_id: str,
    dataset: str,
    expr_row: pd.Series,
    dens_row: pd.Series | None,
) -> float | None:
    spatial = bool(expr_row.get("spatial"))
    if not spatial:
        if not _eligible_control(expr_row):
            return None
        expr_keep, dens_keep = True, True
    else:
        expr_keep, dens_keep = _eligible_spatial(expr_row, dens_row)
        if not (expr_keep or dens_keep):
            return None

    auc_c = _num(expr_row.get("auc_composition")) or 0.5
    auc_e = _num(expr_row.get("auc_expression")) or 0.5
    auc_d = _num(dens_row.get("auc_density")) if dens_row is not None else None
    n = int(expr_row.get("n_labeled_v2") or 0)
    n0 = int(expr_row.get("n_0") or 0)
    n1 = int(expr_row.get("n_1") or 0)

    score = 0.0
    if motif_id in CD8_MOTIFS:
        score += 100.0 if dataset in TRUE_CD8_DATASETS else -50.0

    if not spatial:
        score += 25.0 * auc_c + 25.0 * auc_e
        if auc_d is not None:
            score += 5.0 * auc_d
    else:
        score += 35.0 * (1.0 - min(abs(auc_c - 0.5) / 0.5, 1.0))
        if expr_keep:
            score += 25.0 * (1.0 - min(abs(auc_e - 0.5) / 0.5, 1.0))
        if dens_keep and auc_d is not None:
            score += 18.0 * auc_d
            if str(dens_row.get("decision")) == "keep_spatial_caveat":
                score -= 8.0

    score += 8.0 * math.log10(max(n, 1))
    if n0 + n1:
        score += 6.0 * (1.0 - abs(n0 - n1) / (n0 + n1))
    return score


def assign_motif_sources(
    expr: pd.DataFrame,
    dens: pd.DataFrame,
    datasets: tuple[str, ...] = CANDIDATE_DATASETS,
) -> pd.DataFrame:
    expr = expr[expr["dataset"].isin(datasets)].copy()
    dens = dens[dens["dataset"].isin(datasets)].copy()
    dens_ix = dens.set_index(["dataset", "motif_id"])
    motifs = sorted(expr["motif_id"].unique())
    rows = []
    for motif_id in motifs:
        candidates = []
        for dataset in datasets:
            e_sub = expr[(expr["dataset"] == dataset) & (expr["motif_id"] == motif_id)]
            if e_sub.empty:
                continue
            e_row = e_sub.iloc[0]
            d_row = dens_ix.loc[(dataset, motif_id)] if (dataset, motif_id) in dens_ix.index else None
            if d_row is not None:
                d_row = d_row if isinstance(d_row, pd.Series) else d_row.iloc[0]
            score = suitability_score(motif_id, dataset, e_row, d_row)
            expr_keep, dens_keep = (
                (True, True)
                if not bool(e_row.get("spatial"))
                else _eligible_spatial(e_row, d_row)
            )
            if not bool(e_row.get("spatial")):
                expr_keep = _eligible_control(e_row)
                dens_keep = expr_keep
            candidates.append(dict(
                dataset=dataset,
                score=score,
                e_row=e_row,
                d_row=d_row,
                expr_keep=expr_keep,
                dens_keep=dens_keep,
            ))
        ranked = [c for c in candidates if c["score"] is not None]
        ranked.sort(key=lambda c: c["score"], reverse=True)
        winner = ranked[0] if ranked else None
        e_row = winner["e_row"] if winner else None
        d_row = winner["d_row"] if winner else None
        source = winner["dataset"] if winner else None
        reasons = []
        if winner is None:
            reasons.append("no dataset among the four passed the HNC composition→expression/density gates")
        else:
            if not bool(e_row.get("spatial")):
                reasons.append("control: composition and expression both recover the label")
            else:
                if winner["expr_keep"]:
                    reasons.append("expression panel: composition near chance and expression near chance")
                if winner["dens_keep"]:
                    reasons.append("density panel: composition near chance and density recovers the label")
            if motif_id in CD8_MOTIFS:
                reasons.append(
                    "true CD8 types"
                    if source in TRUE_CD8_DATASETS
                    else "T-cell proxy (no CD8/CD4 split)"
                )
            reasons.append(f"n={int(e_row['n_labeled_v2'])} labeled ({int(e_row['n_0'])}/{int(e_row['n_1'])})")
        rows.append(dict(
            motif_id=motif_id,
            source_dataset=source,
            eligible=winner is not None,
            score=None if winner is None else round(float(winner["score"]), 3),
            reason="; ".join(reasons),
            n_labeled=0 if e_row is None else int(e_row.get("n_labeled_v2") or 0),
            n_0=0 if e_row is None else int(e_row.get("n_0") or 0),
            n_1=0 if e_row is None else int(e_row.get("n_1") or 0),
            auc_composition=None if e_row is None else _num(e_row.get("auc_composition")),
            auc_expression=None if e_row is None else _num(e_row.get("auc_expression")),
            auc_density=None if d_row is None else _num(d_row.get("auc_density")),
            expression_keep=False if winner is None else bool(winner["expr_keep"]),
            density_keep=False if winner is None else bool(winner["dens_keep"]),
            cd8_is_proxy=bool(source and motif_id in CD8_MOTIFS and source not in TRUE_CD8_DATASETS),
            task_type=TASK_TYPE.get(motif_id, "spatial"),
            definition=DEFINITIONS.get(motif_id, ""),
            panel=(
                "control" if motif_id in ("tumor_high", "cd8_high")
                else ("expression" if winner and winner["expr_keep"] and not winner["dens_keep"]
                      else "density" if winner and winner["dens_keep"] and not winner["expr_keep"]
                      else "expression+density" if winner else "none")
            ),
        ))
    return pd.DataFrame(rows)
