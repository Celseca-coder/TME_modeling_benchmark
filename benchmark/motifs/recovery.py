"""Expected-feature recovery rules for motif pseudo-label explanations."""
from __future__ import annotations

import re
from dataclasses import dataclass


SELECTED_EXPLAIN_TASKS = (
    "motif_tumor_high",
    "motif_cd8_high",
    "motif_cd8_clustering",
    "motif_tumor_stroma_mixing",
    "motif_interface_immune",
    "motif_immune_exclusion",
    "motif_t_tumor_mixing",
    "motif_cd8_tumor_contact",
    "motif_macrophage_tumor_niche",
    "motif_apc_t_contact",
)

# Native UTAG domain portraits (majority cell set + centroid markers).
PORTRAIT_SPECS: dict[str, dict] = {
    "motif_tumor_high": dict(
        kind="control",
        hit_sets=("tumor",),
        hit_markers=("panck",),
        miss_sets=(),
        require_both=False,
    ),
    "motif_cd8_high": dict(
        kind="control",
        hit_sets=("cd8", "t_cell"),
        hit_markers=("cd8", "cd3"),
        miss_sets=(),
        require_both=False,
    ),
    "motif_cd8_clustering": dict(
        kind="spatial",
        hit_sets=("cd8",),
        hit_markers=("cd8",),
        miss_sets=("tumor",),
        require_both=False,
    ),
    "motif_tumor_stroma_mixing": dict(
        kind="spatial",
        hit_sets=("tumor", "stroma"),
        hit_markers=("panck", "vimentin", "asma"),
        miss_sets=(),
        require_both=True,
    ),
    "motif_interface_immune": dict(
        kind="spatial",
        hit_sets=("immune",),
        hit_markers=("cd8", "cd3", "cd20", "cd68", "hla-dr"),
        miss_sets=("tumor",),
        require_both=False,
    ),
    "motif_immune_exclusion": dict(
        kind="spatial",
        hit_sets=("cd8",),
        hit_markers=("cd8",),
        miss_sets=("tumor",),
        require_both=False,
    ),
    "motif_t_tumor_mixing": dict(
        kind="spatial",
        hit_sets=("tumor", "t_cell"),
        hit_markers=("panck", "cd3", "cd8"),
        miss_sets=(),
        require_both=True,
    ),
    "motif_cd8_tumor_contact": dict(
        kind="spatial",
        hit_sets=("tumor", "cd8"),
        hit_markers=("panck", "cd8", "cd3"),
        miss_sets=(),
        require_both=True,
    ),
    "motif_macrophage_tumor_niche": dict(
        kind="spatial",
        hit_sets=("tumor", "macrophage"),
        hit_markers=("panck", "cd68"),
        miss_sets=(),
        require_both=True,
    ),
    "motif_apc_t_contact": dict(
        kind="spatial",
        hit_sets=("apc", "t_cell"),
        hit_markers=("hla-dr", "cd68", "cd3", "cd8"),
        miss_sets=(),
        require_both=True,
    ),
}

# GNN-explainer / node attributions: expected cell-set keys in the motif catalog.
GNN_EXPECTED_SETS: dict[str, tuple[str, ...]] = {
    "motif_tumor_high": ("tumor",),
    "motif_cd8_high": ("cd8", "t_cell"),
    "motif_cd8_clustering": ("cd8",),
    "motif_tumor_stroma_mixing": ("tumor", "stroma"),
    "motif_interface_immune": ("immune",),
    "motif_immune_exclusion": ("cd8",),
    "motif_t_tumor_mixing": ("tumor", "t_cell"),
    "motif_cd8_tumor_contact": ("tumor", "cd8"),
    "motif_macrophage_tumor_niche": ("tumor", "macrophage"),
    "motif_apc_t_contact": ("apc", "t_cell"),
}


@dataclass(frozen=True)
class RecoveryRule:
    task: str
    hit: tuple[str, ...]
    miss_as_top: tuple[str, ...]
    kind: str  # control | spatial


RULES: dict[str, RecoveryRule] = {
    "motif_tumor_high": RecoveryRule(
        task="motif_tumor_high",
        hit=(
            r"composition::.*tumor(?!_stroma)",
            r"tissue_density::.*tumor(?!_stroma)",
            r"tumor_area_ratio",
            r"(^|::)tumor$",
        ),
        miss_as_top=(),
        kind="control",
    ),
    "motif_cd8_high": RecoveryRule(
        task="motif_cd8_high",
        hit=(
            r"composition::.*(cd8|t cells?)",
            r"tissue_density::.*(cd8|t cells?)",
            r"(^|::)(cd8 t cells?|t cells?)$",
        ),
        miss_as_top=(),
        kind="control",
    ),
    "motif_cd8_clustering": RecoveryRule(
        task="motif_cd8_clustering",
        hit=(r"cd8.*[_ ][kl]_r", r"cd8.*ripley", r"cd8.*cluster"),
        miss_as_top=(
            r"composition::.*cd8",
            r"tissue_density::.*cd8",
            r"frac__cd8",
            r"(^|::)cd8 t cell$",
        ),
        kind="spatial",
    ),
    "motif_tumor_stroma_mixing": RecoveryRule(
        task="motif_tumor_stroma_mixing",
        hit=(
            r"stroma.*[_ ][kl]_r",
            r"tumor.*[_ ][kl]_r",
            r"mixing",
        ),
        miss_as_top=(
            r"composition::.*(tumor|stroma)$",
            r"tissue_density::.*(tumor|stroma)$",
            r"(^|::)(tumor|stroma)$",
        ),
        kind="spatial",
    ),
    "motif_interface_immune": RecoveryRule(
        task="motif_interface_immune",
        hit=(
            r"tumor_density::(cd8|cd4|b cell|macrophage|immune)",
            r"interface",
            r"(cd8|cd4|b cell).*[_ ][kl]_r",
        ),
        miss_as_top=(
            r"composition::.*(cd8|cd4|b cell|macrophage|immune)",
            r"tissue_density::.*(cd8|cd4|b cell|macrophage)",
        ),
        kind="spatial",
    ),
    "motif_immune_exclusion": RecoveryRule(
        task="motif_immune_exclusion",
        hit=(
            r"tumor_density::cd8",
            r"tissue_density::cd8",
            r"cd8.*[_ ][kl]_r",
        ),
        miss_as_top=(
            r"composition::.*cd8",
            r"frac__cd8",
        ),
        kind="spatial",
    ),
    "motif_t_tumor_mixing": RecoveryRule(
        task="motif_t_tumor_mixing",
        hit=(
            r"(t cells?|cd8).*[_ ][kl]_r",
            r"tumor.*[_ ][kl]_r",
            r"mixing",
        ),
        miss_as_top=(
            r"composition::.*(tumor|t cells?|cd8)",
            r"tissue_density::.*(tumor|t cells?|cd8)",
        ),
        kind="spatial",
    ),
    "motif_cd8_tumor_contact": RecoveryRule(
        task="motif_cd8_tumor_contact",
        hit=(
            r"(cd8|t cells?).*[_ ][kl]_r",
            r"tumor_density::(cd8|t cells?)",
            r"mixing",
        ),
        miss_as_top=(
            r"composition::.*(tumor|cd8|t cells?)",
            r"tissue_density::.*(tumor|cd8|t cells?)",
        ),
        kind="spatial",
    ),
    "motif_macrophage_tumor_niche": RecoveryRule(
        task="motif_macrophage_tumor_niche",
        hit=(
            r"macrophage.*[_ ][kl]_r",
            r"tumor.*[_ ][kl]_r",
            r"mixing",
        ),
        miss_as_top=(
            r"composition::.*(tumor|macrophage)",
            r"tissue_density::.*(tumor|macrophage)",
        ),
        kind="spatial",
    ),
    "motif_apc_t_contact": RecoveryRule(
        task="motif_apc_t_contact",
        hit=(
            r"(apc|dendritic|macrophage).*[_ ][kl]_r",
            r"(t cells?|cd8).*[_ ][kl]_r",
            r"mixing",
        ),
        miss_as_top=(
            r"composition::.*(apc|dendritic|macrophage|t cells?|cd8)",
            r"tissue_density::.*(apc|dendritic|macrophage|t cells?|cd8)",
        ),
        kind="spatial",
    ),
}


# UTAG columns are either smoothed marker summaries (utag_mean__/utag_std__)
# or train-fold domains named by majority cell type (utag_domain::<type>__kk).
UTAG_RULES: dict[str, RecoveryRule] = {
    "motif_tumor_high": RecoveryRule(
        task="motif_tumor_high",
        hit=(
            r"utag_mean__.*panck",
            r"utag_domain::.*tumor",
        ),
        miss_as_top=(),
        kind="control",
    ),
    "motif_cd8_high": RecoveryRule(
        task="motif_cd8_high",
        hit=(
            r"utag_mean__.*(cd8|cd3)",
            r"utag_domain::.*(cd8|t cell)",
        ),
        miss_as_top=(),
        kind="control",
    ),
    "motif_cd8_clustering": RecoveryRule(
        task="motif_cd8_clustering",
        hit=(r"utag_domain::.*cd8",),
        miss_as_top=(r"utag_mean__.*cd8", r"utag_std__.*cd8"),
        kind="spatial",
    ),
    "motif_tumor_stroma_mixing": RecoveryRule(
        task="motif_tumor_stroma_mixing",
        hit=(
            r"utag_domain::.*(stroma|tumor)",
        ),
        miss_as_top=(r"utag_mean__.*panck", r"utag_mean__.*(vimentin|asma)"),
        kind="spatial",
    ),
    "motif_interface_immune": RecoveryRule(
        task="motif_interface_immune",
        hit=(
            r"utag_domain::.*(cd8|cd4|b cell|macrophage|immune)",
        ),
        miss_as_top=(r"utag_mean__.*panck", r"utag_domain::.*tumor"),
        kind="spatial",
    ),
    "motif_immune_exclusion": RecoveryRule(
        task="motif_immune_exclusion",
        hit=(
            r"utag_domain::.*cd8",
        ),
        miss_as_top=(r"utag_mean__.*panck", r"utag_mean__.*cd8", r"utag_domain::.*tumor"),
        kind="spatial",
    ),
    "motif_t_tumor_mixing": RecoveryRule(
        task="motif_t_tumor_mixing",
        hit=(r"utag_domain::.*(t cell|cd8|tumor)",),
        miss_as_top=(r"utag_mean__.*panck", r"utag_mean__.*(cd3|cd8)"),
        kind="spatial",
    ),
    "motif_cd8_tumor_contact": RecoveryRule(
        task="motif_cd8_tumor_contact",
        hit=(r"utag_domain::.*(cd8|t cell|tumor)",),
        miss_as_top=(r"utag_mean__.*panck", r"utag_mean__.*(cd8|cd3)"),
        kind="spatial",
    ),
    "motif_macrophage_tumor_niche": RecoveryRule(
        task="motif_macrophage_tumor_niche",
        hit=(r"utag_domain::.*(macrophage|tumor)",),
        miss_as_top=(r"utag_mean__.*panck", r"utag_mean__.*cd68"),
        kind="spatial",
    ),
    "motif_apc_t_contact": RecoveryRule(
        task="motif_apc_t_contact",
        hit=(r"utag_domain::.*(apc|dendritic|macrophage|t cell|cd8)",),
        miss_as_top=(r"utag_mean__.*(hla-dr|cd68|cd3|cd8)"),
        kind="spatial",
    ),
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _set_types(catalog, set_name: str) -> set[str]:
    members = catalog.cell_sets.get(set_name) or ()
    return {str(t).strip().lower() for t in members}


def portrait_matches(majority: str, top_markers: list[str], set_fracs: dict[str, float], catalog, spec: dict) -> dict:
    """Does one UTAG domain portrait match the motif's expected cell set / markers?"""
    maj = _norm(majority)
    markers = [_norm(m) for m in top_markers]
    hit = False
    for set_name in spec["hit_sets"]:
        types = _set_types(catalog, set_name)
        if maj in types or float(set_fracs.get(set_name, 0.0)) >= 0.25:
            hit = True
    if any(any(key in marker for key in spec["hit_markers"]) for marker in markers):
        hit = True
    miss = False
    for set_name in spec["miss_sets"]:
        types = _set_types(catalog, set_name)
        if maj in types or float(set_fracs.get(set_name, 0.0)) >= 0.50:
            miss = True
    return {"hit": hit, "miss": miss}


def feature_hits(name: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    text = _norm(name)
    return any(re.search(pat, text) for pat in patterns)


def rank_recovery(
    ranked_features: list[str],
    rule: RecoveryRule,
    k: int = 5,
) -> dict:
    """Score a ranked feature list against a motif recovery rule.

    Spatial tasks fail if the known abundance leak is rank-1, even when a
    spatial hit also appears later in the top-k.
    """
    top = ranked_features[:k]
    hit_in_top = [f for f in top if feature_hits(f, rule.hit)]
    miss_top1 = bool(ranked_features) and feature_hits(ranked_features[0], rule.miss_as_top)
    hit_ranks = [
        i + 1 for i, f in enumerate(ranked_features) if feature_hits(f, rule.hit)
    ]
    recovered = bool(hit_in_top)
    if rule.kind == "control":
        passed = recovered
    else:
        passed = recovered and not miss_top1
    return {
        f"hit_in_top{k}": ";".join(hit_in_top) if hit_in_top else "",
        "n_hit_in_top": len(hit_in_top),
        "best_hit_rank": hit_ranks[0] if hit_ranks else None,
        "top1": ranked_features[0] if ranked_features else "",
        "miss_as_top1": miss_top1,
        "recovered": recovered,
        "passed": passed,
    }
