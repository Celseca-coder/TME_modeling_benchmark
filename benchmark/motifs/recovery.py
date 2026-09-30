"""Rank-based motif recovery: hit regexes, UTAG / portrait / GNN specs.

Tabular ``RULES`` are built from catalog cell-set aliases × feature-family
templates, so HNC ``Tumor`` and TNBC ``panCK^{med}`` share the same task
definition.  Spatial tasks fail when a known abundance leak is rank-1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "motifs"

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


@dataclass(frozen=True)
class RecoveryRule:
    task: str
    hit: tuple[str, ...]
    miss_as_top: tuple[str, ...]
    kind: str


# ---------------------------------------------------------------------------
# Cell-set aliases: catalog members (exact, escaped) + cross-cohort synonyms
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, tuple[str, ...]] = {
    "tumor": (
        r"tumor(?!_stroma)",
        r"epithelial",
        r"panck",
        r"pan-?ck",
        r"cytokeratin",
        r"ck[0-9^/]",
        r"basal",
        r"her2",
        r"hr[+\-]",
        r"lar\b",
        r"myoepithelial",
    ),
    # ``t cell`` is a Jackson/METABRIC CD8 proxy. Bound it so ``CD4 T cell``
    # does not count as CD8 (the bare substring ``t cell`` would).
    "cd8": (r"cd8(?!\d)", r"(?:^|(?<=::)|(?<=__))t cells?(?=__|$|[ _]?[kl]_r)"),
    "t_cell": (r"cd8(?!\d)", r"cd4(?!\d)", r"t cells?", r"treg"),
    "macrophage": (r"macrophage", r"m2 mac", r"\bmacs?\b"),
    "apc": (r"apc", r"dendritic", r"\bdcs?\b"),
    "stroma": (r"stroma", r"fibroblast", r"myofibroblast"),
    "immune": (
        r"immune",
        r"cd8(?!\d)",
        r"cd4(?!\d)",
        r"t cells?",
        r"treg",
        r"b cells?",
        r"macrophage",
        r"m2 mac",
        r"apc",
        r"dendritic",
        r"\bdcs?\b",
        r"granulocyte",
        r"neutrophil",
        r"nk\b",
    ),
    "b_cell": (r"b cells?",),
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _catalog_members(set_name: str) -> tuple[str, ...]:
    names: list[str] = []
    if not _CONFIGS_DIR.is_dir():
        return ()
    for path in sorted(_CONFIGS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for member in data.get("cell_sets", {}).get(set_name, []) or []:
            names.append(_norm(member))
    return tuple(dict.fromkeys(names))


def _token(escaped: str) -> str:
    """Catalog type names must not match as substrings of another type.

    Leading ``::`` / ``__`` may already have been consumed by a family prefix
    (``composition::``, ``neighbor_fraction__``), so allow a lookbehind.
    """
    return rf"(?:^|(?<=::)|(?<=__)){escaped}(?=__|$|[ _]?[kl]_r)"


def _alias(set_name: str, extra: tuple[str, ...] | None = None) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for raw in _catalog_members(set_name):
        tok = _token(re.escape(raw))
        if tok not in seen:
            seen.add(tok)
            parts.append(tok)
    for pat in extra if extra is not None else SYNONYMS.get(set_name, ()):
        if pat and pat not in seen:
            seen.add(pat)
            parts.append(pat)
    if not parts:
        raise ValueError(f"Empty alias for cell set {set_name!r}")
    return "(?:" + "|".join(parts) + ")"


TUMOR = _alias("tumor")
CD8 = _alias("cd8")
T_CELL = _alias("t_cell")
MAC = _alias("macrophage")
APC = _alias("apc")
STROMA = _alias("stroma")
IMMUNE = _alias("immune")
B_CELL = _alias("b_cell")

ALIASES = {
    "tumor": TUMOR,
    "cd8": CD8,
    "t_cell": T_CELL,
    "macrophage": MAC,
    "apc": APC,
    "stroma": STROMA,
    "immune": IMMUNE,
    "b_cell": B_CELL,
}


def _or(*aliases: str) -> str:
    parts = list(dict.fromkeys(aliases))
    if len(parts) == 1:
        return parts[0]
    return "(?:" + "|".join(parts) + ")"


def _comp(alias: str, raw: str | None = None) -> tuple[str, ...]:
    pats = [rf"composition::.*{alias}", rf"tissue_density::.*{alias}"]
    if raw:
        pats.append(rf"(^|::){raw}$")
    return tuple(pats)


def _abund(alias: str) -> tuple[str, ...]:
    # Do not use `(^|::){alias}$` here: it also matches spatial columns
    # that merely end in the cell type, e.g. `tumor_density::CD8 T cell`.
    return (
        rf"composition::.*{alias}",
        rf"tissue_density::.*{alias}",
        rf"frac__{alias}",
    )


def _tumdens(alias: str) -> str:
    return rf"tumor_density::.*{alias}"


def _ripley(alias: str, *, cluster: bool = False) -> str:
    tail = r"[_ ][kl]_r|ripley|pcf_r"
    if cluster:
        tail += r"|cluster"
    return rf"{alias}.*(?:{tail})"


def _pair(src: str, dst: str) -> tuple[str, ...]:
    return (
        rf"neighbor_fraction__.*{src}.*__to__.*{dst}",
        rf"neighbor_fraction__.*{dst}.*__to__.*{src}",
    )


def _dist_pair(src: str, dst: str) -> tuple[str, ...]:
    return (
        rf"(min|mean)_dist_.*{src}.*_to_.*{dst}",
        rf"(min|mean)_dist_.*{dst}.*_to_.*{src}",
        rf"spatial-distance::.*{src}.*{dst}",
        rf"spatial-distance::.*{dst}.*{src}",
    )


SPD_CROSS = r"spatial-distance::cross_neighbor_dist"
SPD_SAME = r"spatial-distance::(?:same_neighbor_dist|proportion_same_nearest|mean_proportion_same_neighbors)"
LOCAL_MIX = r"mixing::(?:local_mixing|same_type_neighbor_fraction)"
AREA_RATIO = r"tumor_area_ratio"


def _rule(task: str, kind: str, hit: tuple[str, ...], miss: tuple[str, ...] = ()) -> RecoveryRule:
    return RecoveryRule(task=task, hit=hit, miss_as_top=miss, kind=kind)


RULES: dict[str, RecoveryRule] = {
    "motif_tumor_high": _rule(
        "motif_tumor_high",
        "control",
        (*_comp(TUMOR, raw=r"(?:tumor(?!_stroma)|epithelial|panck|pan-?ck)"), AREA_RATIO),
    ),
    "motif_cd8_high": _rule(
        "motif_cd8_high",
        "control",
        _comp(CD8, raw=r"(?:cd8 t cells?|t cells?)"),
    ),
    "motif_cd8_clustering": _rule(
        "motif_cd8_clustering",
        "spatial",
        (
            _ripley(CD8, cluster=True),
            *_pair(CD8, CD8),
            LOCAL_MIX,
            SPD_SAME,
        ),
        _abund(CD8),
    ),
    "motif_tumor_stroma_mixing": _rule(
        "motif_tumor_stroma_mixing",
        "spatial",
        (
            *_pair(TUMOR, STROMA),
            _ripley(TUMOR),
            _ripley(STROMA),
            LOCAL_MIX,
            *_dist_pair(TUMOR, STROMA),
            SPD_CROSS,
            SPD_SAME,
        ),
        _abund(_or(TUMOR, STROMA)),
    ),
    "motif_interface_immune": _rule(
        "motif_interface_immune",
        "spatial",
        (
            _tumdens(IMMUNE),
            *_pair(IMMUNE, TUMOR),
            _ripley(_or(CD8, T_CELL, B_CELL, MAC, IMMUNE)),
            r"interface",
            *_dist_pair(IMMUNE, TUMOR),
        ),
        (*_abund(_or(IMMUNE, TUMOR)), AREA_RATIO),
    ),
    "motif_immune_exclusion": _rule(
        "motif_immune_exclusion",
        "spatial",
        (
            _tumdens(CD8),
            *_pair(CD8, TUMOR),
            _ripley(CD8),
            *_dist_pair(CD8, TUMOR),
            SPD_CROSS,
        ),
        _abund(CD8),
    ),
    "motif_t_tumor_mixing": _rule(
        "motif_t_tumor_mixing",
        "spatial",
        (
            *_pair(T_CELL, TUMOR),
            _tumdens(T_CELL),
            _ripley(T_CELL),
            _ripley(TUMOR),
            *_dist_pair(T_CELL, TUMOR),
            SPD_CROSS,
        ),
        _abund(_or(TUMOR, T_CELL)),
    ),
    "motif_cd8_tumor_contact": _rule(
        "motif_cd8_tumor_contact",
        "spatial",
        (
            *_pair(CD8, TUMOR),
            _tumdens(CD8),
            _ripley(CD8),
            _ripley(TUMOR),
            *_dist_pair(CD8, TUMOR),
            SPD_CROSS,
        ),
        _abund(_or(TUMOR, CD8)),
    ),
    "motif_macrophage_tumor_niche": _rule(
        "motif_macrophage_tumor_niche",
        "spatial",
        (
            *_pair(MAC, TUMOR),
            _tumdens(MAC),
            _ripley(MAC),
            _ripley(TUMOR),
            *_dist_pair(MAC, TUMOR),
            SPD_CROSS,
        ),
        _abund(_or(TUMOR, MAC)),
    ),
    "motif_apc_t_contact": _rule(
        "motif_apc_t_contact",
        "spatial",
        (
            *_pair(APC, T_CELL),
            _ripley(APC),
            _ripley(T_CELL),
            *_dist_pair(APC, T_CELL),
            SPD_CROSS,
        ),
        _abund(_or(APC, T_CELL)),
    ),
}


def _mil_abund(alias: str) -> tuple[str, ...]:
    return (
        rf"composition__.*{alias}",
        rf"celltype_density__.*{alias}",
    )


MIL_FEATURE_RULES: dict[str, RecoveryRule] = {
    "motif_tumor_high": _rule(
        "motif_tumor_high",
        "control",
        (
            rf"composition__.*{TUMOR}",
            rf"celltype_density__.*{TUMOR}",
            r"expression__.*panck",
        ),
    ),
    "motif_cd8_high": _rule(
        "motif_cd8_high",
        "control",
        (
            rf"composition__.*{CD8}",
            rf"celltype_density__.*{CD8}",
            r"expression__.*(cd8|cd3)",
        ),
    ),
    "motif_cd8_clustering": _rule(
        "motif_cd8_clustering",
        "spatial",
        (
            *_pair(CD8, CD8),
            r"mixing__same_type_neighbor_fraction",
            r"mixing__local_mixing",
        ),
        _mil_abund(CD8),
    ),
    "motif_tumor_stroma_mixing": _rule(
        "motif_tumor_stroma_mixing",
        "spatial",
        (
            *_pair(STROMA, TUMOR),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(TUMOR, STROMA)),
    ),
    "motif_interface_immune": _rule(
        "motif_interface_immune",
        "spatial",
        (
            *_pair(IMMUNE, TUMOR),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(IMMUNE, TUMOR)),
    ),
    "motif_immune_exclusion": _rule(
        "motif_immune_exclusion",
        "spatial",
        (
            *_pair(CD8, TUMOR),
            r"mixing__same_type_neighbor_fraction",
            r"mixing__local_mixing",
        ),
        _mil_abund(CD8),
    ),
    "motif_t_tumor_mixing": _rule(
        "motif_t_tumor_mixing",
        "spatial",
        (
            *_pair(T_CELL, TUMOR),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(TUMOR, T_CELL)),
    ),
    "motif_cd8_tumor_contact": _rule(
        "motif_cd8_tumor_contact",
        "spatial",
        (
            *_pair(CD8, TUMOR),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(TUMOR, CD8)),
    ),
    "motif_macrophage_tumor_niche": _rule(
        "motif_macrophage_tumor_niche",
        "spatial",
        (
            *_pair(MAC, TUMOR),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(TUMOR, MAC)),
    ),
    "motif_apc_t_contact": _rule(
        "motif_apc_t_contact",
        "spatial",
        (
            *_pair(APC, T_CELL),
            r"mixing__local_mixing",
        ),
        _mil_abund(_or(APC, T_CELL)),
    ),
}


UTAG_RULES: dict[str, RecoveryRule] = {
    "motif_tumor_high": _rule(
        "motif_tumor_high", "control",
        (r"utag_mean__.*panck", r"utag_domain::.*tumor"),
    ),
    "motif_cd8_high": _rule(
        "motif_cd8_high", "control",
        (r"utag_mean__.*(cd8|cd3)", r"utag_domain::.*(cd8|t cell)"),
    ),
    "motif_cd8_clustering": _rule(
        "motif_cd8_clustering", "spatial",
        (r"utag_domain::.*cd8",),
        (r"utag_mean__.*cd8", r"utag_std__.*cd8"),
    ),
    "motif_tumor_stroma_mixing": _rule(
        "motif_tumor_stroma_mixing", "spatial",
        (r"utag_domain::.*(stroma|tumor)",),
        (r"utag_mean__.*panck", r"utag_mean__.*(vimentin|asma)"),
    ),
    "motif_interface_immune": _rule(
        "motif_interface_immune", "spatial",
        (r"utag_domain::.*(cd8|cd4|b cell|macrophage|immune)",),
        (r"utag_mean__.*panck", r"utag_domain::.*tumor"),
    ),
    "motif_immune_exclusion": _rule(
        "motif_immune_exclusion", "spatial",
        (r"utag_domain::.*cd8",),
        (r"utag_mean__.*panck", r"utag_mean__.*cd8", r"utag_domain::.*tumor"),
    ),
    "motif_t_tumor_mixing": _rule(
        "motif_t_tumor_mixing", "spatial",
        (r"utag_domain::.*(t cell|cd8|tumor)",),
        (r"utag_mean__.*panck", r"utag_mean__.*(cd3|cd8)"),
    ),
    "motif_cd8_tumor_contact": _rule(
        "motif_cd8_tumor_contact", "spatial",
        (r"utag_domain::.*(cd8|t cell|tumor)",),
        (r"utag_mean__.*panck", r"utag_mean__.*(cd8|cd3)"),
    ),
    "motif_macrophage_tumor_niche": _rule(
        "motif_macrophage_tumor_niche", "spatial",
        (r"utag_domain::.*(macrophage|tumor)",),
        (r"utag_mean__.*panck", r"utag_mean__.*cd68"),
    ),
    "motif_apc_t_contact": _rule(
        "motif_apc_t_contact", "spatial",
        (r"utag_domain::.*(apc|dendritic|macrophage|t cell|cd8)",),
        (r"utag_mean__.*(hla-dr|cd68|cd3|cd8)",),
    ),
}

PORTRAIT_SPECS: dict[str, dict] = {
    "motif_tumor_high": {
        "kind": "control",
        "hit_sets": ("tumor",),
        "hit_markers": ("panck",),
        "miss_sets": (),
        "require_both": False,
    },
    "motif_cd8_high": {
        "kind": "control",
        "hit_sets": ("cd8", "t_cell"),
        "hit_markers": ("cd8", "cd3"),
        "miss_sets": (),
        "require_both": False,
    },
    "motif_cd8_clustering": {
        "kind": "spatial",
        "hit_sets": ("cd8",),
        "hit_markers": ("cd8",),
        "miss_sets": ("tumor",),
        "require_both": False,
    },
    "motif_tumor_stroma_mixing": {
        "kind": "spatial",
        "hit_sets": ("tumor", "stroma"),
        "hit_markers": ("panck", "vimentin", "asma"),
        "miss_sets": (),
        "require_both": True,
    },
    "motif_interface_immune": {
        "kind": "spatial",
        "hit_sets": ("immune",),
        "hit_markers": ("cd8", "cd3", "cd20", "cd68", "hla-dr"),
        "miss_sets": ("tumor",),
        "require_both": False,
    },
    "motif_immune_exclusion": {
        "kind": "spatial",
        "hit_sets": ("cd8",),
        "hit_markers": ("cd8",),
        "miss_sets": ("tumor",),
        "require_both": False,
    },
    "motif_t_tumor_mixing": {
        "kind": "spatial",
        "hit_sets": ("tumor", "t_cell"),
        "hit_markers": ("panck", "cd3", "cd8"),
        "miss_sets": (),
        "require_both": True,
    },
    "motif_cd8_tumor_contact": {
        "kind": "spatial",
        "hit_sets": ("tumor", "cd8"),
        "hit_markers": ("panck", "cd8", "cd3"),
        "miss_sets": (),
        "require_both": True,
    },
    "motif_macrophage_tumor_niche": {
        "kind": "spatial",
        "hit_sets": ("tumor", "macrophage"),
        "hit_markers": ("panck", "cd68"),
        "miss_sets": (),
        "require_both": True,
    },
    "motif_apc_t_contact": {
        "kind": "spatial",
        "hit_sets": ("apc", "t_cell"),
        "hit_markers": ("hla-dr", "cd68", "cd3", "cd8"),
        "miss_sets": (),
        "require_both": True,
    },
}

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


def _set_types(catalog, set_name: str) -> set[str]:
    members = catalog.cell_sets.get(set_name) or ()
    return {str(t).strip().lower() for t in members}


def portrait_matches(majority, top_markers, set_fracs, catalog, spec) -> dict[str, bool]:
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
        if maj in types or float(set_fracs.get(set_name, 0.0)) >= 0.5:
            miss = True
    return {"hit": hit, "miss": miss}


def feature_hits(name: str, patterns) -> bool:
    if not patterns:
        return False
    if isinstance(patterns, str):
        patterns = (patterns,)
    text = _norm(name)
    return any(re.search(pat, text) for pat in patterns)


def rank_recovery(ranked_features, rule: RecoveryRule, k: int = 5) -> dict:
    """Score a ranked feature list against a motif recovery rule.

    Spatial tasks fail if the known abundance leak is rank-1, even when a
    spatial hit also appears later in the top-k.
    """
    top = ranked_features[:k]
    hit_in_top = [f for f in top if feature_hits(f, rule.hit)]
    miss_top1 = bool(ranked_features) and feature_hits(ranked_features[0], rule.miss_as_top)
    hit_ranks = [i + 1 for i, f in enumerate(ranked_features) if feature_hits(f, rule.hit)]
    recovered = bool(hit_in_top)
    passed = recovered if rule.kind == "control" else recovered and not miss_top1
    return {
        f"hit_in_top{k}": ";".join(hit_in_top) if hit_in_top else "",
        "n_hit_in_top": len(hit_in_top),
        "best_hit_rank": hit_ranks[0] if hit_ranks else None,
        "top1": ranked_features[0] if ranked_features else "",
        "miss_as_top1": miss_top1,
        "recovered": recovered,
        "passed": passed,
    }
