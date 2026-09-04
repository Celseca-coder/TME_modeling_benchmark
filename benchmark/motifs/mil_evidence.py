"""Window-level geometric evidence for MIL localization recovery.

Each AttnMIL instance is a square window.  Evidence is computed from cells
inside that window plus tumour polygons, using the same catalog cell-sets as
region-level motif scoring.  Positive / negative masks are within-region
tertiles of a continuous score, except where the protocol names a geometric
exclusion zone (tumour core, intra-tumour CD8).
"""
from __future__ import annotations

import numpy as np

from benchmark.data.dataset import RegionData
from benchmark.motifs.detect import (
    _distance_to_tumor_boundary_um,
    _inside_tumor,
    _mean_neighbor_fraction,
    aligned_labels,
    coordinates_um,
    membership,
)
from benchmark.motifs.recovery import RULES
from benchmark.motifs.spec import MotifCatalog, MotifSpec


def window_inside_mask(
    xy_um: np.ndarray,
    center_um: np.ndarray,
    window_size_um: float,
) -> np.ndarray:
    half = 0.5 * float(window_size_um)
    cx, cy = float(center_um[0]), float(center_um[1])
    return (
        (xy_um[:, 0] >= cx - half)
        & (xy_um[:, 0] < cx + half)
        & (xy_um[:, 1] >= cy - half)
        & (xy_um[:, 1] < cy + half)
    )


def _frac(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _local_neighbor_fraction(
    xy_um: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    radius_um: float,
) -> float:
    if int(source.sum()) < 1 or len(xy_um) < 2:
        return 0.0
    value = _mean_neighbor_fraction(xy_um, source, target, radius_um, min_source_cells=1)
    return 0.0 if value != value else float(value)


def _center_in_tumor(region: RegionData, center_um: np.ndarray) -> bool | None:
    if region.polygons is None:
        return None
    mpp = float(region.microns_per_pixel) or 1.0
    xy_px = np.asarray([[center_um[0] / mpp, center_um[1] / mpp]], dtype=float)
    hit = region.polygon_contains(xy_px, "tumour")
    if hit is None:
        return None
    return bool(hit[0])


def _center_boundary_um(region: RegionData, center_um: np.ndarray) -> float | None:
    dist = _distance_to_tumor_boundary_um(region)
    if dist is None or len(dist) == 0:
        return None
    xy = coordinates_um(region)
    d2 = ((xy - np.asarray(center_um, float)[None, :]) ** 2).sum(axis=1)
    return float(dist[int(np.argmin(d2))])


def window_score(
    region: RegionData,
    catalog: MotifCatalog,
    spec: MotifSpec,
    inside: np.ndarray,
    center_um: np.ndarray,
) -> dict[str, float]:
    """Continuous evidence plus geometric flags for one window."""
    labels = aligned_labels(region, catalog.cell_type_col)
    xy = coordinates_um(region)
    inside = np.asarray(inside, dtype=bool)
    if inside.sum() == 0:
        return dict(score=0.0, pos=0.0, neg=0.0)

    xy_w = xy[inside]
    labels_w = labels.iloc[np.flatnonzero(inside)]

    def set_mask(name: str | None) -> np.ndarray:
        if not name:
            return np.zeros(int(inside.sum()), dtype=bool)
        return membership(labels_w, catalog.resolve_set(name))

    tumor = set_mask("tumor") if "tumor" in catalog.cell_sets else np.zeros(int(inside.sum()), dtype=bool)
    out = dict(score=0.0, pos=0.0, neg=0.0)

    if spec.kind == "composition":
        frac = _frac(set_mask(spec.cell_set))
        out["score"] = frac
        return out

    if spec.kind == "neighbor_fraction":
        src = set_mask(spec.source_set)
        nb = set_mask(spec.neighbor_set)
        mix = _local_neighbor_fraction(xy_w, src, nb, spec.radius_um)
        both = float(src.any() and nb.any())
        out["score"] = mix * both + 0.05 * min(_frac(src), _frac(nb))
        pure_src = _frac(src) >= 0.80 and _frac(nb) <= 0.05
        pure_nb = _frac(nb) >= 0.80 and _frac(src) <= 0.05
        clustering = spec.source_set == spec.neighbor_set
        if clustering:
            out["neg"] = float(_frac(tumor) >= 0.80 and _frac(src) <= 0.05)
        else:
            out["neg"] = float(pure_src or pure_nb)
        out["pos"] = float(both and mix >= 0.15)
        return out

    if spec.kind == "immune_exclusion":
        immune = set_mask(spec.immune_set)
        parent_immune = membership(labels, catalog.resolve_set(spec.immune_set))
        inside_tumor = _inside_tumor(region)
        frac_cd8 = _frac(immune)
        if inside_tumor is not None:
            cd8_idx = np.flatnonzero(parent_immune & inside)
            if len(cd8_idx):
                frac_out = 1.0 - float(inside_tumor[cd8_idx].mean())
            else:
                frac_out = 0.0
            in_tumor = _center_in_tumor(region, center_um)
            out["score"] = frac_cd8 * frac_out
            out["pos"] = float(frac_cd8 > 0 and in_tumor is False)
            out["neg"] = float(frac_cd8 > 0 and in_tumor is True)
            return out
        tumor_w = set_mask(spec.tumor_set)
        mix = _local_neighbor_fraction(xy_w, immune, tumor_w, spec.radius_um)
        out["score"] = frac_cd8 * (1.0 - mix)
        out["pos"] = float(frac_cd8 > 0 and mix < 0.15)
        out["neg"] = float(frac_cd8 > 0 and mix >= 0.40)
        return out

    if spec.kind == "interface":
        immune = set_mask(spec.immune_set)
        dist = _center_boundary_um(region, center_um)
        if dist is None:
            tumor_w = set_mask(spec.tumor_set)
            mix = _local_neighbor_fraction(xy_w, immune, tumor_w, spec.radius_um)
            out["score"] = _frac(immune) * mix
            out["pos"] = float(_frac(immune) > 0 and mix >= 0.15)
            out["neg"] = float(_frac(tumor) >= 0.80 and _frac(immune) <= 0.05)
            return out
        near = dist <= float(spec.radius_um)
        out["score"] = _frac(immune) * (1.0 / (1.0 + dist / max(spec.radius_um, 1.0)))
        out["pos"] = float(_frac(immune) > 0 and near)
        out["neg"] = float(_frac(tumor) >= 0.80 and dist > 2 * spec.radius_um)
        return out

    if spec.kind == "tls_like":
        out["score"] = _frac(set_mask("t_cell")) * _frac(set_mask("b_cell"))
        return out

    return out


def _tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        z = np.zeros_like(values, dtype=bool)
        return z, z
    lo, hi = np.quantile(finite, [1 / 3, 2 / 3])
    if hi <= lo:
        pos = values >= np.median(finite)
        neg = ~pos
        return pos, neg
    return values >= hi, values <= lo


def evidence_for_bag(
    region: RegionData,
    catalog: MotifCatalog,
    spec: MotifSpec,
    centers_um: np.ndarray,
    window_size_um: float,
) -> dict[str, np.ndarray]:
    """Return per-window score / positive / negative evidence arrays."""
    xy = coordinates_um(region)
    n = len(centers_um)
    scores = np.zeros(n, dtype=float)
    pos = np.zeros(n, dtype=bool)
    neg = np.zeros(n, dtype=bool)
    for i, center in enumerate(centers_um):
        inside = window_inside_mask(xy, center, window_size_um)
        rec = window_score(region, catalog, spec, inside, center)
        scores[i] = rec["score"]
        pos[i] = bool(rec.get("pos", 0.0))
        neg[i] = bool(rec.get("neg", 0.0))
    if spec.kind == "composition" or not pos.any():
        pos, tert_neg = _tertiles(scores)
        if spec.kind == "composition":
            neg = tert_neg
        elif not neg.any():
            neg = tert_neg
    pos = pos & ~neg
    return dict(score=scores, pos=pos.astype(float), neg=neg.astype(float))


def spec_for_task(catalog: MotifCatalog, task: str) -> MotifSpec:
    motif_id = str(task).removeprefix("motif_")
    return catalog.motif(motif_id)


def task_kind(task: str) -> str:
    rule = RULES.get(task)
    return rule.kind if rule is not None else "unknown"
