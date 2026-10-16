"""Window-level geometric evidence for MIL localization recovery.

Each AttnMIL instance is a 100 µm square (50 µm step). Detectors use the same
catalog recipes as region scoring, applied to cells inside the window (TLS
reuses the region cluster mask, then averages inside the window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from benchmark.features.window_grid import (
    DEFAULT_MIN_CELLS,
    DEFAULT_STEP_UM,
    DEFAULT_WINDOW_SIZE_UM,
    Window,
    iter_windows,
    window_centers_um,
)
from benchmark.motifs.detect import (
    _distance_to_tumor_boundary_um,
    _inside_tumor,
    _mean_neighbor_fraction,
    _radius_neighbors,
    aligned_labels,
    coordinates_um,
    membership,
    tls_large_cluster_mask,
)
from benchmark.motifs.labels import discovery_mask, working_score
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
        return float("nan")
    return float(_mean_neighbor_fraction(xy_um, source, target, radius_um, min_source_cells=1))


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
        empty = dict(score=np.nan, pos=0.0, neg=0.0)
        for name in catalog.cell_sets:
            empty[f"frac__{name}"] = np.nan
        return empty

    xy_w = xy[inside]
    labels_w = labels.iloc[np.flatnonzero(inside)]

    def set_mask(name: str | None) -> np.ndarray:
        if not name:
            return np.zeros(int(inside.sum()), dtype=bool)
        return membership(labels_w, catalog.resolve_set(name))

    tumor = set_mask("tumor") if "tumor" in catalog.cell_sets else np.zeros(int(inside.sum()), dtype=bool)
    out = dict(score=0.0, pos=0.0, neg=0.0)
    # 加上各cell set的窗组成
    for name in catalog.cell_sets:
        out[f"frac__{name}"] = _frac(set_mask(name))
    if spec.kind == "composition":
        out["score"] = _frac(set_mask(spec.cell_set))
        return out

    if spec.kind == "neighbor_fraction":
        src = set_mask(spec.source_set)
        nb = set_mask(spec.neighbor_set)
        if int(src.sum()) < spec.min_source_cells:
            out["score"] = float("nan")
            return out
        out["score"] = _local_neighbor_fraction(xy_w, src, nb, spec.radius_um)
        return out

    if spec.kind == "immune_exclusion":
        immune = set_mask(spec.immune_set)
        if int(immune.sum()) < spec.min_source_cells:
            out["score"] = float("nan")
            return out
        inside_tumor = _inside_tumor(region)
        if inside_tumor is not None:
            parent_immune = membership(labels, catalog.resolve_set(spec.immune_set))
            cd8_idx = np.flatnonzero(parent_immune & inside)
            frac_out = 1.0 - float(inside_tumor[cd8_idx].mean()) if len(cd8_idx) else np.nan
            out["score"] = frac_out
            return out
        tumor_w = set_mask(spec.tumor_set)
        mix = _local_neighbor_fraction(xy_w, immune, tumor_w, spec.radius_um)
        out["score"] = 1.0 - mix
        return out

    if spec.kind == "interface":
        immune = set_mask(spec.immune_set)
        if int(immune.sum()) < spec.min_source_cells:
            out["score"] = float("nan")
            return out
        dist_um = _distance_to_tumor_boundary_um(region)
        if dist_um is not None:
            parent_immune = membership(labels, catalog.resolve_set(spec.immune_set))
            idx = np.flatnonzero(parent_immune & inside)
            out["score"] = (
                float((dist_um[idx] <= float(spec.radius_um)).mean())
                if len(idx)
                else float("nan")
            )
            return out
        if spec.tumor_set is None:
            out["score"] = float("nan")
            return out
        tumor_w = set_mask(spec.tumor_set)
        neighbors = _radius_neighbors(xy_w, spec.radius_um)
        interface = []
        for i in np.flatnonzero(immune):
            other = [j for j in neighbors[i] if j != i]
            if not other:
                continue
            has_tumor = bool(tumor_w[other].any())
            has_nontumor = bool((~tumor_w[other]).any())
            interface.append(has_tumor and has_nontumor)
        out["score"] = float(np.mean(interface)) if interface else float("nan")
        return out

    if spec.kind == "tls_like":
        cluster = tls_large_cluster_mask(region, labels, catalog, spec)
        out["score"] = float(cluster[inside].mean()) if inside.any() else 0.0
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
    insides: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return per-window raw scores and the window_score records (including frac__)."""
    xy = coordinates_um(region)
    n = len(centers_um)
    scores = np.full(n, np.nan, dtype=float)
    extra: list[dict[str, float]] = []
    for i, center in enumerate(centers_um):
        inside = insides[i] if insides is not None else window_inside_mask(xy, center, window_size_um)
        rec = window_score(region, catalog, spec, inside, center)
        scores[i] = rec["score"]
        extra.append(rec)
    return dict(score=scores, recs=extra)


def window_label(pos: bool, neg: bool) -> float:
    """Instance GT: 1 / 0 / NaN (middle or unlabeled)."""
    if pos and not neg:
        return 1.0
    if neg and not pos:
        return 0.0
    return np.nan


def label_bag_windows(
    region: RegionData,
    catalog: MotifCatalog,
    spec: MotifSpec,
    windows: list[Window] | None = None,
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> pd.DataFrame:
    """One row per occupied window for a single motif."""
    if windows is None:
        windows = iter_windows(region, window_size_um, step_um, min_cells)
    centers = window_centers_um(windows)
    insides = [w.inside for w in windows]
    ev = evidence_for_bag(
        region, catalog, spec, centers, window_size_um, insides=insides,
    )
    frac_names = [f"frac__{name}" for name in catalog.cell_sets]
    rows = []
    for i, win in enumerate(windows):
        rec = ev["recs"][i]
        n_cells = int(win.inside.sum())
        score = rec["score"]
        if n_cells <= 5:
            score = np.nan
        row = {
            "region_id": region.region_id,
            "window_id": win.window_id,
            "cx_um": float(win.center_um[0]),
            "cy_um": float(win.center_um[1]),
            "n_cells": n_cells,
            "motif_id": spec.id,
            "score": float(score) if score == score else np.nan,
            "pos": np.nan,
            "neg": np.nan,
            "label": np.nan,
        }
        for name in frac_names:
            row[name] = rec.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def label_windows_for_region(
    region: RegionData,
    catalog: MotifCatalog,
    motif_ids: list[str] | None = None,
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> pd.DataFrame:
    windows = iter_windows(region, window_size_um, step_um, min_cells)
    specs = catalog.motifs if motif_ids is None else tuple(catalog.motif(i) for i in motif_ids)
    frames = [
        label_bag_windows(region, catalog, spec, windows=windows, window_size_um=window_size_um)
        for spec in specs
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def keep_window_indices(n: int, max_instances: int | None = 512, training: bool = False, seed: int = 0) -> np.ndarray:
    """Same subsample AttnMIL uses when a bag exceeds max_instances."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    if not max_instances or n <= max_instances:
        return np.arange(n, dtype=int)
    if training:
        return np.sort(np.random.default_rng(seed).choice(n, max_instances, replace=False))
    return np.linspace(0, n - 1, max_instances, dtype=int)


def align_attention_to_windows(
    table: pd.DataFrame,
    weights: np.ndarray,
    max_instances: int | None = 512,
) -> pd.DataFrame:
    """Attach attention to one region's window table (one motif or all windows of one bag)."""
    out = table.sort_values("window_id").reset_index(drop=True)
    n = len(out)
    idx = keep_window_indices(n, max_instances=max_instances, training=False)
    att = np.full(n, np.nan, dtype=float)
    w = np.asarray(weights, float).ravel()
    if len(w) != len(idx):
        raise ValueError(f"attention length {len(w)} != kept windows {len(idx)}")
    att[idx] = w
    out = out.copy()
    out["attention"] = att
    return out


def spec_for_task(catalog: MotifCatalog, task: str) -> MotifSpec:
    motif_id = str(task).removeprefix("motif_")
    return catalog.motif(motif_id)


def task_kind(task: str) -> str:
    rule = RULES.get(task)
    return rule.kind if rule is not None else "unknown"


def add_window_pseudo_labels(table: pd.DataFrame, catalog: MotifCatalog) -> pd.DataFrame:
    """Region-style residual + median/tertile_extremes, fit on discovery windows."""
    fit_all = discovery_mask(table, catalog.cv_filter)
    frames = []
    for spec in catalog.motifs:
        part = table[table["motif_id"] == spec.id].copy()
        if part.empty:
            continue
        part.loc[part["n_cells"] <= 5, "score"] = np.nan
        part[spec.score_col] = part["score"]
        fit = fit_all.reindex(part.index).fillna(False)
        used = working_score(part, spec, fit_index=part.index[fit])
        part["score_used"] = used
        fit_vals = used.loc[fit & used.notna()]
        label = pd.Series(np.nan, index=part.index, dtype="float")
        if spec.label_rule == "median":
            if len(fit_vals):
                cut = float(fit_vals.median())
                ok = used.notna()
                label.loc[ok] = (used.loc[ok] >= cut).astype(float)
        else:
            if len(fit_vals) >= 6:
                lo, hi = np.nanquantile(fit_vals.to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
                if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                    label.loc[used <= lo] = 0.0
                    label.loc[used >= hi] = 1.0
        part["label"] = label
        part["pos"] = (label == 1.0).astype(float)
        part["neg"] = (label == 0.0).astype(float)
        part.loc[label.isna(), ["pos", "neg"]] = np.nan
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else table