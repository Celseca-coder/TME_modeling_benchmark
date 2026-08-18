"""Score one RegionData with a frozen MotifCatalog (L0 cell → L3 region)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from benchmark.data.dataset import RegionData
from benchmark.motifs.spec import MotifCatalog, MotifSpec

_TUMOR_KIND = "tumour"


def _cell_type_col(region: RegionData, cell_type_col: str) -> str | None:
    if cell_type_col in region.cell_types.columns:
        return cell_type_col
    if "cell_type" in region.cell_types.columns:
        return "cell_type"
    return None


def aligned_labels(region: RegionData, cell_type_col: str) -> pd.Series:
    col = _cell_type_col(region, cell_type_col)
    if col is None:
        return pd.Series(index=region.coordinates.index, dtype="object")
    return region.cell_types[col].reindex(region.coordinates.index)


def coordinates_um(region: RegionData) -> np.ndarray:
    return region.coordinates[["x", "y"]].to_numpy(float) * float(region.microns_per_pixel)


def membership(labels: pd.Series, members: Iterable[str]) -> np.ndarray:
    members = set(members)
    return labels.isin(members).fillna(False).to_numpy()


def composition_fractions(labels: pd.Series, catalog: MotifCatalog) -> dict[str, float]:
    n = int(labels.notna().sum())
    out: dict[str, float] = {"n_cells": float(n)}
    if n == 0:
        for name in catalog.cell_sets:
            out[f"frac__{name}"] = np.nan
        return out
    values = labels.dropna()
    for name, members in catalog.cell_sets.items():
        out[f"frac__{name}"] = float(values.isin(members).mean())
    return out


def _radius_neighbors(xy_um: np.ndarray, radius_um: float) -> list[list[int]]:
    if len(xy_um) == 0:
        return []
    tree = KDTree(xy_um)
    return tree.query_ball_point(xy_um, r=float(radius_um))


def _mean_neighbor_fraction(
    xy_um: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    radius_um: float,
    min_source_cells: int,
) -> float:
    n_source = int(source.sum())
    if n_source < min_source_cells:
        return np.nan
    neighbors = _radius_neighbors(xy_um, radius_um)
    scores = []
    source_idx = np.flatnonzero(source)
    for i in source_idx:
        other = [j for j in neighbors[i] if j != i]
        if not other:
            continue
        scores.append(float(target[other].mean()))
    return float(np.mean(scores)) if scores else np.nan


def _inside_tumor(region: RegionData) -> np.ndarray | None:
    xy = region.coordinates[["x", "y"]].to_numpy(float)
    return region.polygon_contains(xy, _TUMOR_KIND)


def score_composition(labels: pd.Series, catalog: MotifCatalog, spec: MotifSpec) -> float:
    if spec.cell_set is None:
        raise ValueError(f"{spec.id}: composition motif needs cell_set")
    members = catalog.resolve_set(spec.cell_set)
    valid = labels.dropna()
    if valid.empty:
        return np.nan
    return float(valid.isin(members).mean())


def score_neighbor_fraction(
    region: RegionData,
    labels: pd.Series,
    catalog: MotifCatalog,
    spec: MotifSpec,
) -> float:
    if spec.source_set is None or spec.neighbor_set is None:
        raise ValueError(f"{spec.id}: neighbor_fraction needs source_set and neighbor_set")
    xy = coordinates_um(region)
    source = membership(labels, catalog.resolve_set(spec.source_set))
    target = membership(labels, catalog.resolve_set(spec.neighbor_set))
    return _mean_neighbor_fraction(xy, source, target, spec.radius_um, spec.min_source_cells)


def score_immune_exclusion(
    region: RegionData,
    labels: pd.Series,
    catalog: MotifCatalog,
    spec: MotifSpec,
) -> float:
    """High = CD8 present but not inside tumour (or not touching tumour)."""
    if spec.immune_set is None or spec.tumor_set is None:
        raise ValueError(f"{spec.id}: immune_exclusion needs immune_set and tumor_set")
    immune = membership(labels, catalog.resolve_set(spec.immune_set))
    if int(immune.sum()) < spec.min_source_cells:
        return np.nan
    inside = _inside_tumor(region)
    if inside is not None:
        n_immune = float(immune.sum())
        n_in = float((immune & inside).sum())
        return 1.0 - (n_in / n_immune)
    tumor = membership(labels, catalog.resolve_set(spec.tumor_set))
    mixing = _mean_neighbor_fraction(
        coordinates_um(region), immune, tumor, spec.radius_um, spec.min_source_cells
    )
    return np.nan if np.isnan(mixing) else float(1.0 - mixing)


def tls_l1_mask(
    xy_um: np.ndarray,
    labels: pd.Series,
    catalog: MotifCatalog,
    spec: MotifSpec,
) -> np.ndarray:
    """L1: neighborhood is jointly enriched for every required cell set."""
    if not spec.required_sets:
        raise ValueError(f"{spec.id}: tls_like needs required_sets")
    n = len(xy_um)
    if n == 0:
        return np.zeros(0, dtype=bool)
    set_masks = {
        name: membership(labels, catalog.resolve_set(name)) for name in spec.required_sets
    }
    neighbors = _radius_neighbors(xy_um, spec.radius_um)
    l1 = np.zeros(n, dtype=bool)
    for i, nb in enumerate(neighbors):
        if not nb:
            continue
        ok = True
        for name in spec.required_sets:
            thresh = float(spec.min_frac.get(name, 0.0))
            if set_masks[name][nb].mean() < thresh:
                ok = False
                break
        l1[i] = ok
    return l1


def score_tls_like(
    region: RegionData,
    labels: pd.Series,
    catalog: MotifCatalog,
    spec: MotifSpec,
) -> float:
    """L2/L3: fraction of cells that sit in a large cluster of L1-positive neighborhoods."""
    xy = coordinates_um(region)
    n = len(xy)
    if n == 0:
        return np.nan
    l1 = tls_l1_mask(xy, labels, catalog, spec)
    idx = np.flatnonzero(l1)
    if len(idx) == 0:
        return 0.0
    sub = xy[idx]
    if len(sub) == 1:
        return 0.0 if spec.min_cluster_size > 1 else 1.0 / n
    pairs = KDTree(sub).query_pairs(r=float(spec.radius_um))
    if not pairs:
        return 0.0 if spec.min_cluster_size > 1 else float(len(idx) / n)
    rows: list[int] = []
    cols: list[int] = []
    for i, j in pairs:
        rows.extend((i, j))
        cols.extend((j, i))
    adj = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(idx), len(idx)))
    _n_comp, comp = connected_components(adj, directed=False)
    sizes = pd.Series(comp).value_counts()
    large = set(sizes[sizes >= spec.min_cluster_size].index)
    n_in_large = int(np.isin(comp, list(large)).sum()) if large else 0
    return float(n_in_large / n)


def _distance_to_tumor_boundary_um(region: RegionData) -> np.ndarray | None:
    if region.polygons is None:
        return None
    poly = region.polygons.get(_TUMOR_KIND)
    if poly is None or poly.is_empty:
        return None
    import shapely

    xy = region.coordinates[["x", "y"]].to_numpy(float)
    pts = shapely.points(xy[:, 0], xy[:, 1])
    dist_px = shapely.distance(pts, poly.boundary)
    return np.asarray(dist_px, float) * float(region.microns_per_pixel)


def score_interface(
    region: RegionData,
    labels: pd.Series,
    catalog: MotifCatalog,
    spec: MotifSpec,
) -> float:
    if spec.immune_set is None:
        raise ValueError(f"{spec.id}: interface needs immune_set")
    immune = membership(labels, catalog.resolve_set(spec.immune_set))
    if int(immune.sum()) < spec.min_source_cells:
        return np.nan
    dist_um = _distance_to_tumor_boundary_um(region)
    if dist_um is not None:
        return float((dist_um[immune] <= spec.radius_um).mean())
    if spec.tumor_set is None:
        return np.nan
    tumor = membership(labels, catalog.resolve_set(spec.tumor_set))
    xy = coordinates_um(region)
    neighbors = _radius_neighbors(xy, spec.radius_um)
    interface = []
    for i in np.flatnonzero(immune):
        other = [j for j in neighbors[i] if j != i]
        if not other:
            continue
        has_tumor = bool(tumor[other].any())
        has_nontumor = bool((~tumor[other]).any())
        interface.append(has_tumor and has_nontumor)
    return float(np.mean(interface)) if interface else np.nan


def score_motif(
    region: RegionData,
    catalog: MotifCatalog,
    spec: MotifSpec,
    labels: pd.Series | None = None,
) -> float:
    if labels is None:
        labels = aligned_labels(region, catalog.cell_type_col)
    if spec.kind == "composition":
        return score_composition(labels, catalog, spec)
    if spec.kind == "neighbor_fraction":
        return score_neighbor_fraction(region, labels, catalog, spec)
    if spec.kind == "immune_exclusion":
        return score_immune_exclusion(region, labels, catalog, spec)
    if spec.kind == "tls_like":
        return score_tls_like(region, labels, catalog, spec)
    if spec.kind == "interface":
        return score_interface(region, labels, catalog, spec)
    raise ValueError(f"Unhandled motif kind {spec.kind!r}")


def score_region(region: RegionData, catalog: MotifCatalog) -> dict[str, float]:
    """Return composition auxiliaries plus one score per motif."""
    labels = aligned_labels(region, catalog.cell_type_col)
    out = composition_fractions(labels, catalog)
    for spec in catalog.motifs:
        out[spec.score_col] = score_motif(region, catalog, spec, labels=labels)
    return out


def shuffle_coordinates(region: RegionData, rng: np.random.Generator) -> RegionData:
    """Keep cell types and expression; permute x/y to destroy spatial structure."""
    perm = rng.permutation(len(region.coordinates))
    coords = region.coordinates.copy()
    coords[["x", "y"]] = coords[["x", "y"]].to_numpy()[perm]
    return RegionData(
        region_id=f"{region.region_id}___shuffled",
        coordinates=coords,
        expression=region.expression,
        cell_types=region.cell_types,
        microns_per_pixel=region.microns_per_pixel,
        polygons=region.polygons,
    )
