"""Square-window features pooled back to one row per region.

Windows are 100 µm squares. Each window is the same four extractors used by the
global tabular explanation (composition, density, mixing, point pattern), so the
pooled columns keep those prefixes. Point-pattern radii inside a window are
10, 20 and 50 µm. Mean pooling is the column-wise mean. Attention pooling is a
weighted mean of the raw window vectors; the weights come from gated attention,
not from the hidden state.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data.dataset import RegionData
from benchmark.features.basic_feats import CompositionFeaturizer
from benchmark.features.density_feats import CellTypeDensityFeaturizer
from benchmark.features.mixing import MixingFeaturizer
from benchmark.features.point_pattern import PointPatternFeaturizer

WINDOW_SOURCES = ("composition", "density", "mixing", "point-pattern")
WINDOW_SIZE_UM = 100.0
WINDOW_STEP_UM = 50.0
MIN_CELLS = 10
POINT_PATTERN_RADII = (10, 20, 50)


def make_featurizers(cell_type_col: str) -> dict:
    """Fit-ready extractors. Call ``fit`` on the labelled regions first."""
    return {
        "composition": CompositionFeaturizer(cell_type_col=cell_type_col),
        "density": CellTypeDensityFeaturizer(cell_type_col=cell_type_col),
        "mixing": MixingFeaturizer(cell_type_col=cell_type_col, use_tissue_mask=True),
        "point-pattern": PointPatternFeaturizer(
            cell_type_col=cell_type_col,
            radii=list(POINT_PATTERN_RADII),
            metrics=("K", "L"),
            by_type=True,
            use_tissue_mask=True,
        ),
    }


def feature_names(featurizers: dict) -> list[str]:
    """Global-style ``source::column`` names, in extractor order."""
    names: list[str] = []
    for source, feat in featurizers.items():
        names.extend(f"{source}::{name}" for name in _bare_names(source, feat))
    return names


def columns_for_source(columns: list[str], source: str) -> list[str]:
    prefix = f"{source}::"
    return [col for col in columns if col.startswith(prefix)]


def mean_pool(bag: np.ndarray) -> np.ndarray:
    """Column mean across windows. Empty columns stay NaN."""
    bag = np.asarray(bag, dtype=float)
    if bag.ndim != 2 or bag.shape[0] == 0:
        raise ValueError("mean pool needs one or more window rows")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(bag, axis=0)


def attention_weighted_mean(bag: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of raw window features. NaN entries do not take weight."""
    bag = np.asarray(bag, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if bag.ndim != 2 or weights.shape[0] != bag.shape[0]:
        raise ValueError(
            f"attention weights {weights.shape} do not match bag {bag.shape}"
        )
    finite = np.isfinite(bag)
    safe = np.where(finite, bag, 0.0)
    weighted = weights[:, None] * finite
    denom = weighted.sum(axis=0)
    numer = (weights[:, None] * safe).sum(axis=0)
    out = np.full(bag.shape[1], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = numer[ok] / denom[ok]
    return out


def mean_pool_table(bags: dict[str, np.ndarray], columns: list[str]) -> pd.DataFrame:
    rows = []
    ids = []
    for region_id, bag in bags.items():
        if bag is None or len(bag) == 0:
            continue
        rows.append(mean_pool(bag))
        ids.append(str(region_id))
    table = pd.DataFrame(rows, index=ids, columns=columns)
    table.index.name = "region_id"
    return table


def save_bags(
    path: Path,
    columns: list[str],
    bags: dict[str, np.ndarray],
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": list(columns),
        "meta": dict(meta or {}),
        "bags": {str(region_id): np.asarray(bag, dtype=np.float32) for region_id, bag in bags.items()},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def load_bags(path: Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    bags = {str(region_id): np.asarray(bag, dtype=float) for region_id, bag in payload["bags"].items()}
    return list(payload["columns"]), bags, dict(payload.get("meta") or {})


def extract_bags(
    regions: list[RegionData],
    featurizers: dict,
    window_size_um: float = WINDOW_SIZE_UM,
    step_um: float = WINDOW_STEP_UM,
    min_cells: int = MIN_CELLS,
) -> dict[str, np.ndarray]:
    """One ``(n_windows, n_features)`` bag per region. Windows with too few cells are dropped."""
    names = feature_names(featurizers)
    bags: dict[str, np.ndarray] = {}
    for region in regions:
        bags[str(region.region_id)] = _region_bag(
            region, featurizers, names, window_size_um, step_um, min_cells,
        )
    return bags


def _bare_names(source: str, feat) -> list[str]:
    if source == "composition":
        return list(feat.cell_types_)
    if source == "point-pattern":
        return [
            f"{cell_type}_{metric}_r{radius}"
            for cell_type in feat.cell_types_
            for metric in feat.metrics
            for radius in feat.radii
        ]
    return list(feat.feature_names())


def _region_bag(region, featurizers, names, window_size_um, step_um, min_cells) -> np.ndarray:
    scaled = _to_microns(region)
    width = len(names)
    rows = []
    for bounds in _window_bounds(scaled, window_size_um, step_um):
        cell_ids = _cells_in_window(scaled, bounds)
        if len(cell_ids) < min_cells:
            continue
        window = scaled.subset(
            cell_ids,
            region_id=f"{scaled.region_id}___window",
            polygons=_clip_polygons(scaled.polygons, bounds),
        )
        values = _window_vector(window, featurizers)
        if len(values) != width:
            raise RuntimeError(f"window width {len(values)} != {width}")
        rows.append(values)
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _window_vector(window: RegionData, featurizers: dict) -> list[float]:
    values: list[float] = []
    for source, feat in featurizers.items():
        extracted = feat.extract_region(window)
        for name in _bare_names(source, feat):
            values.append(float(extracted.get(name, np.nan)))
    return values


def _to_microns(region: RegionData) -> RegionData:
    """Copy the region so window radii and square size are both in microns."""
    scale = float(region.microns_per_pixel)
    if scale == 1.0:
        return region
    coordinates = region.coordinates.copy()
    coordinates[["x", "y"]] = coordinates[["x", "y"]].to_numpy(float) * scale
    polygons = None
    if region.polygons:
        from shapely import affinity
        polygons = {
            kind: affinity.scale(geom, xfact=scale, yfact=scale, origin=(0.0, 0.0))
            for kind, geom in region.polygons.items()
            if geom is not None
        }
    return RegionData(
        region_id=region.region_id,
        coordinates=coordinates,
        expression=region.expression,
        cell_types=region.cell_types,
        microns_per_pixel=1.0,
        polygons=polygons,
    )


def _window_bounds(region: RegionData, window_size_um: float, step_um: float):
    xy = region.coordinates[["x", "y"]].to_numpy(float) * float(region.microns_per_pixel)
    if len(xy) == 0:
        return
    margin = window_size_um * 0.1
    xmin, ymin = xy.min(axis=0) - margin
    xmax, ymax = xy.max(axis=0) + margin
    x = float(xmin)
    while x < xmax:
        y = float(ymin)
        while y < ymax:
            yield (x, y, x + window_size_um, y + window_size_um)
            y += step_um
        x += step_um


def _cells_in_window(region: RegionData, bounds) -> list:
    xmin, ymin, xmax, ymax = bounds
    xy = region.coordinates[["x", "y"]].to_numpy(float) * float(region.microns_per_pixel)
    inside = (
        (xy[:, 0] >= xmin) & (xy[:, 0] < xmax)
        & (xy[:, 1] >= ymin) & (xy[:, 1] < ymax)
    )
    return region.coordinates.index[inside].tolist()


def _clip_polygons(polygons, bounds):
    if not polygons:
        return None
    from shapely.geometry import box
    square = box(*bounds)
    clipped = {}
    for kind, geom in polygons.items():
        if geom is None or geom.is_empty:
            continue
        hit = geom.intersection(square)
        if not hit.is_empty:
            clipped[kind] = hit
    return clipped or None
