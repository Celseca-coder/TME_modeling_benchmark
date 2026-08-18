"""Shared spatial geometry helpers for motif detectors."""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from benchmark.data.dataset import RegionData

TUMOR_POLYGON_KIND = "tumour"


def coordinates_um(region: RegionData) -> np.ndarray:
    """Return aligned x/y coordinates in microns."""
    return (
        region.coordinates[["x", "y"]].to_numpy(float)
        * float(region.microns_per_pixel)
    )


def radius_neighbors(xy_um: np.ndarray, radius_um: float) -> list[list[int]]:
    """Indices within a fixed radius, including each point itself."""
    if len(xy_um) == 0:
        return []
    return KDTree(xy_um).query_ball_point(xy_um, r=float(radius_um))


def inside_tumor(region: RegionData) -> np.ndarray | None:
    """Return per-cell tumour-polygon membership when available."""
    xy = region.coordinates[["x", "y"]].to_numpy(float)
    return region.polygon_contains(xy, TUMOR_POLYGON_KIND)


def distance_to_tumor_boundary_um(region: RegionData) -> np.ndarray | None:
    """Return each cell's distance to the tumour boundary in microns."""
    if region.polygons is None:
        return None
    polygon = region.polygons.get(TUMOR_POLYGON_KIND)
    if polygon is None or polygon.is_empty:
        return None

    import shapely

    xy = region.coordinates[["x", "y"]].to_numpy(float)
    points = shapely.points(xy[:, 0], xy[:, 1])
    distance_px = shapely.distance(points, polygon.boundary)
    return np.asarray(distance_px, float) * float(region.microns_per_pixel)
