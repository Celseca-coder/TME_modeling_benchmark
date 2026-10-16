"""Square-window grid shared by AttnMIL bags and MIL-level motif GT."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.data.dataset import RegionData
from benchmark.motifs.detect import coordinates_um

DEFAULT_WINDOW_SIZE_UM = 100.0
DEFAULT_STEP_UM = 50.0
DEFAULT_MIN_CELLS = 1


@dataclass(frozen=True)
class Window:
    window_id: int
    center_um: np.ndarray  # shape (2,)
    origin_um: np.ndarray  # lower-left of the square
    inside: np.ndarray     # bool mask over region cells


def window_inside_mask(
    xy_um: np.ndarray,
    origin_um: np.ndarray,
    window_size_um: float,
) -> np.ndarray:
    """Cells in [origin, origin + size) — same half-open square as MIL bags."""
    x0, y0 = float(origin_um[0]), float(origin_um[1])
    size = float(window_size_um)
    return (
        (xy_um[:, 0] >= x0)
        & (xy_um[:, 0] < x0 + size)
        & (xy_um[:, 1] >= y0)
        & (xy_um[:, 1] < y0 + size)
    )


def iter_windows(
    region: RegionData,
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    min_cells: int = DEFAULT_MIN_CELLS,
) -> list[Window]:
    """Tile from each region's min x/y (same convention as patch / AttnMIL)."""
    xy = coordinates_um(region)
    if len(xy) == 0:
        return []
    size = float(window_size_um)
    step = float(step_um if step_um is not None else window_size_um)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)
    out: list[Window] = []
    wid = 0
    for x0 in xs:
        for y0 in ys:
            origin = np.asarray([x0, y0], dtype=float)
            inside = window_inside_mask(xy, origin, size)
            if int(inside.sum()) < min_cells:
                continue
            center = origin + 0.5 * size
            out.append(Window(window_id=wid, center_um=center, origin_um=origin, inside=inside))
            wid += 1
    return out


def window_centers_um(windows: list[Window]) -> np.ndarray:
    if not windows:
        return np.zeros((0, 2), dtype=float)
    return np.stack([w.center_um for w in windows], axis=0)