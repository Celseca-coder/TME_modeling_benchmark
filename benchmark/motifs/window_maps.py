"""Per-task spatial maps of MIL windows for one region.

CSV schema (long):
  region_id, window_id, cx_um, cy_um, n_cells, motif_id, score, pos, neg, label
Optional: attention  (same keys; merge before plotting)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.features.window_grid import DEFAULT_STEP_UM, DEFAULT_WINDOW_SIZE_UM
from benchmark.motifs.spec import MotifSpec

_SET_SHORT = {
    "tumor": "tumor",
    "cd8": "CD8",
    "t_cell": "T",
    "macrophage": "Mac",
    "apc": "APC",
    "immune": "immune",
    "stroma": "stroma",
    "b_cell": "B",
    "vessel": "vessel",
    "granulocyte": "gran",
}


def score_used_quantity(spec: MotifSpec) -> str:
    """Cell quantity that score_used is computed from (colorbar annotation)."""
    def name(key: str | None, pct: bool = False) -> str:
        short = _SET_SHORT.get(str(key), str(key or "?"))
        return f"{short}%" if pct else short

    if spec.kind == "composition":
        core = name(spec.cell_set, pct=True)
    elif spec.kind == "neighbor_fraction":
        core = f"{name(spec.source_set)}→{name(spec.neighbor_set)} mix"
    elif spec.kind == "immune_exclusion":
        core = f"{name(spec.immune_set)} outside tumor"
    elif spec.kind == "interface":
        core = f"{name(spec.immune_set)} near tumor boundary"
    elif spec.kind == "tls_like":
        sets = "+".join(name(s) for s in spec.required_sets)
        core = f"{sets} large-cluster frac"
    else:
        core = spec.kind
    if spec.residual_on:
        cov = ", ".join(name(s, pct=True) for s in spec.residual_on)
        return f"{core}\nresidual on {cov}"
    return core


def quantity_labels_from_catalog(catalog) -> dict[str, str]:
    return {spec.id: score_used_quantity(spec) for spec in catalog.motifs}


def _as_grid(sub: pd.DataFrame, value_col: str, step_um: float = DEFAULT_STEP_UM):
    if sub.empty:
        raise ValueError(f"no windows to grid for column {value_col!r}")
    xs = np.sort(sub["cx_um"].unique())
    ys = np.sort(sub["cy_um"].unique())
    xi = {v: i for i, v in enumerate(xs)}
    yj = {v: j for j, v in enumerate(ys)}
    z = np.full((len(ys), len(xs)), np.nan)
    values = sub[value_col].to_numpy(dtype=float)
    for cx, cy, val in zip(sub["cx_um"].to_numpy(), sub["cy_um"].to_numpy(), values):
        z[yj[cy], xi[cx]] = val
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else float(step_um)
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else float(step_um)
    extent = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)
    return z, extent, dx, dy


def _draw_label_cells(ax, hit: pd.DataFrame, dx: float, dy: float, edgecolor: str, linewidth: float = 1.05) -> None:
    """Outline the unique step-sized heatmap cell (non-overlapping)."""
    from matplotlib.patches import Rectangle

    hx, hy = 0.5 * float(dx), 0.5 * float(dy)
    for cx, cy in zip(hit["cx_um"].to_numpy(float), hit["cy_um"].to_numpy(float)):
        ax.add_patch(Rectangle(
            (cx - hx, cy - hy), float(dx), float(dy),
            fill=False, edgecolor=edgecolor, linewidth=linewidth, zorder=3,
        ))


def plot_region_tasks(
    df: pd.DataFrame,
    region_id: str,
    *,
    color: str = "score",
    label_col: str = "label",
    motifs: list[str] | None = None,
    cmap: str = "YlOrRd",
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    quantity_labels: dict[str, str] | None = None,
    per_task_scale: bool = True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    bag = df[df["region_id"].astype(str) == str(region_id)].copy()
    if bag.empty:
        raise ValueError(f"region_id={region_id!r} not in table")
    if color not in bag.columns:
        raise ValueError(f"missing column {color!r}; available={list(bag.columns)}")

    ids = motifs or list(pd.unique(bag["motif_id"]))
    if not ids:
        raise ValueError(f"no motifs for region_id={region_id!r}")
    n = len(ids)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols + 0.4, 4.2 * nrows + 0.6),
        squeeze=False,
    )
    shared_norm = None
    if not per_task_scale:
        vmin = np.nanpercentile(bag[color].astype(float), 2)
        vmax = np.nanpercentile(bag[color].astype(float), 98)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        shared_norm = Normalize(vmin=vmin, vmax=vmax)

    half = window_size_um / 2.0
    for k, mid in enumerate(ids):
        ax = axes[k // ncols, k % ncols]
        row, col = divmod(k, ncols)
        sub = bag[bag["motif_id"] == mid]
        if sub.empty:
            ax.set_title(f"{mid}\nno windows", fontsize=9)
            ax.set_axis_off()
            continue
        z, extent, _dx, _dy = _as_grid(sub, color, step_um=step_um)
        if shared_norm is None:
            finite = z[np.isfinite(z)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [2, 98])
                if vmin == vmax:
                    vmin, vmax = float(finite.min()), float(finite.max() + 1e-6)
            else:
                vmin, vmax = 0.0, 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = shared_norm
        im = ax.imshow(
            z,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        pos = sub[sub[label_col] == 1]
        for cx, cy in zip(pos["cx_um"].to_numpy(), pos["cy_um"].to_numpy()):
            ax.add_patch(
                Rectangle(
                    (cx - half, cy - half),
                    window_size_um,
                    window_size_um,
                    fill=False,
                    edgecolor="k",
                    linewidth=0.8,
                    zorder=3,
                )
            )
        qty = (quantity_labels or {}).get(str(mid), color)
        qty_main, *qty_extra = str(qty).split("\n")
        extra = f"\n{qty_extra[0]}" if qty_extra else ""
        ax.set_title(f"{mid}   label=1: {int(len(pos))}{extra}", fontsize=8.5)
        if row == nrows - 1 or k + ncols >= n:
            ax.set_xlabel("x (µm)")
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("y (µm)")
        else:
            ax.set_ylabel("")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cbar.set_label(qty_main, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"{region_id}   color={color}   box=label==1", fontsize=12)
    fig.tight_layout()
    return fig


TYPE_PANEL_COLORS = {
    "tumor": "#d62728",
    "cd8": "#1f77b4",
    "macrophage": "#e377c2",
    "stroma": "#2ca02c",
    "t_cell": "#17becf",
    "immune": "#9467bd",
}

CELL_SIZE_UM = 10.0
CELL_SQUARE_COLOR = "#00C853"


def _shared_extent(bag: pd.DataFrame, step_um: float) -> tuple[float, float, float, float]:
    xs = np.sort(bag["cx_um"].astype(float).unique())
    ys = np.sort(bag["cy_um"].astype(float).unique())
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else float(step_um)
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else float(step_um)
    return (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2)


def _draw_cell_squares(ax, xy_um: np.ndarray, size_um: float, rng: np.random.Generator, n: int = 28) -> None:
    if len(xy_um) == 0:
        return
    take = min(int(n), len(xy_um))
    idx = rng.choice(len(xy_um), size=take, replace=False)
    half = 0.5 * float(size_um)
    from matplotlib.patches import Rectangle
    for x, y in xy_um[idx]:
        ax.add_patch(Rectangle(
            (x - half, y - half), size_um, size_um,
            fill=False, edgecolor=CELL_SQUARE_COLOR, linewidth=0.9, zorder=4,
        ))


def plot_region_pos_neg_with_types(
    df: pd.DataFrame,
    region,
    catalog,
    region_id: str,
    *,
    color: str = "score_used",
    label_col: str = "label",
    motifs: list[str] | None = None,
    type_sets: tuple[str, ...] = ("tumor", "cd8", "macrophage", "stroma"),
    cmap: str = "YlOrRd",
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    cell_size_um: float = CELL_SIZE_UM,
    seed: int = 0,
):
    """Cell-type maps plus score_used heatmaps with pos (black) and neg (blue) step cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from benchmark.motifs.detect import aligned_labels, coordinates_um, membership

    bag = df[df["region_id"].astype(str) == str(region_id)].copy()
    if bag.empty:
        raise ValueError(f"region_id={region_id!r} not in table")
    ids = motifs or list(pd.unique(bag["motif_id"]))
    qty = quantity_labels_from_catalog(catalog)
    xy = coordinates_um(region)
    labels = aligned_labels(region, catalog.cell_type_col)
    extent = _shared_extent(bag, step_um)
    rng = np.random.default_rng(seed)

    n_type = len(type_sets)
    n_score = len(ids)
    ncols = 4
    type_rows = int(np.ceil(n_type / ncols))
    score_rows = int(np.ceil(n_score / ncols))
    nrows = type_rows + 2 * score_rows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols + 0.3, 3.9 * nrows + 0.8),
        squeeze=False,
    )

    for k, name in enumerate(type_sets):
        ax = axes[k // ncols, k % ncols]
        if name in catalog.cell_sets:
            mask = membership(labels, catalog.resolve_set(name))
        else:
            mask = np.zeros(len(xy), dtype=bool)
        other = ~mask
        if other.any():
            ax.scatter(xy[other, 0], xy[other, 1], s=4, c="#d0d0d0", linewidths=0, rasterized=True)
        if mask.any():
            ax.scatter(
                xy[mask, 0], xy[mask, 1], s=8,
                c=TYPE_PANEL_COLORS.get(name, "#333333"),
                linewidths=0, rasterized=True, label=name,
            )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(f"cells: {name}  (n={int(mask.sum())})", fontsize=9)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)" if k % ncols == 0 else "")
    for k in range(n_type, type_rows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    def _score_block(row0: int, which: str) -> None:
        edge = "k" if which == "pos" else "#1565C0"
        target = 1.0 if which == "pos" else 0.0
        tag = "pos label=1" if which == "pos" else "neg label=0"
        for k, mid in enumerate(ids):
            ax = axes[row0 + k // ncols, k % ncols]
            sub = bag[bag["motif_id"] == mid]
            if sub.empty:
                ax.set_title(f"{mid}\nno windows", fontsize=8)
                ax.set_axis_off()
                continue
            z, ext, dx, dy = _as_grid(sub, color, step_um=step_um)
            finite = z[np.isfinite(z)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [2, 98])
                if vmin == vmax:
                    vmin, vmax = float(finite.min()), float(finite.max() + 1e-6)
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(
                z, origin="lower", extent=ext, cmap=cmap,
                norm=Normalize(vmin=vmin, vmax=vmax),
                interpolation="nearest", aspect="equal",
            )
            hit = sub[sub[label_col] == target]
            _draw_label_cells(ax, hit, dx, dy, edge)
            _draw_cell_squares(ax, xy, cell_size_um, rng)
            q = qty.get(str(mid), color).split("\n")[0]
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(f"{mid}   {tag}: {int(len(hit))}\n{q}", fontsize=8)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)" if k % ncols == 0 else "")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(q, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        leftover = score_rows * ncols - n_score
        for k in range(leftover):
            axes[row0 + score_rows - 1, ncols - 1 - k].axis("off")

    _score_block(type_rows, "pos")
    _score_block(type_rows + score_rows, "neg")
    fig.suptitle(
        f"{region_id}   heat={color}   black cell=window+ ({step_um:.0f} µm, no overlap)   "
        f"blue cell=window− ({step_um:.0f} µm, no overlap)   lime square=cell (~{cell_size_um:.0f} µm)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


TNBC_MARKER_PANELS = (
    ("DNA2", "DNA2  (nuclear / DAPI analog)"),
    ("panCK", "panCK  (tumor)"),
    ("CD8", "CD8"),
    ("CD68", "CD68  (macrophage)"),
)


def _marker_values(region, marker: str) -> np.ndarray:
    expr = region.expression
    if marker in expr.columns:
        return pd.to_numeric(expr[marker], errors="coerce").to_numpy(float)
    raise KeyError(f"marker {marker!r} not in expression columns {list(expr.columns)}")


def plot_region_tasks_with_markers(
    df: pd.DataFrame,
    region,
    region_id: str,
    *,
    color: str = "score_used",
    label_col: str = "label",
    motifs: list[str] | None = None,
    markers: tuple[tuple[str, str], ...] = TNBC_MARKER_PANELS,
    cmap_score: str = "YlOrRd",
    cmap_marker: str = "inferno",
    window_size_um: float = DEFAULT_WINDOW_SIZE_UM,
    step_um: float = DEFAULT_STEP_UM,
    per_task_scale: bool = True,
):
    """Window score_used maps plus raw marker scatter on the same spatial extent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    from benchmark.motifs.detect import coordinates_um

    bag = df[df["region_id"].astype(str) == str(region_id)].copy()
    if bag.empty:
        raise ValueError(f"region_id={region_id!r} not in table")
    if color not in bag.columns:
        raise ValueError(f"missing column {color!r}; available={list(bag.columns)}")

    ids = motifs or list(pd.unique(bag["motif_id"]))
    n_mark = len(markers)
    n_score = len(ids)
    ncols = 4
    n_mark_rows = int(np.ceil(n_mark / ncols))
    n_score_rows = int(np.ceil(n_score / ncols))
    nrows = n_mark_rows + n_score_rows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.3 * ncols + 0.4, 4.1 * nrows + 0.8),
        squeeze=False,
    )
    xy = coordinates_um(region)
    xs = bag["cx_um"].to_numpy(float)
    ys = bag["cy_um"].to_numpy(float)
    dx = float(np.median(np.diff(np.sort(np.unique(xs))))) if len(np.unique(xs)) > 1 else float(step_um)
    dy = float(np.median(np.diff(np.sort(np.unique(ys))))) if len(np.unique(ys)) > 1 else float(step_um)
    extent = (
        float(np.nanmin(xs)) - dx / 2,
        float(np.nanmax(xs)) + dx / 2,
        float(np.nanmin(ys)) - dy / 2,
        float(np.nanmax(ys)) + dy / 2,
    )
    half = window_size_um / 2.0

    win_ref = bag.drop_duplicates(["cx_um", "cy_um"])[["cx_um", "cy_um", "window_id"]].copy()
    insides = []
    for cx, cy in zip(win_ref["cx_um"].to_numpy(), win_ref["cy_um"].to_numpy()):
        insides.append(
            (xy[:, 0] >= cx - half) & (xy[:, 0] < cx + half)
            & (xy[:, 1] >= cy - half) & (xy[:, 1] < cy + half)
        )

    for k, (marker, title) in enumerate(markers):
        ax = axes[k // ncols, k % ncols]
        vals = _marker_values(region, marker)
        rows = []
        for (cx, cy), inside in zip(zip(win_ref["cx_um"], win_ref["cy_um"]), insides):
            hit = vals[inside]
            rows.append({"cx_um": cx, "cy_um": cy, "value": float(np.nanmean(hit)) if hit.size else np.nan})
        mark_df = pd.DataFrame(rows)
        z, ext, _dx, _dy = _as_grid(mark_df, "value", step_um=step_um)
        finite = z[np.isfinite(z)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [2, 98])
            if vmin == vmax:
                vmin, vmax = float(finite.min()), float(finite.max() + 1e-12)
        else:
            vmin, vmax = 0.0, 1.0
        im_m = ax.imshow(
            z, origin="lower", extent=ext, cmap=cmap_marker,
            vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (µm)" if k // ncols == n_mark_rows - 1 else "")
        ax.set_ylabel("y (µm)" if k % ncols == 0 else "")
        fig.colorbar(im_m, ax=ax, fraction=0.046, pad=0.04, label="window-mean intensity")

    for k in range(n_mark, n_mark_rows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    shared_norm = None
    if not per_task_scale:
        vmin = np.nanpercentile(bag[color].astype(float), 2)
        vmax = np.nanpercentile(bag[color].astype(float), 98)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        shared_norm = Normalize(vmin=vmin, vmax=vmax)

    for k, mid in enumerate(ids):
        ax = axes[n_mark_rows + k // ncols, k % ncols]
        sub = bag[bag["motif_id"] == mid]
        if sub.empty:
            ax.set_title(f"{mid}\nno windows", fontsize=9)
            ax.set_axis_off()
            continue
        z, ext, _dx, _dy = _as_grid(sub, color, step_um=step_um)
        if shared_norm is None:
            finite = z[np.isfinite(z)]
            if finite.size:
                vmin, vmax = np.nanpercentile(finite, [2, 98])
                if vmin == vmax:
                    vmin, vmax = float(finite.min()), float(finite.max() + 1e-6)
            else:
                vmin, vmax = 0.0, 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = shared_norm
        im = ax.imshow(
            z, origin="lower", extent=ext, cmap=cmap_score, norm=norm,
            interpolation="nearest", aspect="equal",
        )
        pos = sub[sub[label_col] == 1]
        for cx, cy in zip(pos["cx_um"].to_numpy(), pos["cy_um"].to_numpy()):
            ax.add_patch(Rectangle(
                (cx - half, cy - half), window_size_um, window_size_um,
                fill=False, edgecolor="k", linewidth=0.7, zorder=3,
            ))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{mid}   label=1: {int(len(pos))}", fontsize=9)
        last_score_row = n_mark_rows + (n_score - 1) // ncols
        if n_mark_rows + k // ncols == last_score_row:
            ax.set_xlabel("x (µm)")
        if k % ncols == 0:
            ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=color)

    leftover = n_score_rows * ncols - n_score
    for k in range(leftover):
        axes[-1, ncols - 1 - k].axis("off")

    fig.suptitle(
        f"{region_id}   heat={color} (pos/neg cuts)   box=label==1   "
        "top: raw IMC markers",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def save_region_tasks(
    df: pd.DataFrame,
    region_id: str,
    output: str | Path,
    **kwargs,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_region_tasks(df, region_id, **kwargs)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output
