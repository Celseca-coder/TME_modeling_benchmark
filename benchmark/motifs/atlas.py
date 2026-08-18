"""Phase-1 atlas: cheap composition scan + a few spatial maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data.dataset import TMEDataset
from benchmark.motifs.detect import aligned_labels, composition_fractions
from benchmark.motifs.spec import MotifCatalog

_UNIFORM_COLORS = {
    "Tumor": "#d62728",
    "Tumor (Proliferating)": "#ff7f0e",
    "CD8 T cell": "#1f77b4",
    "CD4 T cell": "#17becf",
    "B cell": "#9467bd",
    "APC/Dendritic cell": "#8c564b",
    "Macrophage": "#e377c2",
    "Granulocyte": "#bcbd22",
    "Naive immune cell": "#7f7f7f",
    "Stroma": "#2ca02c",
    "Vessel": "#aec7e8",
    "Other": "#c7c7c7",
}


def quick_region_composition(
    dataset: TMEDataset,
    region_id: str,
    catalog: MotifCatalog,
) -> dict[str, float]:
    """Read only cell_types.csv — no expression / spatial load."""
    path = dataset.region_dir(region_id) / "cell_types.csv"
    table = pd.read_csv(path)
    col = catalog.cell_type_col if catalog.cell_type_col in table.columns else "cell_type"
    labels = table[col]
    row = composition_fractions(labels, catalog)
    row["region_id"] = region_id
    return row


def composition_table(
    dataset: TMEDataset,
    catalog: MotifCatalog,
    region_ids: list[str] | None = None,
) -> pd.DataFrame:
    ids = list(region_ids or dataset.get_metadata()["region_id"].astype(str))
    rows = [quick_region_composition(dataset, rid, catalog) for rid in ids]
    return pd.DataFrame(rows)


def sample_atlas_regions(
    composition: pd.DataFrame,
    n: int = 20,
    seed: int = 0,
) -> list[str]:
    """Stratify by tumor fraction, then add CD8 / B extremes."""
    rng = np.random.default_rng(seed)
    table = composition.dropna(subset=["frac__tumor"]).copy()
    if table.empty:
        return []
    table["tumor_bin"] = pd.qcut(table["frac__tumor"], q=min(3, table["frac__tumor"].nunique()), duplicates="drop")
    picked: list[str] = []
    per_bin = max(2, n // max(1, table["tumor_bin"].nunique()))
    for _, grp in table.groupby("tumor_bin", observed=False):
        take = min(per_bin, len(grp))
        chosen = rng.choice(grp["region_id"].to_numpy(), size=take, replace=False)
        picked.extend(str(x) for x in chosen)

    for col in ("frac__cd8", "frac__b_cell", "frac__t_cell"):
        if col not in table.columns:
            continue
        ranked = table.sort_values(col, ascending=False)["region_id"].astype(str)
        if len(ranked):
            picked.append(ranked.iloc[0])
        if len(ranked) > 1:
            picked.append(ranked.iloc[-1])

    # stable unique, cap at n + a few extremes
    uniq: list[str] = []
    for rid in picked:
        if rid not in uniq:
            uniq.append(rid)
    return uniq[:n]


def plot_region_map(
    dataset: TMEDataset,
    region_id: str,
    catalog: MotifCatalog,
    output: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    region = dataset.load_region(region_id, normalize=False, use_cache=False)
    labels = aligned_labels(region, catalog.cell_type_col)
    xy = region.coordinates[["x", "y"]].to_numpy(float)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    unknown = sorted(set(labels.dropna().unique()) - set(_UNIFORM_COLORS))
    cmap = dict(_UNIFORM_COLORS)
    extras = plt.cm.tab20(np.linspace(0, 1, max(len(unknown), 1)))
    for name, color in zip(unknown, extras):
        cmap[str(name)] = color

    for name, color in cmap.items():
        mask = (labels == name).to_numpy()
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=2, c=[color], label=name, linewidths=0)

    if region.polygons and "tumour" in region.polygons:
        try:
            from shapely.plotting import plot_polygon
            plot_polygon(region.polygons["tumour"], ax=ax, add_points=False, facecolor="none", edgecolor="black")
        except Exception:
            pass

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or region_id, fontsize=10)
    ax.legend(markerscale=4, fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_findings_template(composition: pd.DataFrame, sampled_ids: list[str], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    keep = composition[composition["region_id"].astype(str).isin(sampled_ids)].copy()
    keep["pattern_id"] = ""
    keep["scale_um"] = 50
    keep["spatial_or_composition"] = ""
    keep["positive_example"] = ""
    keep["negative_example"] = ""
    keep["promote"] = ""
    keep.to_csv(output, index=False)
    return output
