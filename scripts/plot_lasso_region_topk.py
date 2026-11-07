#!/usr/bin/env python
"""Score one region with the same patient-fold L1 MIL as the top-20 panel, and plot it.

Trains only the validation fold that holds ``--region``, on the region label.
The window score is the out-of-fold p_w. The figure marks labeled windows in
that region's top 20%: positive (y_w=1) and negative (y_w=0).

    python scripts/plot_lasso_region_topk.py \
        --dataset bc_jackson2020 \
        --region BaselTMA_SP41_14_X1Y8 \
        --motif cd8_tumor_contact
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import (  # noqa: E402
    FAMILY_COLUMN_PREFIXES,
    MIXING_WINDOW_DROPPED,
    combo_name,
)
from benchmark.models.l1_mil import L1MILModel  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    TOPK_FRAC,
    bags_for_regions,
    feature_columns,
    join_features_and_labels,
    patient_region_folds,
    region_label,
    topk_labeled_positions,
)

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
FEATURE_DIR = Path("results/pseudo_labels/mil_windows/features")
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
META_COLS = {"region_id", "window_id", "cx_um", "cy_um"}


def _usecols(families: tuple[str, ...], columns: list[str]) -> list[str]:
    prefixes = [p for fam in families for p in FAMILY_COLUMN_PREFIXES[fam]]
    dropped = set(MIXING_WINDOW_DROPPED)
    keep = []
    for col in columns:
        if col in dropped:
            continue
        if col in META_COLS or any(col == p or col.startswith(p) for p in prefixes):
            keep.append(col)
    return keep


def score_region(
    table: pd.DataFrame,
    region_id: str,
    families: tuple[str, ...],
    y_r: pd.Series,
    *,
    n_folds: int = 5,
    seed: int = 0,
    frac: float = TOPK_FRAC,
) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold L1 MIL scores for one region. Same split and model as the panel."""
    region_id = str(region_id)
    feat_cols = feature_columns(table, families=families)
    held = None
    for k, (tr_regs, va_regs) in enumerate(patient_region_folds(table, y_r, n_folds, seed)):
        if region_id in set(map(str, va_regs)):
            held = (k, set(map(str, tr_regs)))
            break
    if held is None:
        raise ValueError(f"region {region_id!r} is not in any validation fold")
    fold, tr_regs = held
    print(f"region {region_id} is in validation fold {fold}; fitting L1 MIL", flush=True)
    tr = bags_for_regions(table, feat_cols, tr_regs)
    ytr = y_r.reindex(tr.index)
    keep = ytr.isin([0.0, 1.0])
    tr, ytr = tr.loc[keep], ytr.loc[keep]
    if tr.empty or ytr.nunique() < 2:
        raise ValueError(f"fold {fold} has no both-class region labels")
    model = L1MILModel(seed=seed + fold, max_instances=None).fit(tr[["bag"]], ytr.astype(int))
    part = table[table["region_id"] == region_id].sort_values("window_id").copy()
    part["p_w"] = model.window_probabilities(part[feat_cols].to_numpy(np.float32))
    top = topk_labeled_positions(part["y_w"].to_numpy(float), part["p_w"].to_numpy(float), frac)
    in_top = np.zeros(len(part), dtype=bool)
    in_top[top] = True
    part["in_top20"] = in_top
    labeled = part["y_w"].isin([0.0, 1.0])
    chosen = part["in_top20"]
    n_top = int(chosen.sum())
    n_pos = int((chosen & (part["y_w"] == 1.0)).sum())
    info = {
        "fold": int(fold),
        "n_train": int(len(tr)),
        "n_windows": int(len(part)),
        "n_labeled": int(labeled.sum()),
        "n_top": n_top,
        "n_top_pos": n_pos,
        "n_top_neg": int((chosen & (part["y_w"] == 0.0)).sum()),
        "precision_at_20": (n_pos / n_top) if n_top else float("nan"),
        "labeled_prev": float((part.loc[labeled, "y_w"] == 1.0).mean()) if labeled.any() else float("nan"),
    }
    return part, info


def plot_region_topk(part: pd.DataFrame, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    rest = part.loc[~part["in_top20"]]
    pos = part.loc[part["in_top20"] & (part["y_w"] == 1.0)]
    neg = part.loc[part["in_top20"] & (part["y_w"] == 0.0)]
    ax.scatter(rest["cx_um"], rest["cy_um"], s=18, c="#d0d0d0", linewidths=0, label="other windows")
    ax.scatter(
        pos["cx_um"], pos["cy_um"], s=46, c="#d62728", linewidths=0,
        label=f"top 20% positive (n={len(pos)})", zorder=3,
    )
    ax.scatter(
        neg["cx_um"], neg["cy_um"], s=42, c="#1565C0", marker="x", linewidths=1.2,
        label=f"top 20% negative (n={len(neg)})", zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="bc_jackson2020")
    p.add_argument("--region", default="BaselTMA_SP41_14_X1Y8")
    p.add_argument("--motif", default="cd8_tumor_contact")
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    p.add_argument("--frac", type=float, default=TOPK_FRAC)
    p.add_argument("--feature-dir", default=str(FEATURE_DIR))
    p.add_argument("--out-dir", default="results/window_extractor/maps")
    args = p.parse_args()

    families = tuple(args.families)
    group = combo_name(families)
    feat_path = Path(args.feature_dir) / f"{args.dataset}_panel.csv"
    header = pd.read_csv(feat_path, nrows=0).columns.tolist()
    feats = pd.read_csv(feat_path, usecols=_usecols(families, header))
    wins = pd.read_csv(WINDOW_DIR / f"{args.dataset}_windows_noisy.csv")
    bags = pd.read_csv(BAG_DIR / f"{args.dataset}_v2_noisy.csv")
    tab = join_features_and_labels(feats, wins, args.motif, bags)
    print(
        f"joined {args.dataset} {args.motif} windows={len(tab)} group={group}",
        flush=True,
    )
    part, info = score_region(
        tab, args.region, families, region_label(bags, args.motif), frac=args.frac,
    )
    keep = ["region_id", "window_id", "cx_um", "cy_um", "y_w", "p_w", "in_top20"]
    scored = part[keep].sort_values("window_id")

    dest = Path(args.out_dir) / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{args.region}_{args.motif}_{group}_topk20"
    csv_path = dest / f"{stem}.csv"
    png_path = dest / f"{stem}.png"
    scored.to_csv(csv_path, index=False)
    title = (
        f"{args.dataset}  {args.region}  {args.motif}\n"
        f"{group}   fold {info['fold']}   "
        f"top {info['n_top']}: +{info['n_top_pos']} / −{info['n_top_neg']}   "
        f"precision@20={info['precision_at_20']:.3f}"
    )
    fig = plot_region_topk(scored, title)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print({**info, "feature_group": group, "csv": str(csv_path), "png": str(png_path)}, flush=True)


if __name__ == "__main__":
    main()
