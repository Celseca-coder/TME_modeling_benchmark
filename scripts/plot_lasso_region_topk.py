#!/usr/bin/env python
"""Score one region with the same patient-fold L1 as the top-20 panel, and plot it.

Trains only the validation fold that holds ``--region``, so the window score
is the out-of-fold p_w used by precision@20. The figure marks labeled windows
in that region's top 20% : positive (y_w=1) and negative (y_w=0).

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
    combo_name,
)
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    TOPK_FRAC,
    _positive_proba,
    feature_columns,
    join_features_and_labels,
    topk_labeled_positions,
)
from benchmark.validation.splits import safe_patient_kfold  # noqa: E402

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
FEATURE_DIR = Path("results/pseudo_labels/mil_windows/features")
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
META_COLS = {"region_id", "window_id", "cx_um", "cy_um"}


def _usecols(families: tuple[str, ...], columns: list[str]) -> list[str]:
    prefixes = [p for fam in families for p in FAMILY_COLUMN_PREFIXES[fam]]
    keep = []
    for col in columns:
        if col in META_COLS or any(col == p or col.startswith(p) for p in prefixes):
            keep.append(col)
    return keep


def _folds(table: pd.DataFrame, n_folds: int, seed: int):
    meta = (
        table[["region_id", "patient_id", "y_w"]]
        .dropna(subset=["patient_id"])
        .drop_duplicates("region_id")
    )
    has_pos = table.groupby("region_id")["y_w"].apply(lambda s: float((s == 1).any()))
    meta = meta.set_index("region_id")
    meta["has_pos"] = has_pos.reindex(meta.index)
    meta = meta.reset_index()
    folds = safe_patient_kfold(meta, n_folds, "patient_id", stratify_col="has_pos", seed=seed)
    if not folds:
        raise ValueError("not enough patients for CV")
    return folds


def score_region(
    table: pd.DataFrame,
    region_id: str,
    families: tuple[str, ...],
    *,
    n_folds: int = 5,
    seed: int = 0,
    l1_ratio: float = 1.0,
    frac: float = TOPK_FRAC,
) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold L1 scores for one region. Same split and model as the panel."""
    region_id = str(region_id)
    feat_cols = feature_columns(table, families=families)
    held = None
    for k, (tr_regs, va_regs) in enumerate(_folds(table, n_folds, seed)):
        if region_id in set(map(str, va_regs)):
            held = (k, set(map(str, tr_regs)))
            break
    if held is None:
        raise ValueError(f"region {region_id!r} is not in any validation fold")
    fold, tr_regs = held
    print(f"region {region_id} is in validation fold {fold}; fitting L1", flush=True)
    tr = table[table["region_id"].isin(tr_regs) & table["y_w"].isin([0.0, 1.0])].copy()
    if tr.empty or tr["y_w"].nunique() < 2:
        raise ValueError(f"fold {fold} has no both-class training labels")
    model = LinearClassifier(seed=seed + fold, C=1.0, l1_ratio=l1_ratio).fit(
        tr[feat_cols].set_index(tr["row_id"]),
        tr.set_index("row_id")["y_w"].astype(float),
    )
    part = table[table["region_id"] == region_id].copy()
    p = _positive_proba(model, part[feat_cols].set_index(part["row_id"]))
    part["p_w"] = p
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
    p.add_argument("--l1-ratio", type=float, default=1.0)
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
    bags = pd.read_csv(
        BAG_DIR / f"{args.dataset}_v2_noisy.csv",
        usecols=lambda c: c in {"region_id", "patient_id"},
    )
    tab = join_features_and_labels(feats, wins, args.motif, bags)
    print(
        f"joined {args.dataset} {args.motif} windows={len(tab)} group={group}",
        flush=True,
    )
    part, info = score_region(
        tab, args.region, families, l1_ratio=args.l1_ratio, frac=args.frac,
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
