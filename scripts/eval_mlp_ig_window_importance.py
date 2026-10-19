#!/usr/bin/env python
"""Train a window MLP on y_w and attribute features with Integrated Gradients.

Window score s_w is the sum of signed IG across features. Global importance
x_f is the mean absolute IG inside each validation fold. Hit@5 and Precision@5
use the mean rank of x_f across folds.

    python scripts/eval_mlp_ig_window_importance.py \
        --families composition density mixing point_pattern
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import combo_name
from benchmark.motifs.mlp_ig import (
    WindowMLP,
    global_feature_importance,
    integrated_gradients,
    mean_feature_rank,
    recovery_at_k,
)
from benchmark.motifs.source_select import CANDIDATE_DATASETS, selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    PASS_PRECISION_AT,
    PASS_WINDOW,
    feature_columns,
    join_features_and_labels,
    within_region_auroc,
    within_region_topk,
)
from benchmark.validation.splits import safe_patient_kfold

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
K = 5


def evaluate_mlp_ig(
    table: pd.DataFrame,
    families,
    steps: int = 32,
    seed: int = 0,
    epochs: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level CV. Returns fold metrics and the mean feature ranking."""
    feat_cols = feature_columns(table, families=families)
    if "patient_id" not in table.columns:
        raise ValueError("need patient_id for clustered CV")
    meta = (
        table[["region_id", "patient_id", "y_w"]]
        .dropna(subset=["patient_id"])
        .drop_duplicates("region_id")
    )
    has_pos = table.groupby("region_id")["y_w"].apply(lambda s: float((s == 1).any()))
    meta = meta.set_index("region_id")
    meta["has_pos"] = has_pos.reindex(meta.index)
    meta = meta.reset_index()
    folds = safe_patient_kfold(meta, 5, "patient_id", stratify_col="has_pos", seed=seed)
    if not folds:
        raise ValueError("not enough patients for CV")
    rows = []
    orders: list[list[str]] = []
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = table[table["region_id"].isin(set(map(str, tr_regs)))]
        va = table[table["region_id"].isin(set(map(str, va_regs)))].copy()
        labeled_tr = tr.loc[tr["y_w"].isin([0.0, 1.0])].copy()
        if labeled_tr.empty or labeled_tr["y_w"].nunique() < 2:
            continue
        model = WindowMLP(seed=seed + k, epochs=epochs).fit(
            labeled_tr[feat_cols].set_index(labeled_tr["row_id"]),
            labeled_tr.set_index("row_id")["y_w"].astype(float),
        )
        labeled_va = va.loc[va["y_w"].isin([0.0, 1.0])].copy()
        ig, scores_l = integrated_gradients(model, labeled_va[feat_cols], steps=steps)
        scores = pd.Series(np.nan, index=va.index, dtype=float)
        scores.loc[labeled_va.index] = scores_l
        loc = within_region_auroc(va["y_w"], scores, va["region_id"])
        loc.update(within_region_topk(va["y_w"], scores, va["region_id"]))
        loc["fold"] = k
        rows.append(loc)
        importance = global_feature_importance(ig)
        order = np.argsort(-importance)
        orders.append([model._columns[int(i)] for i in order])
    return pd.DataFrame(rows), mean_feature_rank(orders)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CANDIDATE_DATASETS))
    p.add_argument("--motif", default="")
    p.add_argument("--feature-dir", default="results/pseudo_labels/mil_windows/features")
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window-dir", default=str(WINDOW_DIR))
    p.add_argument("--bag-dir", default=str(BAG_DIR))
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/mlp_ig_localisation.csv")
    p.add_argument("--features-out", default="results/pseudo_labels/mil_windows/mlp_ig_features.csv")
    args = p.parse_args()

    families = tuple(args.families)
    group = combo_name(families)
    rows = []
    feature_rows = []
    for dataset in args.datasets:
        feats = pd.read_csv(Path(args.feature_dir) / f"{dataset}_panel.csv")
        wins = pd.read_csv(Path(args.window_dir) / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(Path(args.bag_dir) / f"{dataset}_v2_noisy.csv")
        motifs = [args.motif] if args.motif else list(selected_motifs_for(dataset))
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold, ranking = evaluate_mlp_ig(tab, families, steps=args.steps, epochs=args.epochs)
            if fold.empty:
                auroc = pwr = prev = float("nan")
                n_regions = n_folds = 0
            else:
                auroc = float(fold["auroc_window_mean"].mean())
                pwr = float(fold["precision_at_20"].mean())
                prev = float(fold["labeled_prev"].mean())
                n_regions = int(fold["n_regions_scored"].sum())
                n_folds = int(len(fold))
            ranked = ranking["feature"].tolist() if len(ranking) else []
            rec = recovery_at_k(ranked, motif, k=K) if ranked else {
                "hit_at_k": 0, "precision_at_k": float("nan"), "best_hit_rank": None,
                "top1": "", "miss_as_top1": False, "passed": False, "k": K,
            }
            summary = {
                "dataset": dataset,
                "motif": motif,
                "feature_group": group,
                "method": "mlp_ig",
                "n_windows": int(len(tab)),
                "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
                "auroc_window_mean": auroc,
                "precision_at_20": pwr,
                "labeled_prev": prev,
                "n_regions_scored": n_regions,
                "n_folds": n_folds,
                "pass_window": bool(auroc == auroc and auroc >= PASS_WINDOW),
                "pass_precision_at_20": bool(pwr == pwr and pwr >= PASS_PRECISION_AT),
                "hit_at_5": rec["hit_at_k"],
                "precision_at_5": rec["precision_at_k"],
                "best_hit_rank": rec["best_hit_rank"],
                "top1": rec["top1"],
                "miss_as_top1": rec["miss_as_top1"],
                "feature_passed": rec["passed"],
            }
            rows.append(summary)
            print(summary, flush=True)
            if len(ranking):
                top = ranking.head(15).copy()
                top.insert(0, "motif", motif)
                top.insert(0, "dataset", dataset)
                feature_rows.append(top)
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
            if feature_rows:
                pd.concat(feature_rows, ignore_index=True).to_csv(args.features_out, index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
