#!/usr/bin/env python
"""Train a random forest on y_w; rank windows by impurity-importance scores.

``s_w = sum_j I_j x_{w,j}``, where ``I`` is mean decrease in impurity.
Metrics stay inside each region: AUROC and precision among the top 20% scores.

    python scripts/eval_rf_mdi_window_importance.py \
        --families composition density mixing point_pattern
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import combo_name
from benchmark.models.forest import RandomForestModel
from benchmark.motifs.rf_mdi import mdi_window_scores
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


def evaluate_rf_mdi(
    table: pd.DataFrame,
    families,
    n_estimators: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Patient-level CV. Target is the window label y_w."""
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
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = table[table["region_id"].isin(set(map(str, tr_regs)))]
        va = table[table["region_id"].isin(set(map(str, va_regs)))].copy()
        labeled = tr.loc[tr["y_w"].isin([0.0, 1.0])].copy()
        if labeled.empty or labeled["y_w"].nunique() < 2:
            continue
        Xtr = labeled[feat_cols].set_index(labeled["row_id"])
        ytr = labeled.set_index("row_id")["y_w"].astype(float)
        model = RandomForestModel(seed=seed + k, n_estimators=n_estimators).fit(Xtr, ytr)
        scores = pd.Series(mdi_window_scores(model, va[feat_cols]), index=va.index)
        loc = within_region_auroc(va["y_w"], scores, va["region_id"])
        loc.update(within_region_topk(va["y_w"], scores, va["region_id"]))
        loc["fold"] = k
        rows.append(loc)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CANDIDATE_DATASETS))
    p.add_argument("--motif", default="")
    p.add_argument("--feature-dir", default="results/pseudo_labels/mil_windows/features")
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--window-dir", default=str(WINDOW_DIR))
    p.add_argument("--bag-dir", default=str(BAG_DIR))
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/rf_mdi_localisation.csv")
    args = p.parse_args()

    families = tuple(args.families)
    group = combo_name(families)
    rows = []
    for dataset in args.datasets:
        feats = pd.read_csv(Path(args.feature_dir) / f"{dataset}_panel.csv")
        wins = pd.read_csv(Path(args.window_dir) / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(Path(args.bag_dir) / f"{dataset}_v2_noisy.csv")
        motifs = [args.motif] if args.motif else list(selected_motifs_for(dataset))
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold = evaluate_rf_mdi(tab, families, n_estimators=args.n_estimators)
            if fold.empty:
                auroc = pwr = prev = float("nan")
                n_regions = n_folds = 0
            else:
                auroc = float(fold["auroc_window_mean"].mean())
                pwr = float(fold["precision_at_20"].mean())
                prev = float(fold["labeled_prev"].mean())
                n_regions = int(fold["n_regions_scored"].sum())
                n_folds = int(len(fold))
            summary = {
                "dataset": dataset,
                "motif": motif,
                "feature_group": group,
                "method": "rf_mdi",
                "n_windows": int(len(tab)),
                "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
                "auroc_window_mean": auroc,
                "precision_at_20": pwr,
                "labeled_prev": prev,
                "n_regions_scored": n_regions,
                "n_folds": n_folds,
                "pass_window": bool(auroc == auroc and auroc >= PASS_WINDOW),
                "pass_precision_at_20": bool(pwr == pwr and pwr >= PASS_PRECISION_AT),
            }
            rows.append(summary)
            print(summary, flush=True)
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
