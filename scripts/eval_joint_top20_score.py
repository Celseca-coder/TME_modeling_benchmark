#!/usr/bin/env python
"""Mean score of the within-region top 20% windows, joint four-family panel.

The published table reports the positive-window rate inside that top 20%.
This script reports the scores themselves: for each region, the mean score of
its labeled windows in the top 20%, then the mean over regions and folds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import combo_name  # noqa: E402
from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES, selected_motifs_for  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    evaluate_table,
    join_features_and_labels,
)
from scripts.eval_ig_mil_window_importance import evaluate_gated_localisation  # noqa: E402
from scripts.eval_mlp_ig_window_importance import evaluate_mlp_ig  # noqa: E402
from scripts.eval_rf_mdi_window_importance import evaluate_rf_mdi  # noqa: E402

FAMILIES = ("composition", "density", "mixing", "point_pattern")
FEATURE_DIR = Path(
    "/autofs/nas8/tywang/tjzou/TME_modeling_benchmark/results/pseudo_labels/mil_windows/features"
)
WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
OUT_DIR = Path(
    "/autofs/nas8/tywang/tjzou/TME_modeling_benchmark/results/pseudo_labels/mil_windows"
)

METHOD_NAME = {
    "lasso": "Lasso",
    "shap_mil": "SHAP+Gated Attention",
    "ig": "IG",
    "attention": "attention",
    "mlp_ig": "MLP IG",
    "rf_mdi": "RF MDI",
}
SCORE_KIND = {
    "lasso": "mean positive-class probability",
    "shap_mil": "mean Shapley window score",
    "ig": "mean integrated-gradient window score",
    "attention": "mean softmax attention weight",
    "mlp_ig": "mean summed signed IG",
    "rf_mdi": "mean MDI window score",
}


def _mean_score(fold: pd.DataFrame) -> tuple[float, float]:
    if fold.empty or "score_top20_mean" not in fold.columns:
        return float("nan"), float("nan")
    return float(fold["score_top20_mean"].mean()), float(fold["precision_at_20"].mean())


def _row(dataset: str, motif: str, method: str, fold: pd.DataFrame) -> dict:
    score, prec = _mean_score(fold)
    return {
        "dataset": dataset,
        "motif": motif,
        "method": method,
        "method_name": METHOD_NAME[method],
        "score_kind": SCORE_KIND[method],
        "score_top20_mean": score,
        "precision_at_20": prec,
        "feature_group": combo_name(FAMILIES),
    }


def _load(dataset: str):
    feats = pd.read_csv(FEATURE_DIR / f"{dataset}_panel.csv")
    wins = pd.read_csv(WINDOW_DIR / f"{dataset}_windows_noisy.csv")
    bags = pd.read_csv(BAG_DIR / f"{dataset}_v2_noisy.csv")
    return feats, wins, bags


def run_lasso() -> list[dict]:
    rows = []
    by_ds: dict[str, list[str]] = {}
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        by_ds.setdefault(dataset, []).append(motif)
    for dataset, motifs in by_ds.items():
        feats, wins, bags = _load(dataset)
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold = evaluate_table(tab, families=FAMILIES, l1_ratio=1.0)
            rec = _row(dataset, motif, "lasso", fold)
            rows.append(rec)
            print(rec, flush=True)
    return rows


def run_gated() -> list[dict]:
    rows = []
    for dataset in dict.fromkeys(ds for ds, _ in SELECTED_MOTIF_SOURCES):
        feats, wins, bags = _load(dataset)
        y_index = bags.set_index(bags["region_id"].astype(str))
        for motif in selected_motifs_for(dataset):
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(y_index[f"{motif}_label_v2"], errors="coerce")
            folds = evaluate_gated_localisation(tab, y_r, FAMILIES)
            for method, fold in folds.items():
                rec = _row(dataset, motif, method, fold)
                rows.append(rec)
                print(rec, flush=True)
    return rows


def run_mlp() -> list[dict]:
    rows = []
    for dataset in dict.fromkeys(ds for ds, _ in SELECTED_MOTIF_SOURCES):
        feats, wins, bags = _load(dataset)
        for motif in selected_motifs_for(dataset):
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold, _ranking = evaluate_mlp_ig(tab, FAMILIES)
            rec = _row(dataset, motif, "mlp_ig", fold)
            rows.append(rec)
            print(rec, flush=True)
    return rows


def run_rf() -> list[dict]:
    rows = []
    for dataset in dict.fromkeys(ds for ds, _ in SELECTED_MOTIF_SOURCES):
        feats, wins, bags = _load(dataset)
        for motif in selected_motifs_for(dataset):
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold = evaluate_rf_mdi(tab, FAMILIES)
            rec = _row(dataset, motif, "rf_mdi", fold)
            rows.append(rec)
            print(rec, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lasso", "gated", "mlp", "rf"], required=True)
    args = parser.parse_args()
    runners = {"lasso": run_lasso, "gated": run_gated, "mlp": run_mlp, "rf": run_rf}
    rows = runners[args.method]()
    dest = OUT_DIR / f"joint_top20_score_{args.method}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
