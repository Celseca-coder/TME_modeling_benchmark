#!/usr/bin/env python
"""Within-region precision@20% for every selected task × family (L1 Lasso).

Ranks labeled windows by p_w; GT is unchanged (noisy y_w).

    python scripts/eval_window_extractor_precision_at20.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import WINDOW_PANEL_COMBOS, combo_name
from benchmark.motifs.source_select import CANDIDATE_DATASETS, selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    PASS_PRECISION_AT_20,
    evaluate_table,
    join_features_and_labels,
)

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
DEFAULT_FEATURE_DIR = Path("results/pseudo_labels/mil_windows/features")
DEFAULT_OUT = Path("results/pseudo_labels/mil_windows/extractor_precision_at20_l1.csv")

CONTROL_MOTIFS = ("tumor_high", "cd8_high")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CANDIDATE_DATASETS))
    p.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--l1-ratio", type=float, default=1.0)
    args = p.parse_args()

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if dest.exists():
        rows = pd.read_csv(dest).to_dict("records")
        done = {(r["dataset"], r["motif"], r["feature_group"]) for r in rows}
    else:
        done = set()

    for dataset in args.datasets:
        feats = pd.read_csv(Path(args.feature_dir) / f"{dataset}_panel.csv")
        wins = pd.read_csv(WINDOW_DIR / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(
            BAG_DIR / f"{dataset}_v2_noisy.csv",
            usecols=lambda c: c in {"region_id", "patient_id"},
        )
        for families in WINDOW_PANEL_COMBOS:
            group = combo_name(families)
            for motif in selected_motifs_for(dataset):
                key = (dataset, motif, group)
                if key in done:
                    print(f"skip {key}", flush=True)
                    continue
                tab = join_features_and_labels(feats, wins, motif, bags)
                fold = evaluate_table(tab, families=families, l1_ratio=args.l1_ratio)
                summary = {
                    "dataset": dataset,
                    "motif": motif,
                    "kind": "control" if motif in CONTROL_MOTIFS else "spatial",
                    "feature_group": group,
                    "n_windows": int(len(tab)),
                    "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
                    "precision_at_20_mean": float(fold["precision_at_20_mean"].mean()),
                    "prevalence_mean": float(fold["prevalence_mean"].mean()),
                    "n_regions_scored": int(fold["n_regions_scored"].sum()),
                    "n_folds": int(len(fold)),
                    "pass_threshold": PASS_PRECISION_AT_20,
                    "pass_precision_at_20": bool(
                        fold["precision_at_20_mean"].mean() >= PASS_PRECISION_AT_20
                    ),
                }
                rows.append(summary)
                done.add(key)
                pd.DataFrame(rows).to_csv(dest, index=False)
                print(summary, flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
