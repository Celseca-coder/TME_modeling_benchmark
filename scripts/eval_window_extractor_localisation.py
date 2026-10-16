#!/usr/bin/env python
"""Did this extractor recover GT-important windows? Train/eval on y_w.

    python scripts/eval_window_extractor_localisation.py \
        --dataset tnbc_wang2023 --motif cd8_high \
        --features results/pseudo_labels/mil_windows/features/tnbc_wang2023_composition.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.source_select import selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    PASS_WINDOW,
    evaluate_table,
    join_features_and_labels,
)

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--motif", default="")
    p.add_argument("--features", required=True, help="CSV from extract_window_tabular_features.py")
    p.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset columns to these panel families, e.g. composition density",
    )
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/extractor_localisation.csv")
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    wins = pd.read_csv(WINDOW_DIR / f"{args.dataset}_windows_noisy.csv")
    bags = pd.read_csv(BAG_DIR / f"{args.dataset}_v2_noisy.csv",
                       usecols=lambda c: c in {"region_id", "patient_id"})
    motifs = [args.motif] if args.motif else list(selected_motifs_for(args.dataset))
    families = tuple(args.families) if args.families else None
    group = (
        "+".join(families)
        if families
        else Path(args.features).stem.replace(f"{args.dataset}_", "", 1)
    )

    rows = []
    for motif in motifs:
        tab = join_features_and_labels(feats, wins, motif, bags)
        fold = evaluate_table(tab, families=families)
        summary = {
            "dataset": args.dataset,
            "motif": motif,
            "feature_group": group,
            "n_windows": int(len(tab)),
            "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
            "auroc_window_mean": float(fold["auroc_window_mean"].mean()),
            "n_regions_scored": int(fold["n_regions_scored"].sum()),
            "n_folds": int(len(fold)),
            "pass_window": bool(fold["auroc_window_mean"].mean() >= PASS_WINDOW),
        }
        rows.append(summary)
        print(summary, flush=True)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if dest.exists():
        old = pd.read_csv(dest)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(dest, index=False)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()