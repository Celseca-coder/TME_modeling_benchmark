#!/usr/bin/env python
"""Within-region precision / recall of y_w=1 among the top 20% scored windows.

Uses the same features, noisy window GT, and patient CV as the localisation
panel. Default solver is L2 (lbfgs): ranking of p_w matches L1 closely and
finishes in hours rather than days.

    python scripts/eval_window_extractor_topk.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import (  # noqa: E402
    FAMILY_COLUMN_PREFIXES,
    WINDOW_PANEL_COMBOS,
    combo_name,
)
from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    evaluate_table,
    join_features_and_labels,
)

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
FEATURE_DIR = Path("results/pseudo_labels/mil_windows/features")
DEFAULT_OUT = Path("results/pseudo_labels/mil_windows/extractor_localisation_topk20.csv")

META_COLS = {"region_id", "window_id"}


def _usecols(families: tuple[str, ...], columns: list[str]) -> list[str]:
    prefixes = [p for fam in families for p in FAMILY_COLUMN_PREFIXES[fam]]
    keep = []
    for col in columns:
        if col in META_COLS or any(col == p or col.startswith(p) for p in prefixes):
            keep.append(col)
    return keep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--l1-ratio", type=float, default=0.0)
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()

    pairs = list(SELECTED_MOTIF_SOURCES)
    if args.datasets:
        pairs = [(d, m) for d, m in pairs if d in set(args.datasets)]

    by_ds: dict[str, list[str]] = {}
    for dataset, motif in pairs:
        by_ds.setdefault(dataset, []).append(motif)

    rows = []
    for dataset, motifs in by_ds.items():
        feat_path = FEATURE_DIR / f"{dataset}_panel.csv"
        header = pd.read_csv(feat_path, nrows=0).columns.tolist()
        wins = pd.read_csv(WINDOW_DIR / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(
            BAG_DIR / f"{dataset}_v2_noisy.csv",
            usecols=lambda c: c in {"region_id", "patient_id"},
        )
        for families in WINDOW_PANEL_COMBOS:
            group = combo_name(families)
            feats = pd.read_csv(feat_path, usecols=_usecols(families, header))
            for motif in motifs:
                tab = join_features_and_labels(feats, wins, motif, bags)
                fold = evaluate_table(tab, families=families, l1_ratio=args.l1_ratio)
                rec = {
                    "dataset": dataset,
                    "motif": motif,
                    "feature_group": group,
                    "n_windows": int(len(tab)),
                    "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
                    "n_regions_topk": int(fold["n_regions_topk"].sum()),
                    "n_folds": int(len(fold)),
                    "precision_at_20": float(fold["precision_at_20"].mean()),
                    "recall_at_20": float(fold["recall_at_20"].mean()),
                    "labeled_prev": float(fold["labeled_prev"].mean()),
                    "auroc_window_mean": float(fold["auroc_window_mean"].mean()),
                }
                rows.append(rec)
                print(rec, flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {dest}  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
