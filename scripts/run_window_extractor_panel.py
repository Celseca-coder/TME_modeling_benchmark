#!/usr/bin/env python
"""Extract the four global tabular families on MIL windows and eval every combo.

Lasso / SHAP in the region-level table share the same feature matrix; window
localisation uses Lasso on y_w, so each combo is scored once. KRONOS / Eva /
UTAG / SPACE-GM are not window tabular extractors and are not run here.

    python scripts/run_window_extractor_panel.py --dataset tnbc_wang2023
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import (
    WINDOW_PANEL_COMBOS,
    WINDOW_PANEL_FAMILIES,
    combo_name,
    expand_panel_groups,
    LocalWindowInstanceFeaturizer,
)
from benchmark.features.window_grid import DEFAULT_WINDOW_SIZE_UM, iter_windows
from benchmark.motifs.source_select import CANDIDATE_DATASETS, selected_motifs_for
from benchmark.motifs.spec import load_motif_catalog
from benchmark.motifs.window_extractor_eval import (
    PASS_WINDOW,
    evaluate_table,
    join_features_and_labels,
)
from benchmark.utils.registry import load_dataset

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
DEFAULT_FEATURE_DIR = Path("results/pseudo_labels/mil_windows/features")
DEFAULT_OUT = Path("results/pseudo_labels/mil_windows/extractor_localisation_panel.csv")


def _extract(dataset: str, data_root: str, feature_dir: Path) -> Path:
    dest = feature_dir / f"{dataset}_panel.csv"
    if dest.exists():
        print(f"reuse {dest}", flush=True)
        return dest

    catalog = load_motif_catalog(dataset=dataset)
    ds = load_dataset(dataset, data_root=data_root)
    win_path = WINDOW_DIR / f"{dataset}_windows_noisy.csv"
    keys = pd.read_csv(win_path, usecols=["region_id", "motif_id"])
    motifs = selected_motifs_for(dataset)
    if motifs:
        keys = keys[keys["motif_id"].astype(str).isin(motifs)]
    region_ids = sorted(keys["region_id"].astype(str).unique())
    print(f"[{dataset}] load {len(region_ids)} regions  groups={expand_panel_groups(WINDOW_PANEL_FAMILIES)}", flush=True)

    regions = [ds.load_region(rid, normalize=False, use_cache=False) for rid in region_ids]
    feat = LocalWindowInstanceFeaturizer(
        feature_groups=expand_panel_groups(WINDOW_PANEL_FAMILIES),
        cell_type_col=catalog.cell_type_col,
    ).fit(regions)
    print(f"[{dataset}] n_features={len(feat.feature_names())}", flush=True)

    frames = []
    for i, region in enumerate(regions):
        windows = iter_windows(region)
        part = feat.extract_table(region, windows, DEFAULT_WINDOW_SIZE_UM).reset_index()
        part.insert(0, "region_id", str(region.region_id))
        frames.append(part)
        if i % 20 == 0 or i + 1 == len(regions):
            print(f"[{dataset}] extract {i + 1}/{len(regions)} {region.region_id} windows={len(part)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  rows={len(out)}  cols={len(out.columns)}", flush=True)
    return dest


def _upsert(path: Path, new: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not new.empty:
        old = pd.read_csv(path)
        key = ["dataset", "motif", "feature_group"]
        if all(c in old.columns for c in key):
            idx = new.set_index(key).index
            old = old[~old.set_index(key).index.isin(idx)]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _eval(dataset: str, features: Path, out: Path) -> None:
    feats = pd.read_csv(features)
    wins = pd.read_csv(WINDOW_DIR / f"{dataset}_windows_noisy.csv")
    bags = pd.read_csv(
        BAG_DIR / f"{dataset}_v2_noisy.csv",
        usecols=lambda c: c in {"region_id", "patient_id"},
    )
    motifs = list(selected_motifs_for(dataset))
    rows = []
    for families in WINDOW_PANEL_COMBOS:
        group = combo_name(families)
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            fold = evaluate_table(tab, families=families)
            summary = {
                "dataset": dataset,
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
    _upsert(out, pd.DataFrame(rows))
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CANDIDATE_DATASETS))
    p.add_argument("--data-root", default="/autofs/bal14/zqwu/CellularTables/TME_benchmark_data")
    p.add_argument("--feature-dir", default=str(DEFAULT_FEATURE_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    feature_dir = Path(args.feature_dir)
    out = Path(args.out)
    for dataset in args.datasets:
        if args.eval_only:
            dest = feature_dir / f"{dataset}_panel.csv"
            if not dest.exists():
                raise FileNotFoundError(dest)
        else:
            dest = _extract(dataset, args.data_root, feature_dir)
        if not args.extract_only:
            _eval(dataset, dest, out)


if __name__ == "__main__":
    main()
