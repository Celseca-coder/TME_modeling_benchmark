#!/usr/bin/env python
"""Write per-window extractor tables (join keys: region_id, window_id).

    python scripts/extract_window_tabular_features.py \
        --dataset tnbc_wang2023 --groups composition mixing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import (
    LocalWindowInstanceFeaturizer,
    WINDOW_PANEL_FAMILIES,
    expand_panel_groups,
)
from benchmark.features.window_grid import DEFAULT_WINDOW_SIZE_UM, iter_windows
from benchmark.motifs.source_select import selected_motifs_for
from benchmark.motifs.spec import load_motif_catalog
from benchmark.utils.registry import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--groups", nargs="+", default=["composition", "mixing"])
    p.add_argument(
        "--panel",
        action="store_true",
        help="Extract the global tabular families: composition, density, mixing, point_pattern",
    )
    p.add_argument("--data-root", default="/autofs/bal14/zqwu/CellularTables/TME_benchmark_data")
    p.add_argument("--window-csv", default="")
    p.add_argument("--out-dir", default="results/pseudo_labels/mil_windows/features")
    args = p.parse_args()
    groups = expand_panel_groups(WINDOW_PANEL_FAMILIES) if args.panel else tuple(args.groups)

    catalog = load_motif_catalog(dataset=args.dataset)
    ds = load_dataset(args.dataset, data_root=args.data_root)
    win_path = Path(args.window_csv) if args.window_csv else Path(
        f"/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset/"
        f"{args.dataset}_windows_noisy.csv"
    )
    keys = pd.read_csv(win_path, usecols=["region_id", "window_id"])
    region_ids = sorted(keys["region_id"].astype(str).unique())
    motifs = selected_motifs_for(args.dataset)
    if motifs:
        keep = pd.read_csv(win_path, usecols=["region_id", "motif_id"])
        keep = keep[keep["motif_id"].astype(str).isin(motifs)]
        region_ids = sorted(keep["region_id"].astype(str).unique())

    regions = [ds.load_region(rid, normalize=False, use_cache=False) for rid in region_ids]
    feat = LocalWindowInstanceFeaturizer(
        feature_groups=groups,
        cell_type_col=catalog.cell_type_col,
    ).fit(regions)

    frames = []
    for i, region in enumerate(regions):
        windows = iter_windows(region)
        part = feat.extract_table(region, windows, DEFAULT_WINDOW_SIZE_UM).reset_index()
        part.insert(0, "region_id", str(region.region_id))
        frames.append(part)
        if i % 20 == 0 or i + 1 == len(regions):
            print(f"  extract {i + 1}/{len(regions)} {region.region_id} windows={len(part)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    stem = "panel" if args.panel else "_".join(groups)
    dest = Path(args.out_dir) / f"{args.dataset}_{stem}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  rows={len(out)}  cols={len(out.columns)}", flush=True)


if __name__ == "__main__":
    main()