#!/usr/bin/env python
"""Write MIL window-level motif GT (pos/neg/NaN), aligned to AttnMIL tiles.

    python scripts/generate_mil_window_labels.py --dataset hnc_wu2022 --data-root "$DATA_ROOT"

Does not replace region bag labels. Default motifs are the 10 selected
dataset×motif pairs; pass --all-motifs for the full catalog. Optional
--region-labels keeps only motifs whose region pseudo-label is non-missing.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.window_grid import (  # noqa: E402
    DEFAULT_MIN_CELLS,
    DEFAULT_STEP_UM,
    DEFAULT_WINDOW_SIZE_UM,
)
from benchmark.motifs.mil_evidence import (  # noqa: E402
    add_window_pseudo_labels,
    label_windows_for_region,
)
from benchmark.motifs.source_select import selected_motifs_for  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402

_WORKER: dict = {}


def _progress(items, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except ImportError:
        return items


def _init_worker(dataset_name, data_root, catalog_path, window_size_um, step_um, min_cells, motif_ids):
    _WORKER["dataset"] = load_dataset(dataset_name, data_root=data_root)
    _WORKER["catalog"] = load_motif_catalog(catalog_path, dataset=dataset_name)
    _WORKER["window_size_um"] = window_size_um
    _WORKER["step_um"] = step_um
    _WORKER["min_cells"] = min_cells
    _WORKER["motif_ids"] = motif_ids


def _label_region_id(rid: str) -> pd.DataFrame:
    dataset = _WORKER["dataset"]
    region = dataset.load_region(rid, normalize=False, use_cache=False)
    table = label_windows_for_region(
        region,
        _WORKER["catalog"],
        motif_ids=_WORKER["motif_ids"],
        window_size_um=_WORKER["window_size_um"],
        step_um=_WORKER["step_um"],
        min_cells=_WORKER["min_cells"],
    )
    dataset.clear_region_cache()
    return table


def _filter_by_region_labels(table: pd.DataFrame, region_labels: pd.DataFrame) -> pd.DataFrame:
    lab = region_labels.copy()
    lab["region_id"] = lab["region_id"].astype(str)
    keep = []
    for motif_id, part in table.groupby("motif_id", sort=False):
        col = f"{motif_id}_label"
        if col not in lab.columns:
            continue
        labeled = set(lab.loc[lab[col].notna(), "region_id"].astype(str))
        keep.append(part[part["region_id"].astype(str).isin(labeled)])
    return pd.concat(keep, ignore_index=True) if keep else table.iloc[0:0].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--region-labels", default=None,
                    help="Region pseudo-label CSV; default results/pseudo_labels/<dataset>.csv")
    ap.add_argument("--all-regions", action="store_true",
                    help="Score every region×motif, ignore missing region labels.")
    ap.add_argument("--motifs", nargs="*", default=None,
                    help="Override motifs. Default: the selected pair(s) for this dataset.")
    ap.add_argument("--all-motifs", action="store_true",
                    help="Score every catalog motif instead of the 10 selected pairs.")
    ap.add_argument("--window-size-um", type=float, default=DEFAULT_WINDOW_SIZE_UM)
    ap.add_argument("--step-um", type=float, default=DEFAULT_STEP_UM)
    ap.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS)
    ap.add_argument("--max-regions", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    if args.motifs:
        motif_ids = list(args.motifs)
    elif args.all_motifs:
        motif_ids = [spec.id for spec in catalog.motifs]
    else:
        motif_ids = list(selected_motifs_for(args.dataset))
        if not motif_ids:
            raise SystemExit(f"{args.dataset}: no selected motif×dataset pairs")
    print(f"motifs: {motif_ids}", flush=True)
    dataset = load_dataset(args.dataset, data_root=args.data_root)
    meta = dataset.get_metadata()
    region_ids = [str(x) for x in meta["region_id"].tolist() if dataset.region_dir(str(x)).is_dir()]
    if args.max_regions:
        region_ids = region_ids[: args.max_regions]

    if args.n_jobs <= 1:
        _init_worker(args.dataset, args.data_root, args.catalog,
                     args.window_size_um, args.step_um, args.min_cells, motif_ids)
        frames = [_label_region_id(rid) for rid in _progress(region_ids, "mil-gt")]
    else:
        frames = []
        with ProcessPoolExecutor(
            max_workers=args.n_jobs,
            initializer=_init_worker,
            initargs=(args.dataset, args.data_root, args.catalog,
                      args.window_size_um, args.step_um, args.min_cells, motif_ids),
        ) as pool:
            for table in _progress(pool.map(_label_region_id, region_ids, chunksize=2), "mil-gt"):
                frames.append(table)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not table.empty:
        keep = [c for c in ("region_id", "dataset", "cohort", "patient_id") if c in meta.columns]
        if keep:
            meta_key = meta[keep].copy()
            meta_key["region_id"] = meta_key["region_id"].astype(str)
            table["region_id"] = table["region_id"].astype(str)
            table = table.merge(meta_key.drop_duplicates("region_id"), on="region_id", how="left")
        table = add_window_pseudo_labels(table, catalog)
    if not args.all_regions:
        lab_path = Path(args.region_labels or (_CODE / "results" / "pseudo_labels" / f"{args.dataset}.csv"))
        if lab_path.exists():
            table = _filter_by_region_labels(table, pd.read_csv(lab_path))
        else:
            print(f"No region labels at {lab_path}; writing all windows. Use --all-regions to silence.", flush=True)

    out = Path(args.output or (_CODE / "results" / "pseudo_labels" / "mil_windows" / f"{args.dataset}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(table)} rows)", flush=True)


if __name__ == "__main__":
    main()