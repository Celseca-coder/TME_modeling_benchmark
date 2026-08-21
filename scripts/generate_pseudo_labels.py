#!/usr/bin/env python
"""Apply frozen motif recipes and write region-level pseudo labels + QC.

    python scripts/generate_pseudo_labels.py --dataset hnc_wu2022 --data-root "$DATA_ROOT"
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.detect import score_region  # noqa: E402
from benchmark.motifs.labels import add_pseudo_labels, discovery_mask  # noqa: E402
from benchmark.motifs.qc import qc_table, spatial_null_report  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402

_WORKER: dict = {}


def _progress(items, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except ImportError:
        return items


def _init_worker(dataset_name: str, data_root: str | None, catalog_path: str | None) -> None:
    _WORKER["dataset"] = load_dataset(dataset_name, data_root=data_root)
    _WORKER["catalog"] = load_motif_catalog(catalog_path, dataset=dataset_name)


def _score_region_id(rid: str) -> dict:
    dataset = _WORKER["dataset"]
    catalog = _WORKER["catalog"]
    region = dataset.load_region(rid, normalize=False, use_cache=False)
    row = score_region(region, catalog)
    row["region_id"] = rid
    dataset.clear_region_cache()
    return row


def _score_all(dataset, catalog, region_ids: list[str], n_jobs: int,
               dataset_name: str, data_root: str | None, catalog_path: str | None):
    if n_jobs <= 1:
        rows = []
        for rid in _progress(region_ids, "motifs"):
            region = dataset.load_region(rid, normalize=False, use_cache=False)
            row = score_region(region, catalog)
            row["region_id"] = rid
            rows.append(row)
            dataset.clear_region_cache()
        return rows
    rows = []
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=(dataset_name, data_root, catalog_path),
    ) as pool:
        for row in _progress(pool.map(_score_region_id, region_ids, chunksize=4), "motifs"):
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-regions", type=int, default=None)
    ap.add_argument("--qc-regions", type=int, default=16,
                    help="Regions used for the spatial-shuffle null (spatial motifs only)")
    ap.add_argument("--skip-spatial-null", action="store_true",
                    help="Skip the coordinate-shuffle QC (faster for batch runs).")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Process workers for region scoring.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    dataset = load_dataset(args.dataset, data_root=args.data_root)
    meta = dataset.get_metadata().copy()
    region_ids = meta["region_id"].astype(str).tolist()
    present = [rid for rid in region_ids if dataset.region_dir(rid).is_dir()]
    n_missing = len(region_ids) - len(present)
    if n_missing:
        print(f"Skipping {n_missing} metadata rows with no region directory", flush=True)
    region_ids = present
    if args.max_regions:
        region_ids = region_ids[: args.max_regions]
    meta = meta[meta["region_id"].astype(str).isin(region_ids)].copy()

    print(
        f"Scoring {len(region_ids)} regions with {len(catalog.motifs)} motifs "
        f"(n_jobs={args.n_jobs})",
        flush=True,
    )
    rows = _score_all(
        dataset, catalog, region_ids, args.n_jobs,
        args.dataset, args.data_root, args.catalog,
    )

    scores = pd.DataFrame(rows)
    keep_cols = [
        c for c in ("region_id", "patient_id", "dataset", "cohort") if c in meta.columns
    ]
    table = meta[keep_cols].copy()
    table["region_id"] = table["region_id"].astype(str)
    scores["region_id"] = scores["region_id"].astype(str)
    table = table.merge(scores, on="region_id", how="left")

    fit_mask = discovery_mask(table, catalog.cv_filter)
    n_fit = int(fit_mask.sum())
    print(f"Cut-points fit on {n_fit}/{len(table)} regions ({catalog.cv_filter!r})", flush=True)

    labeled = add_pseudo_labels(table, catalog, fit_mask=fit_mask)
    out = Path(args.output or (_CODE / "results" / "pseudo_labels" / f"{args.dataset}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(labeled)} rows)", flush=True)

    qc = qc_table(labeled, catalog, fit_mask=fit_mask)
    qc_path = out.with_name(out.stem + "_qc.csv")

    if not args.skip_spatial_null:
        fit_ids = labeled.loc[fit_mask, "region_id"].astype(str).tolist()
        qc_ids = fit_ids[: args.qc_regions]
        print(f"Spatial null on {len(qc_ids)} regions", flush=True)
        null_rows = []
        qc_regions = [
            dataset.load_region(rid, normalize=False, use_cache=False) for rid in qc_ids
        ]
        for spec in catalog.motifs:
            if not spec.spatial:
                continue
            report = spatial_null_report(qc_regions, catalog, spec, seed=args.seed)
            report["motif_id"] = spec.id
            null_rows.append(report)
        if null_rows:
            null = pd.DataFrame(null_rows)
            qc = qc.merge(null, on="motif_id", how="left")
    qc.to_csv(qc_path, index=False)
    print(f"Wrote {qc_path}", flush=True)
    print(qc.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
