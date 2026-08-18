#!/usr/bin/env python
"""Phase 1: composition scan + spatial maps for a frozen motif catalog.

    python scripts/explore_motif_atlas.py --dataset hnc_wu2022 --data-root "$DATA_ROOT"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.atlas import (  # noqa: E402
    composition_table,
    plot_region_map,
    sample_atlas_regions,
    write_findings_template,
)
from benchmark.motifs.spec import default_catalog_path, load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--n", type=int, default=None, help="Override catalog atlas_n")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    dataset = load_dataset(args.dataset, data_root=args.data_root)
    out_dir = Path(args.output_dir or (_CODE / "results" / "motif_atlas" / args.dataset))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = dataset.get_metadata()
    if catalog.cv_filter:
        filtered = meta.query(catalog.cv_filter)
        meta = filtered if len(filtered) else meta
    region_ids = meta["region_id"].astype(str).tolist()
    print(f"Scanning composition for {len(region_ids)} regions ({dataset.name})", flush=True)

    comp = composition_table(dataset, catalog, region_ids)
    comp_path = out_dir / "composition_scan.csv"
    comp.to_csv(comp_path, index=False)
    print(f"Wrote {comp_path}", flush=True)

    n = args.n or catalog.atlas_n
    sampled = sample_atlas_regions(comp, n=n, seed=args.seed)
    print(f"Plotting {len(sampled)} atlas regions", flush=True)
    plot_dir = out_dir / "maps"
    for rid in sampled:
        dest = plot_dir / f"{rid}.png"
        plot_region_map(dataset, rid, catalog, dest)
        print(f"  {dest}", flush=True)

    findings = write_findings_template(comp, sampled, out_dir / "findings_template.csv")
    pd.Series(sampled, name="region_id").to_csv(out_dir / "sampled_regions.csv", index=False)
    print(f"Wrote {findings}", flush=True)


if __name__ == "__main__":
    main()
