#!/usr/bin/env python
"""score_used maps with non-overlapping pos/neg cells, cell-type maps, and cell-size squares.

    python scripts/plot_window_maps_with_markers.py \
        --input results/pseudo_labels/mil_windows/tnbc_wang2023.csv \
        --dataset tnbc_wang2023 --region NTImg0011
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.motifs.window_maps import plot_region_pos_neg_with_types  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402

DEFAULT_MOTIFS = [
    "tumor_high",
    "cd8_high",
    "cd8_clustering",
    "tumor_stroma_mixing",
    "interface_immune",
    "immune_exclusion",
    "t_tumor_mixing",
    "cd8_tumor_contact",
    "macrophage_tumor_niche",
    "apc_t_contact",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--dataset", default="tnbc_wang2023")
    p.add_argument("--region", required=True)
    p.add_argument("--data-root", default="/autofs/bal14/zqwu/CellularTables/TME_benchmark_data")
    p.add_argument("--color", default="score_used")
    p.add_argument("--label-col", default="label")
    p.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    p.add_argument("--out", default="")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    catalog = load_motif_catalog(dataset=args.dataset)
    dataset = load_dataset(args.dataset, data_root=args.data_root)
    region = dataset.load_region(args.region, normalize=False, use_cache=False)
    motifs = [m.strip() for m in args.motifs.split(",") if m.strip()]
    fig = plot_region_pos_neg_with_types(
        df, region, catalog, args.region,
        color=args.color,
        label_col=args.label_col,
        motifs=motifs,
    )
    out = Path(args.out) if args.out else Path(
        f"results/pseudo_labels/mil_windows/maps/{args.dataset}/"
        f"{args.region}_{args.color}_pos_neg_types.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
