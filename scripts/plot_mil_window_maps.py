#!/usr/bin/env python
"""Plot per-task MIL window maps for one region.

    python scripts/plot_mil_window_maps.py \\
        --input results/pseudo_labels/mil_windows/tnbc_wang2023.csv \\
        --region NTImg0011
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.source_select import selected_motifs_for  # noqa: E402
from benchmark.motifs.window_maps import plot_region_tasks  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="mil_windows long CSV")
    p.add_argument("--region", required=True)
    p.add_argument("--color", default="score", help="score | attention | score_clean")
    p.add_argument("--label-col", default="label")
    p.add_argument("--motifs", default="",
                    help="comma list; empty=selected pairs for this dataset file")
    p.add_argument("--attention-csv", default="", help="optional: region_id,window_id,attention")
    p.add_argument("--out", default="")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if args.attention_csv:
        att = pd.read_csv(args.attention_csv)
        df = df.merge(att, on=["region_id", "window_id"], how="left")

    motifs = [m.strip() for m in args.motifs.split(",") if m.strip()] or None
    if motifs is None:
        stem = Path(args.input).stem
        for suffix in ("_windows_noisy", "_noisy", "_windows"):
            stem = stem.removesuffix(suffix)
        picked = selected_motifs_for(stem)
        if picked:
            motifs = list(picked)
    fig = plot_region_tasks(
        df, args.region, color=args.color, label_col=args.label_col, motifs=motifs
    )
    out = Path(args.out) if args.out else Path(
        f"window_map_{args.region}_{args.color}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
