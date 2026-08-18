#!/usr/bin/env python
"""Refine tertile pseudo labels with patient-level bootstrap confidence.

Does not recompute motif scores. Reads an existing label table (from
``generate_pseudo_labels.py``) and writes ``<motif>_label_v2`` plus
``<motif>_p_low`` / ``<motif>_p_high``.

    python scripts/refine_pseudo_labels.py --dataset hnc_wu2022 --primary-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.labels import add_bootstrap_labels  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

PRIMARY_SPATIAL = (
    "immune_exclusion",
    "cd8_clustering",
    "tls_like",
    "interface_immune",
    "tumor_stroma_mixing",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--primary-only",
        action="store_true",
        help="Only refine the 5 core spatial motifs (controls copied as-is).",
    )
    ap.add_argument(
        "--motifs",
        nargs="*",
        default=None,
        help="Subset of motif ids. Default: all tertile motifs in the catalog.",
    )
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    inp = Path(args.input or (_CODE / "results" / "pseudo_labels" / f"{args.dataset}.csv"))
    if not inp.exists():
        raise FileNotFoundError(
            f"Label table not found: {inp}\nRun scripts/generate_pseudo_labels.py first."
        )
    table = pd.read_csv(inp)

    if catalog.cv_filter and "dataset" in table.columns:
        fit_mask = table.eval(catalog.cv_filter)
        if not bool(fit_mask.any()):
            fit_mask = pd.Series(True, index=table.index)
    else:
        fit_mask = pd.Series(True, index=table.index)

    motifs = args.motifs
    if args.primary_only:
        motifs = list(PRIMARY_SPATIAL)

    refined = add_bootstrap_labels(
        table,
        catalog,
        fit_mask=fit_mask,
        n_boot=args.n_boot,
        confidence=args.confidence,
        seed=args.seed,
        motifs=motifs,
    )

    rows = []
    for spec in catalog.motifs:
        v2 = f"{spec.id}_label_v2"
        if v2 not in refined.columns:
            continue
        lab = refined[v2]
        v1 = refined[spec.label_col] if spec.label_col in refined.columns else pd.Series(dtype=float)
        rows.append({
            "motif_id": spec.id,
            "n_v1_labeled": int(v1.notna().sum()) if len(v1) else 0,
            "n_v2_labeled": int(lab.notna().sum()),
            "n_v2_0": int((lab == 0).sum()),
            "n_v2_1": int((lab == 1).sum()),
            "n_uncertain": int(lab.isna().sum()),
            "confidence": args.confidence,
            "n_boot": args.n_boot,
        })

    out = Path(args.output or inp.with_name(inp.stem + "_v2.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    refined.to_csv(out, index=False)
    summary = pd.DataFrame(rows)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out}  ({len(refined)} rows)", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
