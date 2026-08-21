#!/usr/bin/env python
"""Assign each motif to its best source among HNC / Jackson / METABRIC / TNBC.

Writes HNC-style selected tables into ``results/pseudo_labels_all`` and a
combined metadata sidecar into ``/autofs/nas8/tywang/tjzou/CombinedDataset``.

Uses already-generated ``*_v2.csv`` label tables; does not re-score regions.

    python scripts/build_combined_pseudo_labels.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.source_select import (  # noqa: E402
    CANDIDATE_DATASETS,
    assign_motif_sources,
)

DEFAULT_COMBINED = Path("/autofs/nas8/tywang/tjzou/CombinedDataset")
DISPLAY_NAME = {
    "hnc_wu2022": "HNC-Wu2022",
    "bc_jackson2020": "BC-Jackson2020",
    "bc_metabric_ali2020": "BC-METABRIC",
    "tnbc_wang2023": "TNBC-Wang2023",
}


def _load_labels(labels_dir: Path, dataset: str) -> pd.DataFrame:
    path = labels_dir / f"{dataset}_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df["region_id"] = df["region_id"].astype(str)
    df["source_dataset"] = dataset
    if "dataset" not in df.columns:
        if "cohort" in df.columns:
            df["dataset"] = df["cohort"]
        else:
            df["dataset"] = dataset
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
    return df


def _id_cols(df: pd.DataFrame) -> list[str]:
    cols = ["source_dataset", "region_id"]
    for extra in ("patient_id", "dataset", "cohort"):
        if extra in df.columns:
            cols.append(extra)
    return cols


def build_long(assignment: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rec in assignment.itertuples(index=False):
        if not rec.eligible or not rec.source_dataset:
            continue
        table = labels[rec.source_dataset]
        score_col = f"{rec.motif_id}_score_used"
        label_col = f"{rec.motif_id}_label_v2"
        if score_col not in table.columns or label_col not in table.columns:
            raise KeyError(f"{rec.source_dataset} missing {score_col} or {label_col}")
        sub = table[_id_cols(table) + [score_col, label_col]].copy()
        sub = sub.rename(columns={score_col: "score_used", label_col: "pseudo_label"})
        labeled = sub[sub["pseudo_label"].notna()].copy()
        labeled["motif_id"] = rec.motif_id
        labeled["task_type"] = rec.task_type
        labeled["panel"] = rec.panel
        labeled["label_column"] = label_col
        labeled["source_dataset_name"] = DISPLAY_NAME[rec.source_dataset]
        rows.append(labeled)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    front = [
        "source_dataset", "source_dataset_name", "region_id", "patient_id",
        "dataset", "motif_id", "task_type", "panel", "score_used",
        "pseudo_label", "label_column",
    ]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def build_wide(assignment: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    assigned = assignment[assignment["eligible"] & assignment["source_dataset"].notna()]
    frames = []
    for dataset, table in labels.items():
        keep = ["source_dataset", "region_id"]
        for extra in ("patient_id", "dataset", "cohort", "n_cells"):
            if extra in table.columns:
                keep.append(extra)
        part = table[keep].copy()
        part["region_uid"] = part["source_dataset"] + ":" + part["region_id"].astype(str)
        motifs_here = assigned[assigned["source_dataset"] == dataset]
        for rec in motifs_here.itertuples(index=False):
            score_col = f"{rec.motif_id}_score_used"
            label_col = f"{rec.motif_id}_label_v2"
            part[score_col] = table[score_col]
            part[label_col] = table[label_col]
        frames.append(part)
    wide = pd.concat(frames, ignore_index=True, sort=False)
    front = ["region_uid", "source_dataset", "region_id", "patient_id", "dataset", "cohort", "n_cells"]
    front = [c for c in front if c in wide.columns]
    motif_cols = [c for c in wide.columns if c.endswith("_score_used") or c.endswith("_label_v2")]
    return wide[front + motif_cols]


def build_catalog(assignment: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in assignment.itertuples(index=False):
        n = n0 = n1 = 0
        if rec.eligible and len(long):
            lab = long.loc[long["motif_id"] == rec.motif_id, "pseudo_label"]
            n = int(lab.notna().sum())
            n0 = int((lab == 0).sum())
            n1 = int((lab == 1).sum())
        rows.append(dict(
            motif_id=rec.motif_id,
            source_dataset=rec.source_dataset,
            source_dataset_name=DISPLAY_NAME.get(rec.source_dataset) if rec.source_dataset else None,
            eligible=rec.eligible,
            task_type=rec.task_type,
            panel=rec.panel,
            label_col=f"{rec.motif_id}_label_v2",
            score_col=f"{rec.motif_id}_score_used",
            definition=rec.definition,
            n_labeled=n,
            n_0=n0,
            n_1=n1,
            auc_composition=rec.auc_composition,
            auc_expression=rec.auc_expression,
            auc_density=rec.auc_density,
            cd8_is_proxy=rec.cd8_is_proxy,
            reason=rec.reason,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expr-panel", default=str(_CODE / "results/motif_screening/motif_panel_composition_expression.csv"))
    ap.add_argument("--dens-panel", default=str(_CODE / "results/motif_screening/motif_panel_composition_density.csv"))
    ap.add_argument("--labels-dir", default=str(_CODE / "results/pseudo_labels"))
    ap.add_argument("--output-dir", default=str(_CODE / "results/pseudo_labels_all"))
    ap.add_argument("--combined-dir", default=str(DEFAULT_COMBINED))
    args = ap.parse_args()

    expr = pd.read_csv(args.expr_panel)
    dens = pd.read_csv(args.dens_panel)
    assignment = assign_motif_sources(expr, dens)
    labels = {ds: _load_labels(Path(args.labels_dir), ds) for ds in CANDIDATE_DATASETS}
    long = build_long(assignment, labels)
    wide = build_wide(assignment, labels)
    catalog = build_catalog(assignment, long)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(out / "motif_source_assignment.csv", index=False)
    eligible_catalog = catalog.loc[catalog["eligible"] == True].copy()
    eligible_catalog.to_csv(out / "selected_catalog.csv", index=False)
    catalog.to_csv(out / "motif_source_assignment_detail.csv", index=False)
    long.to_csv(out / "selected_motif_labels.csv", index=False)
    wide.to_csv(out / "selected_labels_wide.csv", index=False)

    combined = Path(args.combined_dir)
    combined.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(combined / "motif_source_assignment.csv", index=False)
    eligible_catalog.to_csv(combined / "selected_catalog.csv", index=False)
    long.to_csv(combined / "selected_motif_labels.csv", index=False)
    wide.to_csv(combined / "metadata.csv", index=False)

    print(catalog.to_string(index=False), flush=True)
    print(f"\nWrote {out}", flush=True)
    print(f"Wrote {combined}", flush=True)
    print(f"long rows={len(long)}  wide rows={len(wide)}  eligible motifs={int(catalog.eligible.sum())}", flush=True)


if __name__ == "__main__":
    main()
