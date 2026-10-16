#!/usr/bin/env python
"""Recut clean pseudo-label scores.

Spatial motifs: lowest 20% negative, highest 20% positive, middle 60% missing.
Controls (tumor_high, cd8_high): discovery-cohort median, same two-way split
as the global pseudo-labels. Clean source tables are not modified.

    python scripts/recut_pseudo_labels_extreme20.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.labels import discovery_mask  # noqa: E402
from benchmark.motifs.source_select import (  # noqa: E402
    SELECTED_MOTIF_SOURCES,
    selected_motifs_for,
)
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

CLEAN_REGION_DIR = _CODE / "results" / "pseudo_labels"
CLEAN_WINDOW_DIR = CLEAN_REGION_DIR / "mil_windows"
PACK_ROOT = Path("/autofs/nas8/tywang/tjzou/PseudoLabelDatasetExtreme20")
CONTROL_MOTIFS = ("tumor_high", "cd8_high")
DATASETS = (
    "bc_jackson2020",
    "hnc_wu2022",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)
LOW_Q = 0.20
HIGH_Q = 0.80
WINDOW_KEEP = {
    "region_id", "window_id", "cx_um", "cy_um", "n_cells", "motif_id",
    "score_used", "score", "label", "pos", "neg", "cohort", "patient_id", "dataset",
}


def extreme_quantile_labels(
    score: pd.Series,
    fit: pd.Series | None = None,
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
) -> tuple[pd.Series, float, float]:
    """Label the lowest ``low_q`` of finite scores 0 and the top ``1 - high_q`` 1.

    The middle stays missing. Ties are broken by original row order, so each
    motif is 20% negative, 60% missing, and 20% positive among finite scores.
    A raw score threshold would assign every tied row to the same side.
    ``fit`` is unused: the percentages apply to the table being written.
    """
    del fit
    values = pd.to_numeric(score, errors="coerce")
    label = pd.Series(np.nan, index=values.index, dtype="float")
    finite_idx = values.index[values.notna()]
    n = int(len(finite_idx))
    if n < 6:
        return label, float("nan"), float("nan")
    order = values.loc[finite_idx].to_numpy(dtype=float).argsort(kind="mergesort")
    ranked = finite_idx.to_numpy()[order]
    n_low = int(round(low_q * n))
    n_high = int(round((1.0 - high_q) * n))
    n_low = min(n_low, n)
    n_high = min(n_high, n - n_low)
    if n_low:
        label.loc[ranked[:n_low]] = 0.0
    if n_high:
        label.loc[ranked[-n_high:]] = 1.0
    lo = float(values.loc[ranked[n_low - 1]]) if n_low else float("nan")
    hi = float(values.loc[ranked[-n_high]]) if n_high else float("nan")
    return label, lo, hi


def median_labels(score: pd.Series, fit: pd.Series) -> tuple[pd.Series, float]:
    """Discovery-cohort median split used for composition controls.

    Finite scores at or above the discovery median are 1. There is no middle
    band. Rows with a missing score stay unlabeled.
    """
    values = pd.to_numeric(score, errors="coerce")
    fit_vals = values.loc[fit.fillna(False)].dropna()
    label = pd.Series(np.nan, index=values.index, dtype="float")
    if fit_vals.empty:
        return label, float("nan")
    mid = float(fit_vals.median())
    valid = values.notna()
    label.loc[valid] = (values.loc[valid] >= mid).astype(float)
    return label, mid


def _fractions(label: pd.Series, finite: pd.Series) -> dict:
    base = label.loc[finite]
    n = int(finite.sum())
    n1 = int((base == 1.0).sum())
    n0 = int((base == 0.0).sum())
    n_nan = int(base.isna().sum())
    return {
        "n_finite": n,
        "n_pos": n1,
        "n_neg": n0,
        "n_nan": n_nan,
        "frac_pos": n1 / n if n else float("nan"),
        "frac_neg": n0 / n if n else float("nan"),
        "frac_nan": n_nan / n if n else float("nan"),
    }


def recut_region(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = CLEAN_REGION_DIR / f"{dataset}_v2.csv"
    table = pd.read_csv(path)
    catalog = load_motif_catalog(dataset=dataset)
    fit = discovery_mask(table, catalog.cv_filter)
    rows = []
    motifs = [c[: -len("_score_used")] for c in table.columns if c.endswith("_score_used")]
    for motif in motifs:
        score = pd.to_numeric(table[f"{motif}_score_used"], errors="coerce")
        if motif in CONTROL_MOTIFS:
            label, mid = median_labels(score, fit)
            lo = hi = mid
        else:
            label, lo, hi = extreme_quantile_labels(score, fit)
        old_col = f"{motif}_label_v2"
        if old_col in table.columns:
            table[f"{motif}_label_v2_original"] = table[old_col]
        table[old_col] = label
        finite = score.notna()
        rec = _fractions(label, finite)
        rec.update({
            "level": "region",
            "dataset": dataset,
            "motif": motif,
            "n_fit": int((fit & finite).sum()),
            "cut_low": lo,
            "cut_high": hi,
        })
        rows.append(rec)
    return table, pd.DataFrame(rows)


def recut_windows(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = CLEAN_WINDOW_DIR / f"{dataset}.csv"
    motifs = set(selected_motifs_for(dataset))
    table = pd.read_csv(path, usecols=lambda c: c in WINDOW_KEEP)
    table = table[table["motif_id"].astype(str).isin(motifs)].copy()
    catalog = load_motif_catalog(dataset=dataset)
    fit_all = discovery_mask(table, catalog.cv_filter)
    score_col = "score_used" if "score_used" in table.columns else "score"
    rows = []
    parts = []
    for motif, part in table.groupby("motif_id", sort=False):
        part = part.copy()
        score = pd.to_numeric(part[score_col], errors="coerce")
        fit = fit_all.reindex(part.index).fillna(False)
        if str(motif) in CONTROL_MOTIFS:
            label, mid = median_labels(score, fit)
            lo = hi = mid
        else:
            label, lo, hi = extreme_quantile_labels(score, fit)
        if "label" in part.columns:
            part["label_original"] = pd.to_numeric(part["label"], errors="coerce")
        part["score_used"] = score
        part["label"] = label
        part["pos"] = (label == 1.0).astype(float)
        part["neg"] = (label == 0.0).astype(float)
        part.loc[label.isna(), ["pos", "neg"]] = np.nan
        finite = score.notna()
        rec = _fractions(label, finite)
        rec.update({
            "level": "window",
            "dataset": dataset,
            "motif": motif,
            "n_fit": int((fit & finite).sum()),
            "cut_low": lo,
            "cut_high": hi,
        })
        rows.append(rec)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else table.iloc[0:0]
    return out, pd.DataFrame(rows)


def _write_region_pack(tables: dict[str, pd.DataFrame], report: pd.DataFrame, dest: Path) -> None:
    per = dest / "per_dataset"
    per.mkdir(parents=True, exist_ok=True)
    for dataset, table in tables.items():
        path = per / f"{dataset}_v2.csv"
        table.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(table)}", flush=True)
    report.to_csv(dest / "cut_report.csv", index=False)

    catalog_src = PACK_ROOT.parent / "selected_catalog.csv"
    catalog = pd.read_csv(catalog_src) if catalog_src.exists() else None
    long_rows = []
    wide_frames = []
    summary = []
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        table = tables[dataset]
        score_col = f"{motif}_score_used"
        label_col = f"{motif}_label_v2"
        piece = table[["region_id", "patient_id", score_col, label_col]].copy()
        if "cohort" in table.columns:
            piece.insert(2, "cohort", table["cohort"])
        piece.insert(0, "source_dataset", dataset)
        piece = piece.rename(columns={score_col: "score_used", label_col: "pseudo_label"})
        piece["motif_id"] = motif
        labeled = piece[piece["pseudo_label"].notna()]
        long_rows.append(labeled)
        keep_wide = table[["region_id", "patient_id", score_col, label_col]].copy()
        keep_wide.insert(0, "source_dataset", dataset)
        keep_wide["region_id"] = keep_wide["region_id"].astype(str)
        keep_wide["patient_id"] = keep_wide["patient_id"].astype(str)
        wide_frames.append(keep_wide)
        n1 = int((table[label_col] == 1).sum())
        n0 = int((table[label_col] == 0).sum())
        rec = {
            "motif_id": motif,
            "source_dataset": dataset,
            "separator": "median" if motif in CONTROL_MOTIFS else "extreme20",
            "low_q": 0.5 if motif in CONTROL_MOTIFS else LOW_Q,
            "high_q": 0.5 if motif in CONTROL_MOTIFS else HIGH_Q,
            "n_labeled": n0 + n1,
            "n_0": n0,
            "n_1": n1,
        }
        if catalog is not None:
            hit = catalog[(catalog["motif_id"] == motif) & (catalog["source_dataset"] == dataset)]
            if len(hit):
                for col in ("source_dataset_name", "task_type", "panel", "definition"):
                    if col in hit.columns:
                        rec[col] = hit.iloc[0][col]
        summary.append(rec)
    if long_rows:
        pd.concat(long_rows, ignore_index=True).to_csv(dest / "selected_motif_labels.csv", index=False)
    if wide_frames:
        wide = wide_frames[0]
        for extra in wide_frames[1:]:
            wide = wide.merge(extra, on=["source_dataset", "region_id", "patient_id"], how="outer")
        wide.to_csv(dest / "metadata.csv", index=False)
    pd.DataFrame(summary).to_csv(dest / "selected_catalog.csv", index=False)


def _write_window_pack(tables: dict[str, pd.DataFrame], report: pd.DataFrame, dest: Path) -> None:
    per = dest / "per_dataset"
    per.mkdir(parents=True, exist_ok=True)
    frames = []
    summary = []
    for dataset, table in tables.items():
        path = per / f"{dataset}_windows.csv"
        table.to_csv(path, index=False)
        print(f"wrote {path}  rows={len(table)}", flush=True)
        for motif, part in table.groupby("motif_id", sort=False):
            labeled = part[part["label"].notna()].copy()
            labeled.insert(0, "source_dataset", dataset)
            frames.append(labeled)
            n1 = int((part["label"] == 1).sum())
            n0 = int((part["label"] == 0).sum())
            summary.append({
                "motif_id": motif,
                "source_dataset": dataset,
                "separator": "median" if str(motif) in CONTROL_MOTIFS else "extreme20",
                "n_windows": int(len(part)),
                "n_labeled": n0 + n1,
                "n_0": n0,
                "n_1": n1,
            })
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(dest / "selected_window_labels.csv", index=False)
    pd.DataFrame(summary).to_csv(dest / "selected_catalog.csv", index=False)
    report.to_csv(dest / "cut_report.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(PACK_ROOT))
    args = p.parse_args()
    root = Path(args.out)
    region_tables = {}
    window_tables = {}
    region_reports = []
    window_reports = []
    for dataset in DATASETS:
        print(f"[{dataset}] region", flush=True)
        region, region_report = recut_region(dataset)
        region_tables[dataset] = region
        region_reports.append(region_report)
        print(f"[{dataset}] window", flush=True)
        window, window_report = recut_windows(dataset)
        window_tables[dataset] = window
        window_reports.append(window_report)
    region_report = pd.concat(region_reports, ignore_index=True)
    window_report = pd.concat(window_reports, ignore_index=True)
    _write_region_pack(region_tables, region_report, root / "region")
    _write_window_pack(window_tables, window_report, root / "window")
    readme = root / "README.txt"
    readme.write_text(
        "Spatial motifs use a rank separator. Controls use the global median split.\n"
        "Spatial, among finite score_used, ties broken by original row order:\n"
        "  label 0 = lowest 20% (negative)\n"
        "  label 1 = highest 20% (positive)\n"
        "  NaN     = middle 60%, plus rows whose score_used is missing\n"
        "Controls (tumor_high, cd8_high): discovery-cohort median. A finite score\n"
        "at or above that median is positive; below it is negative.\n"
        "region/per_dataset/*_v2.csv : same columns as the clean v2 table.\n"
        "  *_label_v2 is the new cut. *_label_v2_original keeps the previous label.\n"
        "window/per_dataset/*_windows.csv : the 10 selected motif x dataset pairs.\n"
        "  label/pos/neg are the new cut. label_original keeps the tertile or median label.\n"
        "The noisy tertile pack at PseudoNoisyDataset/per_dataset and mil_windows/ is unchanged.\n"
        "Sources: TME_modeling_benchmark/results/pseudo_labels/*_v2.csv\n"
        "         TME_modeling_benchmark/results/pseudo_labels/mil_windows/<dataset>.csv\n"
    )
    print(region_report[["dataset", "motif", "frac_pos", "frac_neg", "frac_nan"]].to_string(index=False), flush=True)
    print(window_report[["dataset", "motif", "frac_pos", "frac_neg", "frac_nan"]].to_string(index=False), flush=True)
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
