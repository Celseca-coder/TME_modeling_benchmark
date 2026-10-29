#!/usr/bin/env python
"""Build a no-direction 1-in-5 pack from PseudoLabelDatasetExtreme20 and add noise.

Source tables are the clean Extreme20 pack. Scores are ``*_score_used`` /
``score_used``. The directional Extreme20 label (low 20% = 0, high 20% = 1,
middle missing) is kept as ``*_label_v2_original`` / ``label_original``.

Among finite scores, ties broken by original row order:
  label 1 = lowest fifth or highest fifth
  label 0 = middle three fifths
  NaN     = missing score

Controls use the same two-tail cut. There is no median split and no unlabeled
middle band.

Noise matches the other packs: ``N(0, 0.10 * SD(score))``, recut with the
clean fifth boundaries, then a 5% symmetric flip of labeled rows. Region SD
uses every finite score. Window SD uses the discovery cohort when the catalog
filter can be evaluated.

    python scripts/build_nodirect_1in5.py
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
from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

SRC = Path("/autofs/nas8/tywang/tjzou/PseudoLabelDatasetExtreme20")
PACK = Path("/autofs/nas8/tywang/tjzou/PseudoLabelDataset1in5NoDirect")
CATALOG_SRC = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/selected_catalog.csv")
DATASETS = (
    "bc_jackson2020",
    "hnc_wu2022",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)
TAIL = 0.20


def fifths_positive_tails(score: pd.Series) -> tuple[pd.Series, float, float]:
    """Mark both outer fifths positive and the middle three fifths negative.

    Returns ``(label, cut_low, cut_high)``. ``cut_low`` is the highest score
    inside the bottom fifth. ``cut_high`` is the lowest score inside the top
    fifth. A later noisy score is positive when it falls on or outside those
    two values.
    """
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    label = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    n = int(finite.size)
    if n < 5:
        return pd.Series(label, index=score.index), float("nan"), float("nan")
    order = np.argsort(values[finite], kind="mergesort")
    ranked = finite[order]
    n_low = int(round(TAIL * n))
    n_high = int(round(TAIL * n))
    if n_low + n_high > n:
        n_high = n - n_low
    n_mid = n - n_low - n_high
    if n_low:
        label[ranked[:n_low]] = 1.0
    if n_mid:
        label[ranked[n_low:n_low + n_mid]] = 0.0
    if n_high:
        label[ranked[n_low + n_mid:]] = 1.0
    lo = float(values[ranked[n_low - 1]]) if n_low else float("nan")
    hi = float(values[ranked[n_low + n_mid]]) if n_high else float("nan")
    return pd.Series(label, index=score.index), lo, hi


def recut_tails(score: pd.Series, lo: float, hi: float) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float")
    valid = values.notna()
    if not valid.any() or not np.isfinite(lo) or not np.isfinite(hi):
        return out
    extreme = valid & ((values <= lo) | (values >= hi))
    out.loc[extreme] = 1.0
    out.loc[valid & ~extreme] = 0.0
    return out


def symmetric_flip(labels: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
    out = labels.copy()
    if rate <= 0:
        return out
    idx = out.index[out.notna()]
    if len(idx) == 0:
        return out
    n_flip = int(round(rate * len(idx)))
    if n_flip == 0:
        return out
    chosen = rng.choice(idx.to_numpy(), size=min(n_flip, len(idx)), replace=False)
    out.loc[chosen] = 1.0 - out.loc[chosen]
    return out


def _fractions(label: pd.Series, finite: pd.Series) -> dict:
    base = label.loc[finite.fillna(False)]
    n = int(finite.fillna(False).sum())
    n1 = int((base == 1.0).sum())
    n0 = int((base == 0.0).sum())
    return {
        "n_finite": n,
        "n_pos": n1,
        "n_neg": n0,
        "frac_pos": n1 / n if n else float("nan"),
        "frac_neg": n0 / n if n else float("nan"),
    }


def _motif_ids(table: pd.DataFrame) -> list[str]:
    suffix = "_score_used"
    skip = ("_score_used_clean",)
    ids = []
    for col in table.columns:
        if not col.endswith(suffix) or col.endswith(skip):
            continue
        ids.append(col[: -len(suffix)])
    return ids


def _pos_neg(label: pd.Series) -> tuple[pd.Series, pd.Series]:
    pos = (label == 1.0).astype(float)
    neg = (label == 0.0).astype(float)
    missing = label.isna()
    pos = pos.mask(missing, np.nan)
    neg = neg.mask(missing, np.nan)
    return pos, neg


def build_region(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    path = SRC / "region" / "per_dataset" / f"{dataset}_v2.csv"
    table = pd.read_csv(path)
    cuts: dict[str, tuple[float, float]] = {}
    rows = []
    for motif in _motif_ids(table):
        score = pd.to_numeric(table[f"{motif}_score_used"], errors="coerce")
        label, lo, hi = fifths_positive_tails(score)
        directional = f"{motif}_label_v2"
        if directional in table.columns:
            table[f"{motif}_label_v2_original"] = pd.to_numeric(table[directional], errors="coerce")
        table[f"{motif}_score_used"] = score
        table[f"{motif}_label_v2"] = label
        cuts[motif] = (lo, hi)
        finite = score.notna()
        rec = _fractions(label, finite)
        rec.update({
            "level": "region",
            "dataset": dataset,
            "motif": motif,
            "cut_low": lo,
            "cut_high": hi,
        })
        rows.append(rec)
    drop = [c for c in table.columns if c.endswith("_score_used_clean") or c.endswith("_label_v2_clean")]
    table = table.drop(columns=drop)
    return table, pd.DataFrame(rows), cuts


def noise_region(
    table: pd.DataFrame,
    cuts: dict[str, tuple[float, float]],
    scale: float,
    seed: int,
    label_flip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = table.copy()
    rows = []
    for motif, (lo, hi) in cuts.items():
        score_col = f"{motif}_score_used"
        label_col = f"{motif}_label_v2"
        scores = pd.to_numeric(out[score_col], errors="coerce")
        sd = float(np.nanstd(scores.to_numpy(dtype=float)))
        sigma = scale * sd if np.isfinite(sd) and sd > 0 else 0.0
        noise = rng.normal(0.0, sigma, size=len(scores))
        noise = np.where(scores.notna().to_numpy(), noise, np.nan)
        noisy = pd.Series(scores.to_numpy(dtype=float) + noise, index=scores.index)
        clean_label = pd.to_numeric(out[label_col], errors="coerce")
        new_label = symmetric_flip(recut_tails(noisy, lo, hi), label_flip, rng)
        out[f"{motif}_score_used_clean"] = scores
        out[f"{motif}_label_v2_clean"] = clean_label
        out[score_col] = noisy
        out[label_col] = new_label
        both = clean_label.notna() & new_label.notna()
        n_old = int(clean_label.notna().sum())
        n_flip = int((both & (clean_label != new_label)).sum())
        rows.append({
            "level": "region",
            "motif": motif,
            "sigma": sigma,
            "score_sd": sd,
            "n_labeled_clean": n_old,
            "n_labeled_noisy": int(new_label.notna().sum()),
            "n_flipped": n_flip,
            "flip_rate": n_flip / n_old if n_old else float("nan"),
            "n_gained_label": int((clean_label.isna() & new_label.notna()).sum()),
            "n_lost_label": int((clean_label.notna() & new_label.isna()).sum()),
            "cut_low": lo,
            "cut_high": hi,
        })
    return out, pd.DataFrame(rows)


def build_windows(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    path = SRC / "window" / "per_dataset" / f"{dataset}_windows.csv"
    table = pd.read_csv(path)
    catalog = load_motif_catalog(dataset=dataset)
    fit_all = discovery_mask(table, catalog.cv_filter)
    cuts: dict[str, tuple[float, float]] = {}
    rows = []
    parts = []
    for motif, part in table.groupby("motif_id", sort=False):
        part = part.copy()
        score = pd.to_numeric(part["score_used"], errors="coerce")
        label, lo, hi = fifths_positive_tails(score)
        part["label_original"] = pd.to_numeric(part["label"], errors="coerce")
        part["score"] = score
        part["score_used"] = score
        part["label"] = label
        pos, neg = _pos_neg(label)
        part["pos"] = pos
        part["neg"] = neg
        fit = fit_all.reindex(part.index).fillna(False)
        fit_scores = score.loc[fit]
        pool = fit_scores if fit_scores.notna().any() else score
        sd = float(np.nanstd(pool.to_numpy(dtype=float)))
        cuts[str(motif)] = (lo, hi)
        finite = score.notna()
        rec = _fractions(label, finite)
        rec.update({
            "level": "window",
            "dataset": dataset,
            "motif": str(motif),
            "n_fit": int((fit & finite).sum()),
            "score_sd": sd,
            "cut_low": lo,
            "cut_high": hi,
        })
        rows.append(rec)
        keep = [
            "region_id", "window_id", "cx_um", "cy_um", "n_cells", "motif_id",
            "score", "pos", "neg", "label", "score_used", "label_original",
        ]
        for extra in ("patient_id", "dataset", "cohort"):
            if extra in part.columns:
                keep.append(extra)
        parts.append(part[keep])
    out = pd.concat(parts, ignore_index=True) if parts else table.iloc[0:0]
    return out, pd.DataFrame(rows), cuts


def noise_windows(
    table: pd.DataFrame,
    dataset: str,
    cuts: dict[str, tuple[float, float]],
    scale: float,
    seed: int,
    label_flip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = load_motif_catalog(dataset=dataset)
    fit_all = discovery_mask(table, catalog.cv_filter)
    rng = np.random.default_rng(seed)
    parts = []
    rows = []
    for motif, part in table.groupby("motif_id", sort=False):
        part = part.copy()
        lo, hi = cuts[str(motif)]
        scores = pd.to_numeric(part["score_used"], errors="coerce")
        fit = fit_all.reindex(part.index).fillna(False)
        fit_scores = scores.loc[fit]
        pool = fit_scores if fit_scores.notna().any() else scores
        sd = float(np.nanstd(pool.to_numpy(dtype=float)))
        if not np.isfinite(sd) or sd == 0:
            sd = float(np.nanstd(scores.to_numpy(dtype=float)))
        sigma = scale * sd if np.isfinite(sd) and sd > 0 else 0.0
        noise = rng.normal(0.0, sigma, size=len(scores))
        noise = np.where(scores.notna().to_numpy(), noise, np.nan)
        noisy = pd.Series(scores.to_numpy(dtype=float) + noise, index=part.index)
        clean_label = pd.to_numeric(part["label"], errors="coerce")
        new_label = symmetric_flip(recut_tails(noisy, lo, hi), label_flip, rng)
        pos, neg = _pos_neg(new_label)
        part["score_clean"] = scores
        part["label_clean"] = clean_label
        part["score_used"] = noisy
        part["score"] = noisy
        part["label"] = new_label
        part["pos"] = pos
        part["neg"] = neg
        both = clean_label.notna() & new_label.notna()
        n_old = int(clean_label.notna().sum())
        n_flip = int((both & (clean_label != new_label)).sum())
        rows.append({
            "level": "window",
            "motif": str(motif),
            "sigma": sigma,
            "score_sd": sd,
            "n_windows": int(len(part)),
            "n_labeled_clean": n_old,
            "n_labeled_noisy": int(new_label.notna().sum()),
            "n_flipped": n_flip,
            "flip_rate": n_flip / n_old if n_old else float("nan"),
            "n_gained_label": int((clean_label.isna() & new_label.notna()).sum()),
            "n_lost_label": int((clean_label.notna() & new_label.isna()).sum()),
            "cut_low": lo,
            "cut_high": hi,
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def _write_region_sidecars(tables: dict[str, pd.DataFrame], dest: Path) -> None:
    catalog = pd.read_csv(CATALOG_SRC) if CATALOG_SRC.exists() else None
    long_rows = []
    summary = []
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        table = tables[dataset]
        score_col = f"{motif}_score_used"
        label_col = f"{motif}_label_v2"
        cols = ["region_id", "patient_id", score_col, label_col]
        if "cohort" in table.columns:
            cols.insert(2, "cohort")
        piece = table[cols].copy()
        piece.insert(0, "source_dataset", dataset)
        piece = piece.rename(columns={score_col: "score_used", label_col: "pseudo_label"})
        piece["motif_id"] = motif
        long_rows.append(piece[piece["pseudo_label"].notna()])
        n1 = int((table[label_col] == 1).sum())
        n0 = int((table[label_col] == 0).sum())
        rec = {
            "motif_id": motif,
            "source_dataset": dataset,
            "separator": "nodirect_1in5",
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
    pd.DataFrame(summary).to_csv(dest / "selected_catalog.csv", index=False)


def _write_window_sidecars(tables: dict[str, pd.DataFrame], dest: Path) -> None:
    frames = []
    summary = []
    for dataset, table in tables.items():
        for motif, part in table.groupby("motif_id", sort=False):
            labeled = part[part["label"].notna()].copy()
            labeled.insert(0, "source_dataset", dataset)
            frames.append(labeled)
            n1 = int((part["label"] == 1).sum())
            n0 = int((part["label"] == 0).sum())
            summary.append({
                "motif_id": motif,
                "source_dataset": dataset,
                "separator": "nodirect_1in5",
                "n_windows": int(len(part)),
                "n_labeled": n0 + n1,
                "n_0": n0,
                "n_1": n1,
            })
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(dest / "selected_window_labels.csv", index=False)
    pd.DataFrame(summary).to_csv(dest / "selected_catalog.csv", index=False)


def _readme() -> str:
    return (
        "No-direction 1-in-5 labels built from PseudoLabelDatasetExtreme20 clean scores.\n"
        "Both outer fifths are the positive class. The middle three fifths are the negative class.\n"
        "Among finite scores, ties are broken by original row order:\n"
        "  label 1 = lowest 1/5 or highest 1/5\n"
        "  label 0 = middle 3/5\n"
        "  NaN     = missing score_used\n"
        "Controls (tumor_high, cd8_high) use this same cut, not the median split.\n"
        "Region and window clean tables keep the pre-noise score in score_used.\n"
        "  *_label_v2 / label is the no-direction cut.\n"
        "  *_label_v2_original / label_original keeps the directional Extreme20 label\n"
        "  (lowest 20% = 0, highest 20% = 1, middle missing; controls were median).\n"
        "Sources: PseudoLabelDatasetExtreme20/region/per_dataset/*_v2.csv (score_used)\n"
        "         PseudoLabelDatasetExtreme20/window/per_dataset/*_windows.csv (score_used)\n"
        "Window tables are the 10 selected motif x dataset pairs.\n"
        "\n"
        "Noisy copies sit next to the clean tables (*_v2_noisy.csv, *_windows_noisy.csv).\n"
        "Noise is N(0, 0.10 * SD(score_used)), then the label is recut with the clean\n"
        "fifth boundaries (score <= cut_low or score >= cut_high is 1, the open\n"
        "interval between them is 0), then 5% of labeled rows are flipped.\n"
        "Region SD uses all finite scores. Window SD uses the discovery cohort.\n"
        "See region/cut_report.csv, region/noise_report.csv, window/cut_report.csv,\n"
        "and window/noise_report.csv.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PACK))
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--label-flip", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.out)
    region_dir = root / "region" / "per_dataset"
    window_dir = root / "window" / "per_dataset"
    region_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)

    region_tables: dict[str, pd.DataFrame] = {}
    window_tables: dict[str, pd.DataFrame] = {}
    region_cuts_rows = []
    window_cuts_rows = []
    region_noise_rows = []
    window_noise_rows = []
    for dataset in DATASETS:
        print(f"[{dataset}] region", flush=True)
        region, region_cut, cuts = build_region(dataset)
        region_tables[dataset] = region
        region_cuts_rows.append(region_cut)
        clean_path = region_dir / f"{dataset}_v2.csv"
        region.to_csv(clean_path, index=False)
        print(f"wrote {clean_path}  rows={len(region)}", flush=True)
        noisy, noise_report = noise_region(region, cuts, args.scale, args.seed, args.label_flip)
        noisy_path = region_dir / f"{dataset}_v2_noisy.csv"
        noisy.to_csv(noisy_path, index=False)
        noise_report.insert(0, "file", clean_path.name)
        region_noise_rows.append(noise_report)
        print(f"wrote {noisy_path}  rows={len(noisy)}", flush=True)

        print(f"[{dataset}] window", flush=True)
        window, window_cut, window_cuts = build_windows(dataset)
        window_tables[dataset] = window
        window_cuts_rows.append(window_cut)
        clean_win = window_dir / f"{dataset}_windows.csv"
        window.to_csv(clean_win, index=False)
        print(f"wrote {clean_win}  rows={len(window)}", flush=True)
        noisy_win, win_noise = noise_windows(
            window, dataset, window_cuts, args.scale, args.seed, args.label_flip,
        )
        noisy_win_path = window_dir / f"{dataset}_windows_noisy.csv"
        noisy_win.to_csv(noisy_win_path, index=False)
        win_noise.insert(0, "file", clean_win.name)
        window_noise_rows.append(win_noise)
        print(f"wrote {noisy_win_path}  rows={len(noisy_win)}", flush=True)

    region_cut_report = pd.concat(region_cuts_rows, ignore_index=True)
    window_cut_report = pd.concat(window_cuts_rows, ignore_index=True)
    region_cut_report.to_csv(root / "region" / "cut_report.csv", index=False)
    window_cut_report.to_csv(root / "window" / "cut_report.csv", index=False)
    region_noise = pd.concat(region_noise_rows, ignore_index=True)
    window_noise = pd.concat(window_noise_rows, ignore_index=True)
    region_noise.to_csv(root / "region" / "noise_report.csv", index=False)
    window_noise.to_csv(root / "window" / "noise_report.csv", index=False)
    _write_region_sidecars(region_tables, root / "region")
    _write_window_sidecars(window_tables, root / "window")
    readme = root / "README.txt"
    readme.write_text(_readme())
    print(region_cut_report[["dataset", "motif", "n_finite", "frac_pos", "frac_neg"]].to_string(index=False), flush=True)
    print(window_cut_report[["dataset", "motif", "n_finite", "frac_pos", "frac_neg"]].to_string(index=False), flush=True)
    print(
        f"region mean flip rate {region_noise['flip_rate'].mean():.3f}  "
        f"window mean flip rate {window_noise['flip_rate'].mean():.3f}",
        flush=True,
    )
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
