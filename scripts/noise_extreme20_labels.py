#!/usr/bin/env python
"""Add the standard pseudo-label noise to PseudoLabelDatasetExtreme20.

Same protocol as ``add_pseudo_label_noise.py`` and
``add_mil_window_label_noise.py``: ``N(0, 0.10 * SD(score_used))``, recut with
the clean separator, then a 5% symmetric label flip.

Controls are recut at the median. Spatial motifs are recut at the 20% / 80%
score quantiles (the extreme20 separator), not the old tertiles. Region cuts
and SDs use every finite score. Window cuts and SDs use the discovery cohort.

    python scripts/noise_extreme20_labels.py
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
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

PACK = Path("/autofs/nas8/tywang/tjzou/PseudoLabelDatasetExtreme20")
CONTROL_MOTIFS = ("tumor_high", "cd8_high")
LOW_Q = 0.20
HIGH_Q = 0.80


def _motif_ids(table: pd.DataFrame) -> list[str]:
    ids = []
    for col in table.columns:
        if col.endswith("_label_v2") and not col.endswith("_label_v2_original"):
            ids.append(col[: -len("_label_v2")])
    return ids


def _cuts(scores: pd.Series, control: bool) -> tuple[float, float]:
    values = scores.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if control:
        mid = float(np.median(values))
        return mid, mid
    lo, hi = np.quantile(values, [LOW_Q, HIGH_Q])
    return float(lo), float(hi)


def _recut(noisy: pd.Series, lo: float, hi: float, control: bool) -> pd.Series:
    out = pd.Series(np.nan, index=noisy.index, dtype="float")
    valid = noisy.notna()
    if not valid.any() or not np.isfinite(lo):
        return out
    if control or not np.isfinite(hi) or lo == hi:
        out.loc[valid] = (noisy.loc[valid] >= lo).astype(float)
        return out
    out.loc[noisy <= lo] = 0.0
    out.loc[noisy >= hi] = 1.0
    return out


def _symmetric_flip(labels: pd.Series, rate: float, rng: np.random.Generator) -> pd.Series:
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


def _report_row(motif: str, old: pd.Series, new: pd.Series, sigma: float, sd: float, lo: float, hi: float) -> dict:
    both = old.notna() & new.notna()
    n_old = int(old.notna().sum())
    n_flip = int((both & (old != new)).sum())
    return dict(
        motif=motif,
        kind="control" if motif in CONTROL_MOTIFS else "spatial",
        sigma=sigma,
        score_sd=sd,
        n_labeled_clean=n_old,
        n_labeled_noisy=int(new.notna().sum()),
        n_flipped=n_flip,
        flip_rate=n_flip / n_old if n_old else float("nan"),
        n_gained_label=int((old.isna() & new.notna()).sum()),
        n_lost_label=int((old.notna() & new.isna()).sum()),
        cut_low=lo,
        cut_high=hi,
    )


def add_region_noise(table: pd.DataFrame, scale: float, seed: int, label_flip: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = table.copy()
    rows = []
    for motif in _motif_ids(table):
        score_col = f"{motif}_score_used"
        label_col = f"{motif}_label_v2"
        if score_col not in out.columns:
            continue
        scores = pd.to_numeric(out[score_col], errors="coerce")
        sd = float(np.nanstd(scores.to_numpy(dtype=float)))
        sigma = scale * sd if np.isfinite(sd) and sd > 0 else 0.0
        noise = rng.normal(0.0, sigma, size=len(scores))
        noise = np.where(scores.notna().to_numpy(), noise, np.nan)
        noisy = scores + noise
        control = motif in CONTROL_MOTIFS
        lo, hi = _cuts(scores, control)
        new_label = _symmetric_flip(_recut(pd.Series(noisy, index=scores.index), lo, hi, control), label_flip, rng)
        out[f"{motif}_score_used_clean"] = scores
        out[f"{motif}_label_v2_clean"] = pd.to_numeric(out[label_col], errors="coerce")
        out[score_col] = noisy
        out[label_col] = new_label
        rows.append(_report_row(motif, out[f"{motif}_label_v2_clean"], new_label, sigma, sd, lo, hi))
    return out, pd.DataFrame(rows)


def add_window_noise(
    table: pd.DataFrame,
    dataset: str,
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
        scores = pd.to_numeric(part["score_used"], errors="coerce")
        fit = fit_all.reindex(part.index).fillna(False)
        fit_scores = scores.loc[fit]
        pool = fit_scores if fit_scores.notna().any() else scores
        control = str(motif) in CONTROL_MOTIFS
        lo, hi = _cuts(pool, control)
        sd = float(np.nanstd(pool.to_numpy(dtype=float)))
        if not np.isfinite(sd) or sd == 0:
            sd = float(np.nanstd(scores.to_numpy(dtype=float)))
        sigma = scale * sd if np.isfinite(sd) and sd > 0 else 0.0
        noise = rng.normal(0.0, sigma, size=len(scores))
        noise = np.where(scores.notna().to_numpy(), noise, np.nan)
        noisy = scores + noise
        old = pd.to_numeric(part["label"], errors="coerce")
        new_label = _symmetric_flip(
            _recut(pd.Series(noisy, index=part.index), lo, hi, control),
            label_flip,
            rng,
        )
        part["score_clean"] = scores
        part["label_clean"] = old
        part["score_used"] = noisy
        part["score"] = noisy
        part["label"] = new_label
        part["pos"] = (new_label == 1.0).astype(float)
        part["neg"] = (new_label == 0.0).astype(float)
        part.loc[new_label.isna(), ["pos", "neg"]] = np.nan
        rows.append(_report_row(str(motif), old, new_label, sigma, sd, lo, hi))
        parts.append(part)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", default=str(PACK))
    p.add_argument("--scale", type=float, default=0.10)
    p.add_argument("--label-flip", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    root = Path(args.pack)
    region_reports = []
    for path in sorted((root / "region" / "per_dataset").glob("*_v2.csv")):
        if path.name.endswith("_noisy.csv"):
            continue
        table = pd.read_csv(path)
        noisy, report = add_region_noise(table, args.scale, args.seed, args.label_flip)
        dest = path.with_name(path.stem + "_noisy.csv")
        noisy.to_csv(dest, index=False)
        report.insert(0, "file", path.name)
        report.insert(0, "level", "region")
        region_reports.append(report)
        print(f"wrote {dest}  rows={len(noisy)}", flush=True)
    region_report = pd.concat(region_reports, ignore_index=True)
    region_report.to_csv(root / "region" / "noise_report.csv", index=False)

    window_reports = []
    for path in sorted((root / "window" / "per_dataset").glob("*_windows.csv")):
        if path.name.endswith("_noisy.csv"):
            continue
        dataset = path.name.replace("_windows.csv", "")
        table = pd.read_csv(path)
        noisy, report = add_window_noise(table, dataset, args.scale, args.seed, args.label_flip)
        dest = path.with_name(path.stem + "_noisy.csv")
        noisy.to_csv(dest, index=False)
        report.insert(0, "file", path.name)
        report.insert(0, "level", "window")
        window_reports.append(report)
        print(f"wrote {dest}  rows={len(noisy)}", flush=True)
    window_report = pd.concat(window_reports, ignore_index=True)
    window_report.to_csv(root / "window" / "noise_report.csv", index=False)
    note = (
        "\nNoisy copies sit next to the clean tables (*_v2_noisy.csv, *_windows_noisy.csv).\n"
        "Noise is N(0, 0.10 * SD(score_used)), then the label is recut with the clean\n"
        "separator (control median, spatial 20/80 quantiles), then 5% of labeled rows\n"
        "are flipped. Region SD and cuts use all finite scores. Window SD and cuts use\n"
        "the discovery cohort. See region/noise_report.csv and window/noise_report.csv.\n"
    )
    readme = root / "README.txt"
    text = readme.read_text() if readme.exists() else ""
    if "Noisy copies sit next to the clean tables" not in text:
        readme.write_text(text + note)
    print(region_report.loc[region_report["kind"] == "control", ["file", "motif", "n_labeled_clean", "n_labeled_noisy", "flip_rate"]].to_string(index=False), flush=True)
    print(window_report[["file", "motif", "kind", "n_labeled_clean", "n_labeled_noisy", "flip_rate"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
