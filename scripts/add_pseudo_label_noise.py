#!/usr/bin/env python
"""Add modest Gaussian noise to motif scores and recut v2 pseudo-labels.

Does not overwrite the clean ``*_v2.csv``. Writes ``*_v2_noisy.csv`` plus a
flip report. Noise is ``N(0, scale * sd(score_used))`` with default
``scale=0.10`` (about 10% of the score spread). Original median / tertile
cuts are kept, so this is measurement error around a fixed boundary.

    python scripts/add_pseudo_label_noise.py \
        --input-dir results/pseudo_labels \
        --scale 0.10 --label-flip 0.05 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_MOTIFS = ("tumor_high", "cd8_high")


def _motif_ids(table: pd.DataFrame) -> list[str]:
    ids = []
    for col in table.columns:
        if col.endswith("_label_v2"):
            ids.append(col[: -len("_label_v2")])
    return ids


def _original_cuts(scores: pd.Series, control: bool) -> tuple[float, float]:
    values = scores.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if control:
        mid = float(np.median(values))
        return mid, mid
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(lo), float(hi)


def _recut(noisy: pd.Series, lo: float, hi: float, control: bool) -> pd.Series:
    out = pd.Series(np.nan, index=noisy.index, dtype="float")
    valid = noisy.notna()
    if not valid.any() or not np.isfinite(lo):
        return out
    if control:
        out.loc[valid] = (noisy.loc[valid] >= lo).astype(float)
        return out
    if not np.isfinite(hi) or lo == hi:
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


def add_noise(
    table: pd.DataFrame,
    scale: float,
    seed: int,
    label_flip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        lo, hi = _original_cuts(scores, motif in CONTROL_MOTIFS)
        new_label = _recut(noisy, lo, hi, motif in CONTROL_MOTIFS)
        new_label = _symmetric_flip(new_label, label_flip, rng)

        out[f"{motif}_score_used_clean"] = scores
        out[f"{motif}_label_v2_clean"] = out[label_col]
        out[score_col] = noisy
        out[label_col] = new_label

        old = pd.to_numeric(table[label_col], errors="coerce")
        both = old.notna() & new_label.notna()
        n_old = int(old.notna().sum())
        n_new = int(new_label.notna().sum())
        n_flip = int((both & (old != new_label)).sum())
        n_gained = int((old.isna() & new_label.notna()).sum())
        n_lost = int((old.notna() & new_label.isna()).sum())
        rows.append(dict(
            motif=motif,
            kind="control" if motif in CONTROL_MOTIFS else "spatial",
            sigma=sigma,
            score_sd=sd,
            n_labeled_clean=n_old,
            n_labeled_noisy=n_new,
            n_flipped=n_flip,
            flip_rate=n_flip / n_old if n_old else float("nan"),
            n_gained_label=n_gained,
            n_lost_label=n_lost,
            cut_low=lo,
            cut_high=hi,
        ))
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="results/pseudo_labels")
    ap.add_argument("--pattern", default="*_v2.csv")
    ap.add_argument("--scale", type=float, default=0.10,
                    help="Gaussian SD as a fraction of score_used SD (default 0.10).")
    ap.add_argument(
        "--label-flip",
        type=float,
        default=0.05,
        help="Extra symmetric 0/1 flip rate on recut labels (default 0.05). "
             "Needed for spatial v2: confident tertiles rarely cross under score noise alone.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.input_dir)
    paths = sorted(
        p for p in root.glob(args.pattern)
        if not p.name.endswith("_noisy.csv") and "noise_report" not in p.name
    )
    if not paths:
        raise SystemExit(f"no files matching {root}/{args.pattern}")

    reports = []
    for path in paths:
        table = pd.read_csv(path)
        noisy, report = add_noise(
            table, scale=args.scale, seed=args.seed, label_flip=args.label_flip
        )
        out_path = path.with_name(path.stem + "_noisy.csv")
        noisy.to_csv(out_path, index=False)
        report.insert(0, "file", path.name)
        reports.append(report)
        print(f"Wrote {out_path}  ({len(noisy)} rows)", flush=True)
        print(report[["motif", "n_flipped", "flip_rate", "n_gained_label", "n_lost_label"]].to_string(index=False), flush=True)

    all_report = pd.concat(reports, ignore_index=True)
    report_path = root / "noise_report.csv"
    all_report.to_csv(report_path, index=False)
    print(f"Wrote {report_path}", flush=True)
    print(
        f"mean flip rate among originally labeled = {all_report['flip_rate'].mean():.3f} "
        f"(scale={args.scale}, label_flip={args.label_flip}, seed={args.seed})",
        flush=True,
    )


if __name__ == "__main__":
    main()
