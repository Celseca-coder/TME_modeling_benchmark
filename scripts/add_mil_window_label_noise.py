#!/usr/bin/env python
"""Add region-style noise to MIL window pseudo-labels (long tables).

Does not overwrite the clean ``mil_windows/<dataset>.csv``. Writes
``<dataset>_noisy.csv`` plus a flip report. Same protocol as
``add_pseudo_label_noise.py``: ``N(0, scale * SD(score_used))``, recut with
the original discovery median / tertile cuts, then a symmetric label flip.

Cuts and SD are estimated on discovery windows (catalog ``cv_filter``) so they
match ``add_window_pseudo_labels``. Control motifs use the median; spatial
motifs use tertile extremes.

    python scripts/add_mil_window_label_noise.py \
        --input-dir results/pseudo_labels/mil_windows \
        --pack-dir /autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows \
        --scale 0.10 --label-flip 0.05 --seed 0
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
from benchmark.motifs.source_select import selected_motifs_for  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

CONTROL_MOTIFS = ("tumor_high", "cd8_high")
DATASETS = (
    "hnc_wu2022",
    "bc_jackson2020",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)
DATASET_NAMES = {
    "hnc_wu2022": "HNC-Wu2022",
    "bc_jackson2020": "BC-Jackson2020",
    "bc_metabric_ali2020": "BC-METABRIC",
    "tnbc_wang2023": "TNBC-Wang2023",
}
PACK_COLS = [
    "region_id", "window_id", "cx_um", "cy_um", "n_cells", "motif_id",
    "score", "pos", "neg", "label",
    "score_clean", "label_clean", "pos_clean", "neg_clean",
]


def _score_series(part: pd.DataFrame) -> pd.Series:
    col = "score_used" if "score_used" in part.columns else "score"
    return pd.to_numeric(part[col], errors="coerce")


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
    if control or (not np.isfinite(hi) or lo == hi):
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


def add_window_noise(
    table: pd.DataFrame,
    *,
    dataset: str,
    scale: float,
    seed: int,
    label_flip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = load_motif_catalog(dataset=dataset)
    fit_all = discovery_mask(table, catalog.cv_filter)
    rng = np.random.default_rng(seed)
    frames = []
    rows = []
    for motif, part in table.groupby("motif_id", sort=False):
        part = part.copy()
        scores = _score_series(part)
        fit = fit_all.reindex(part.index).fillna(False)
        fit_scores = scores.loc[fit]
        control = motif in CONTROL_MOTIFS
        lo, hi = _original_cuts(fit_scores if fit_scores.notna().any() else scores, control)
        sd = float(np.nanstd(fit_scores.to_numpy(dtype=float)))
        if not np.isfinite(sd) or sd == 0:
            sd = float(np.nanstd(scores.to_numpy(dtype=float)))
        sigma = scale * sd if np.isfinite(sd) and sd > 0 else 0.0
        noise = rng.normal(0.0, sigma, size=len(scores))
        noise = np.where(scores.notna().to_numpy(), noise, np.nan)
        noisy = scores + noise
        new_label = _recut(pd.Series(noisy, index=part.index), lo, hi, control)
        new_label = _symmetric_flip(new_label, label_flip, rng)

        part["score_clean"] = scores
        part["label_clean"] = pd.to_numeric(part.get("label"), errors="coerce")
        part["pos_clean"] = pd.to_numeric(part.get("pos"), errors="coerce")
        part["neg_clean"] = pd.to_numeric(part.get("neg"), errors="coerce")
        part["score_used"] = noisy
        part["score"] = noisy
        part["label"] = new_label
        part["pos"] = (new_label == 1.0).astype(float)
        part["neg"] = (new_label == 0.0).astype(float)
        part.loc[new_label.isna(), ["pos", "neg"]] = np.nan

        old = part["label_clean"]
        both = old.notna() & new_label.notna()
        n_old = int(old.notna().sum())
        n_flip = int((both & (old != new_label)).sum())
        rows.append(dict(
            motif=motif,
            kind="control" if control else "spatial",
            sigma=sigma,
            score_sd=sd,
            n_windows=len(part),
            n_labeled_clean=n_old,
            n_labeled_noisy=int(new_label.notna().sum()),
            n_flipped=n_flip,
            flip_rate=n_flip / n_old if n_old else float("nan"),
            n_gained_label=int((old.isna() & new_label.notna()).sum()),
            n_lost_label=int((old.notna() & new_label.isna()).sum()),
            cut_low=lo,
            cut_high=hi,
        ))
        frames.append(part)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def _dataset_from_stem(stem: str) -> str:
    name = stem.removesuffix("_windows").removesuffix("_noisy")
    return name


def _pack_selected(noisy_by_ds: dict[str, pd.DataFrame], pack_dir: Path) -> None:
    catalog_path = pack_dir.parent / "selected_catalog.csv"
    if not catalog_path.exists():
        catalog_path = pack_dir / "selected_catalog.csv"
    if not catalog_path.exists():
        print(f"No selected_catalog.csv at {catalog_path}; skip selected pack.", flush=True)
        return
    catalog = pd.read_csv(catalog_path)
    frames = []
    summary = []
    for rec in catalog.itertuples(index=False):
        ds = str(rec.source_dataset)
        motif = str(rec.motif_id)
        table = noisy_by_ds.get(ds)
        if table is None:
            continue
        part = table[table["motif_id"].astype(str) == motif].copy()
        if part.empty:
            continue
        labeled = part[part["label"].notna()].copy()
        labeled.insert(0, "source_dataset", ds)
        labeled.insert(1, "source_dataset_name", getattr(rec, "source_dataset_name", DATASET_NAMES.get(ds, ds)))
        labeled["task_type"] = getattr(rec, "task_type", "")
        labeled["panel"] = getattr(rec, "panel", "")
        labeled["pseudo_label"] = labeled["label"]
        labeled["label_column"] = "label"
        keep = [
            "source_dataset", "source_dataset_name", "region_id", "window_id",
            "cx_um", "cy_um", "n_cells", "motif_id", "task_type", "panel",
            "score_used", "pseudo_label", "label_column", "pos", "neg",
            "score_clean", "label_clean", "pos_clean", "neg_clean",
        ]
        keep = [c for c in keep if c in labeled.columns]
        frames.append(labeled[keep])
        n1 = int((labeled["label"] == 1).sum())
        n0 = int((labeled["label"] == 0).sum())
        summary.append({
            "motif_id": motif,
            "source_dataset": ds,
            "source_dataset_name": getattr(rec, "source_dataset_name", DATASET_NAMES.get(ds, ds)),
            "eligible": True,
            "task_type": getattr(rec, "task_type", ""),
            "panel": getattr(rec, "panel", ""),
            "n_labeled": n0 + n1,
            "n_0": n0,
            "n_1": n1,
            "definition": getattr(rec, "definition", ""),
        })
    if frames:
        selected = pd.concat(frames, ignore_index=True)
        dest = pack_dir / "selected_window_labels.csv"
        selected.to_csv(dest, index=False)
        print(f"Wrote {dest}  ({len(selected)} rows)", flush=True)
    if summary:
        dest = pack_dir / "selected_catalog.csv"
        pd.DataFrame(summary).to_csv(dest, index=False)
        print(f"Wrote {dest}", flush=True)


def pack_windows(
    noisy_by_ds: dict[str, pd.DataFrame],
    reports: list[pd.DataFrame],
    pack_dir: Path,
) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    dest_ds = pack_dir / "per_dataset"
    dest_ds.mkdir(parents=True, exist_ok=True)
    for ds, table in noisy_by_ds.items():
        cols = [c for c in PACK_COLS if c in table.columns]
        extra = [c for c in ("score_used",) if c in table.columns and c not in cols]
        out = table[cols + extra]
        path = dest_ds / f"{ds}_windows_noisy.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {path}  ({len(out)} rows)", flush=True)
    report = pd.concat(reports, ignore_index=True)
    report_path = pack_dir / "noise_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Wrote {report_path}", flush=True)
    _pack_selected(noisy_by_ds, pack_dir)
    readme = pack_dir / "README.txt"
    readme.write_text(
        "MIL window-level noisy motif GT "
        "(score noise scale=0.10 * SD on discovery score_used + 5% symmetric flip).\n"
        "Cuts match add_window_pseudo_labels (median / tertile_extremes on discovery).\n"
        "Does not replace region bag labels in ../per_dataset/*_v2_noisy.csv.\n"
        "per_dataset/*_windows_noisy.csv : selected motif×dataset pairs only.\n"
        "selected_window_labels.csv : labeled windows for the 10 motif×source pairs.\n"
        "Clean originals: TME_modeling_benchmark/results/pseudo_labels/mil_windows/*.csv\n"
    )
    print(f"Wrote {readme}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="results/pseudo_labels/mil_windows")
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument(
        "--pack-dir",
        default="/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows",
        help="PseudoNoisyDataset window pack root (region analog of per_dataset/).",
    )
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--label-flip", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.input_dir)
    paths = sorted(
        p for p in root.glob(args.pattern)
        if p.is_file()
        and not p.name.endswith("_noisy.csv")
        and "noise_report" not in p.name
        and _dataset_from_stem(p.stem) in DATASETS
    )
    if not paths:
        raise SystemExit(f"no files matching {root}/{args.pattern}")

    reports = []
    noisy_by_ds: dict[str, pd.DataFrame] = {}
    for path in paths:
        dataset = _dataset_from_stem(path.stem)
        print(f"reading {path}", flush=True)
        table = pd.read_csv(path)
        keep = selected_motifs_for(dataset)
        if keep:
            table = table[table["motif_id"].astype(str).isin(keep)].copy()
        if table.empty:
            print(f"skip {dataset}: no selected motif rows", flush=True)
            continue
        print(f"  kept motifs {keep}  ({len(table)} rows)", flush=True)
        noisy, report = add_window_noise(
            table, dataset=dataset, scale=args.scale, seed=args.seed,
            label_flip=args.label_flip,
        )
        out_path = path.with_name(f"{dataset}_noisy.csv")
        noisy.to_csv(out_path, index=False)
        report.insert(0, "file", path.name)
        reports.append(report)
        noisy_by_ds[dataset] = noisy
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
    if not args.no_pack:
        pack_windows(noisy_by_ds, reports, Path(args.pack_dir))


if __name__ == "__main__":
    main()
