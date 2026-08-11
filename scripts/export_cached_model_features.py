#!/usr/bin/env python
"""Aggregate cached model outputs into a region-by-feature CSV for Lasso.

This exporter is for representations that are already frozen or were computed
independently per region:

* ``kronos`` / ``eva``: stack the fixed-length ``feature`` arrays in NPZ files;
* ``utag-message-passing``: aggregate cell-by-marker ``X`` as mean and std;
* ``cytocommunity``: aggregate cell-level ``TCN_Label`` as region proportions.

The output contains one ``region_id`` column followed by numeric feature
columns and can be passed to ``scripts/run_stability_lasso.py`` with
``--feature-source precomputed``.

Do not use this exporter for UTAG ``domains`` or ``combined`` in confirmatory
CV. Their scaler and KMeans model must be fitted inside each training fold; use
``run_stability_lasso.py --feature-source utag`` instead.

Examples
--------
python scripts/export_cached_model_features.py \
  --source utag-message-passing --dataset hnc_wu2022 \
  --input-root model_results/UTAG/message_passing_cache/HNC-Wu2022/<signature>/<markers> \
  --output results/features/hnc_wu2022_utag_message_passing.csv

python scripts/export_cached_model_features.py \
  --source kronos --dataset hnc_wu2022 \
  --input-root model_results/KRONOS/embeddings/rasterized/HNC-Wu2022/<signature> \
  --output results/features/hnc_wu2022_kronos.csv

python scripts/export_cached_model_features.py \
  --source cytocommunity --dataset hnc_wu2022 \
  --input-root model_results/CytoCommunity/native_local_runs_cutoff02/hnc_wu2022 \
  --output results/features/hnc_wu2022_cytocommunity.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import load_dataset


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:200] or "region"


def _region_lookup(region_ids: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for region_id in map(str, region_ids):
        safe = _safe_name(region_id)
        previous = lookup.get(safe)
        if previous is not None and previous != region_id:
            raise ValueError(
                f"Region IDs {previous!r} and {region_id!r} share safe name {safe!r}"
            )
        lookup[safe] = region_id
    return lookup


def _select_npz(
    input_root: Path,
    region_lookup: dict[str, str],
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(input_root.rglob("*.npz")):
        region_id = region_lookup.get(path.stem)
        if region_id is None:
            continue
        if region_id in selected:
            duplicates.setdefault(region_id, [selected[region_id]]).append(path)
        else:
            selected[region_id] = path
    if duplicates:
        example_id, paths = next(iter(duplicates.items()))
        rendered = "\n".join(f"  - {path}" for path in paths[:5])
        raise ValueError(
            "Multiple cache variants matched the same region. Point --input-root "
            f"at one exact mode/signature directory. Example {example_id!r}:\n{rendered}"
        )
    return selected


def _validate_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        raise ValueError("No features were exported")
    if not table.index.is_unique:
        raise ValueError("Exported region IDs are not unique")
    if not table.columns.is_unique:
        raise ValueError("Exported feature names are not unique")
    table = table.apply(pd.to_numeric, errors="raise")
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"Exported feature table contains {bad} non-finite value(s)")
    table.index = table.index.astype(str)
    table.index.name = "region_id"
    return table


def aggregate_embeddings(
    paths: dict[str, Path],
    prefix: str,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    region_ids: list[str] = []
    width: int | None = None
    for region_id, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as archive:
            if "feature" not in archive.files:
                raise ValueError(f"{path} has no 'feature' array")
            feature = np.asarray(archive["feature"], dtype=np.float32).reshape(-1)
        if width is None:
            width = len(feature)
        elif len(feature) != width:
            raise ValueError(
                f"Inconsistent embedding width: {path} has {len(feature)}, expected {width}"
            )
        region_ids.append(region_id)
        rows.append(feature)
    if width is None:
        return pd.DataFrame()
    columns = [f"{prefix}_{index:04d}" for index in range(width)]
    return pd.DataFrame(rows, index=region_ids, columns=columns)


def aggregate_utag_message_passing(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    region_ids: list[str] = []
    expected_markers: list[str] | None = None
    for region_id, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as archive:
            if "X" not in archive.files or "markers" not in archive.files:
                raise ValueError(f"{path} must contain 'X' and 'markers'")
            values = np.asarray(archive["X"], dtype=np.float32)
            markers = [str(marker) for marker in archive["markers"].tolist()]
        if values.ndim != 2 or values.shape[1] != len(markers):
            raise ValueError(
                f"{path} has incompatible X shape {values.shape} and {len(markers)} markers"
            )
        if expected_markers is None:
            expected_markers = markers
        elif markers != expected_markers:
            raise ValueError(
                "UTAG marker panels differ across caches. Point --input-root at "
                f"one exact marker-hash directory; first mismatch: {path}"
            )
        rows.append(np.concatenate([values.mean(0), values.std(0)]))
        region_ids.append(region_id)
    if expected_markers is None:
        return pd.DataFrame()
    columns = (
        [f"utag_mean__{marker}" for marker in expected_markers]
        + [f"utag_std__{marker}" for marker in expected_markers]
    )
    return pd.DataFrame(rows, index=region_ids, columns=columns)


def _cytocommunity_region_id(path: Path, lookup: dict[str, str]) -> str | None:
    stem = path.stem
    if "__" in stem:
        safe = _safe_name(stem.split("__", 1)[1])
        if safe in lookup:
            return lookup[safe]
    for part in reversed(path.parts):
        if "__" not in part:
            continue
        safe = _safe_name(part.split("__", 1)[1])
        if safe in lookup:
            return lookup[safe]
    return None


def aggregate_cytocommunity(
    input_root: Path,
    lookup: dict[str, str],
) -> pd.DataFrame:
    assignments: dict[str, pd.Series] = {}
    labels: set[str] = set()
    for path in sorted(input_root.rglob("ResultTable_*.csv")):
        region_id = _cytocommunity_region_id(path, lookup)
        if region_id is None:
            continue
        if region_id in assignments:
            raise ValueError(
                f"Multiple CytoCommunity ResultTable files matched region {region_id!r}; "
                "point --input-root at one exact run directory"
            )
        frame = pd.read_csv(path)
        normalized = {
            re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
            for column in frame.columns
        }
        label_column = next(
            (
                normalized[key]
                for key in ("tcnlabel", "tcn", "communitylabel")
                if key in normalized
            ),
            None,
        )
        if label_column is None:
            raise ValueError(f"{path} has no TCN_Label column")
        values = frame[label_column].dropna().astype(str)
        if values.empty:
            raise ValueError(f"{path} contains no non-null TCN labels")
        counts = values.value_counts(normalize=True)
        assignments[region_id] = counts
        labels.update(counts.index)
    ordered_labels = sorted(labels, key=lambda value: (not value.isdigit(), value))
    table = pd.DataFrame(
        {
            region_id: [float(counts.get(label, 0.0)) for label in ordered_labels]
            for region_id, counts in assignments.items()
        },
        index=[f"tcn_fraction__{_safe_name(label)}" for label in ordered_labels],
    ).T
    return table


def export(args: argparse.Namespace) -> pd.DataFrame:
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    dataset = load_dataset(args.dataset, data_root=args.data_root)
    region_ids = dataset.get_metadata()["region_id"].astype(str).tolist()
    lookup = _region_lookup(region_ids)

    if args.source == "cytocommunity":
        table = aggregate_cytocommunity(input_root, lookup)
    else:
        paths = _select_npz(input_root, lookup)
        if args.source == "utag-message-passing":
            table = aggregate_utag_message_passing(paths)
        else:
            table = aggregate_embeddings(paths, args.source)

    table = _validate_table(table)
    missing = sorted(set(region_ids) - set(table.index))
    print(
        f"{args.dataset}: exported={len(table)}, metadata={len(region_ids)}, "
        f"missing={len(missing)}, features={table.shape[1]}",
        flush=True,
    )
    if args.strict and missing:
        raise ValueError(
            f"{len(missing)} metadata regions have no exported features; "
            f"examples: {missing[:10]}"
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(output, index=False)
    print(f"Wrote {output}", flush=True)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["utag-message-passing", "kronos", "eva", "cytocommunity"],
    )
    parser.add_argument("--dataset", required=True, help="Dataset registry name.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--input-root",
        required=True,
        help="One exact cache mode/signature/run directory.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless every metadata region has a feature row.",
    )
    return parser.parse_args()


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
