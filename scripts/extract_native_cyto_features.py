#!/usr/bin/env python
"""Extract region-level features from native CytoCommunity Step4 outputs.

Native CytoCommunity writes one ``ResultTable_<image>.csv`` per image/region.
Each table is cell-level and contains a ``TCN_Label`` column. This script turns
those cell-level labels into one fixed-length feature row per image/region.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def parse_image_name(result_path: Path) -> str:
    name = result_path.name
    prefix = "ResultTable_"
    suffix = ".csv"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected ResultTable filename: {result_path}")
    return name[len(prefix) : -len(suffix)]


def load_region_name_map(dataset_dir: Path) -> dict[str, str]:
    """Return image_name -> original region_id mapping when available."""
    candidates = [
        dataset_dir / "MERFISH-Brain_Input" / "region_name_map.csv",
        dataset_dir / "region_name_map.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        mapping = pd.read_csv(path)
        cols = set(mapping.columns)
        if {"image_name", "region_id"}.issubset(cols):
            return dict(zip(mapping["image_name"].astype(str), mapping["region_id"].astype(str)))
        if {"safe_image_name", "region_id"}.issubset(cols):
            return dict(zip(mapping["safe_image_name"].astype(str), mapping["region_id"].astype(str)))
    return {}


def infer_region_id(dataset: str, image_name: str, name_map: dict[str, str]) -> str:
    if image_name in name_map:
        return name_map[image_name]
    prefix = f"{dataset}__"
    if image_name.startswith(prefix):
        return image_name[len(prefix) :]
    return image_name


def entropy(proportions: list[float]) -> float:
    return -sum(p * math.log(p) for p in proportions if p > 0)


def natural_tcn_sort(labels: list[int]) -> list[int]:
    return sorted(labels)


def extract_one_result(
    result_path: Path,
    dataset: str,
    image_name: str,
    region_id: str,
    tcn_labels: list[int],
) -> dict[str, object]:
    df = pd.read_csv(result_path)
    if "TCN_Label" not in df.columns:
        raise ValueError(f"Missing TCN_Label column in {result_path}")

    labels = pd.to_numeric(df["TCN_Label"], errors="coerce").dropna().astype(int)
    n_cells = int(len(labels))
    counts = labels.value_counts().to_dict()

    row: dict[str, object] = {
        "dataset": dataset,
        "image_name": image_name,
        "region_id": region_id,
        "result_table": str(result_path),
        "n_cells": n_cells,
    }

    fractions: list[float] = []
    for label in tcn_labels:
        count = int(counts.get(label, 0))
        frac = count / n_cells if n_cells else 0.0
        row[f"tcn_{label}_count"] = count
        row[f"tcn_{label}_frac"] = frac
        fractions.append(frac)

    if n_cells:
        dominant_label = max(tcn_labels, key=lambda label: (counts.get(label, 0), -label))
    else:
        dominant_label = ""
    row["tcn_entropy"] = entropy(fractions)
    row["dominant_tcn"] = dominant_label
    row["observed_tcn_labels"] = ";".join(str(x) for x in natural_tcn_sort(list(counts)))
    return row


def discover_tcn_labels(result_paths: list[Path]) -> list[int]:
    labels: set[int] = set()
    for path in result_paths:
        df = pd.read_csv(path, usecols=["TCN_Label"])
        values = pd.to_numeric(df["TCN_Label"], errors="coerce").dropna().astype(int)
        labels.update(values.unique().tolist())
    if not labels:
        raise ValueError("No TCN labels found in ResultTable files.")
    return natural_tcn_sort(list(labels))


def extract_features(native_root: Path, output: Path, num_tcn: int | None = None) -> pd.DataFrame:
    if not native_root.exists():
        raise FileNotFoundError(f"Native root not found: {native_root}")

    dataset_dirs = sorted(
        path for path in native_root.iterdir()
        if path.is_dir() and any(path.glob("Step4_Output_*/ResultTable_*.csv"))
    )
    if not dataset_dirs:
        raise ValueError(f"No dataset folders with Step4 ResultTable files found under {native_root}")

    all_results = [
        result
        for dataset_dir in dataset_dirs
        for result in sorted(dataset_dir.glob("Step4_Output_*/ResultTable_*.csv"))
    ]
    tcn_labels = list(range(1, num_tcn + 1)) if num_tcn else discover_tcn_labels(all_results)

    rows: list[dict[str, object]] = []
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        name_map = load_region_name_map(dataset_dir)
        result_paths = sorted(dataset_dir.glob("Step4_Output_*/ResultTable_*.csv"))
        print(f"[{dataset}] extracting {len(result_paths)} ResultTable file(s)")
        for result_path in result_paths:
            image_name = parse_image_name(result_path)
            region_id = infer_region_id(dataset, image_name, name_map)
            rows.append(extract_one_result(result_path, dataset, image_name, region_id, tcn_labels))

    features = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-root",
        required=True,
        help="Root containing native CytoCommunity dataset folders with Step4_Output_* directories.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for region-level TCN composition features.",
    )
    parser.add_argument(
        "--num-tcn",
        type=int,
        default=None,
        help="Expected number of TCN labels. If omitted, labels are inferred from all ResultTable files.",
    )
    args = parser.parse_args()

    features = extract_features(Path(args.native_root), Path(args.output), num_tcn=args.num_tcn)
    print(f"Wrote {args.output} ({len(features)} rows, {len(features.columns)} columns)")


if __name__ == "__main__":
    main()
