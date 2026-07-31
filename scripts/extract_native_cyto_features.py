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
    """Parse image name.
    
        Args:
            result_path (Path): Path to a benchmark result file.
    
        Returns:
            str: The operation result.
    
    Args:
        result_path (Path): Path to a benchmark result file."""
    name = result_path.name
    prefix = "ResultTable_"
    suffix = ".csv"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected ResultTable filename: {result_path}")
    return name[len(prefix) : -len(suffix)]


def load_region_name_map(dataset_dir: Path) -> dict[str, str]:
    """Return image_name -> original region_id mapping when available.
    
    Args:
        dataset_dir (Path): Directory containing the source dataset."""
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
    """Execute the infer region id operation.
    
        Args:
            dataset (str): Dataset name used to filter benchmark records.
            image_name (str): Image identifier used to match native feature files.
            name_map (dict[str, str]): Mapping from source identifiers to canonical names.
    
        Returns:
            str: The operation result.
    
    Args:
        dataset (str): Dataset name used to filter benchmark records."""
    if image_name in name_map:
        return name_map[image_name]
    prefix = f"{dataset}__"
    if image_name.startswith(prefix):
        return image_name[len(prefix) :]
    return image_name


def entropy(proportions: list[float]) -> float:
    """Execute the entropy operation.
    
        Args:
            proportions (list[float]): Cell-type proportions used to derive community labels.
    
        Returns:
            float: The operation result.
    
    Args:
        proportions (list[float]): Cell-type proportions used to derive community labels."""
    return -sum(p * math.log(p) for p in proportions if p > 0)


def natural_tcn_sort(labels: list[int]) -> list[int]:
    """Execute the natural tcn sort operation.
    
        Args:
            labels (list[int]): Cell-type or class label assigned to each observation.
    
        Returns:
            list[int]: The operation result.
    
    Args:
        labels (list[int]): Cell-type or class label assigned to each observation."""
    return sorted(labels)


def extract_one_result(
    result_path: Path,
    dataset: str,
    image_name: str,
    region_id: str,
    tcn_labels: list[int],
) -> dict[str, object]:
    """Extract one result.
    
        Args:
            result_path (Path): Path to a benchmark result file.
            dataset (str): Dataset name used to filter benchmark records.
            image_name (str): Image identifier used to match native feature files.
            region_id (str): Unique identifier of a tissue region.
            tcn_labels (list[int]): Tissue cellular-neighborhood labels assigned to cells.
    
        Returns:
            dict[str, object]: The operation result.
    
    Args:
        result_path (Path): Path to a benchmark result file."""
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
    """Execute the discover tcn labels operation.
    
        Args:
            result_paths (list[Path]): Benchmark result files to combine.
    
        Returns:
            list[int]: The operation result.
    
    Args:
        result_paths (list[Path]): Benchmark result files to combine."""
    labels: set[int] = set()
    for path in result_paths:
        df = pd.read_csv(path, usecols=["TCN_Label"])
        values = pd.to_numeric(df["TCN_Label"], errors="coerce").dropna().astype(int)
        labels.update(values.unique().tolist())
    if not labels:
        raise ValueError("No TCN labels found in ResultTable files.")
    return natural_tcn_sort(list(labels))


def extract_features(native_root: Path, output: Path, num_tcn: int | None = None) -> pd.DataFrame:
    """Extract features.
    
        Args:
            native_root (Path): Root directory containing native CytoCommunity outputs.
            output (Path): Destination path for the generated result file.
            num_tcn (int | None): Number of tcn.
    
        Returns:
            pd.DataFrame: The operation result.
    
    Args:
        native_root (Path): Root directory containing native CytoCommunity outputs."""
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
    """Execute the main operation."""
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
