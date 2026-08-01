#!/usr/bin/env python
"""Benchmark all C(6, 2)=15 pairwise combinations of region-level features.

The six groups are composition, density, expression, distance, point_pattern,
and mixing. Classification uses balanced L2 logistic regression; survival uses
ridge-penalized Cox regression, matching the individual baseline scripts.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.basic_feats import CompositionFeaturizer, MeanExpressionFeaturizer
from benchmark.features.combined import CombinedFeaturizer
from benchmark.features.density_feats import CellTypeDensityFeaturizer
from benchmark.features.mixing import MixingFeaturizer
from benchmark.features.point_pattern import PointPatternFeaturizer
from benchmark.features.spatial_distance import SpatialDistanceFeaturizer
from benchmark.models.linear import LinearClassifier, LinearCox
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation import (
    PRIMARY_METRIC,
    cohort_split_test,
    cross_validate,
    summarize_folds,
)

GROUPS = ("composition", "density", "expression", "distance", "point_pattern", "mixing")
COMBINATIONS = tuple("+".join(pair) for pair in itertools.combinations(GROUPS, 2))


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def make_featurizer(groups, cell_type_col="cell_type", fixed_markers=None, point_metrics=None):
    """Create one split-local pair of extractors using the baseline defaults."""
    factories = {
        "composition": lambda: CompositionFeaturizer(cell_type_col=cell_type_col),
        "density": lambda: CellTypeDensityFeaturizer(cell_type_col=cell_type_col),
        "expression": MeanExpressionFeaturizer,
        "distance": lambda: SpatialDistanceFeaturizer(cell_type_col=cell_type_col),
        "point_pattern": lambda: PointPatternFeaturizer(
            # Match scripts/run_point_pattern_baseline.py and the extractor's
            # native default: the single-feature baseline computes K and L only.
            metrics=tuple(point_metrics or ("K", "L")),
            cell_type_col=cell_type_col,
        ),
        "mixing": lambda: MixingFeaturizer(cell_type_col=cell_type_col),
    }
    return CombinedFeaturizer(
        {group: factories[group]() for group in groups}, fixed_markers=fixed_markers
    )


def shared_markers(ds, task, gentest):
    """Return markers present in every labelled train and test region."""
    cohort_col = ds.validation_config["cohort_col"]
    meta = ds.get_task_metadata(task)
    train_ids = meta[meta[cohort_col].isin(gentest["train"])]["region_id"].tolist()
    test_ids = meta[meta[cohort_col].isin(gentest["test"])]["region_id"].tolist()
    if not train_ids or not test_ids:
        return None
    regions = ds.load_regions(train_ids + test_ids)
    marker_sets = [set(region.expression.columns) for region in regions]
    return sorted(set.intersection(*marker_sets)) if marker_sets else []


def run(dataset_names, combinations, seeds, data_root=None, point_metrics=None, selected_runs=None):
    """Run combinations, optionally restricted to ``(dataset, task, scheme)`` triples."""
    selected_runs = set(selected_runs) if selected_runs is not None else None
    rows = []
    for dataset_name in dataset_names:
        ds = load_dataset(dataset_name, data_root=data_root)
        marker_cache = {}
        print(f"=== {dataset_name} ===", flush=True)
        for combination in combinations:
            groups = tuple(combination.split("+"))
            normalize = "expression" in groups
            print(f"-- {combination} --", flush=True)

            for task in ds.task_ids:
                if selected_runs is not None and (dataset_name, task, "cv") not in selected_runs:
                    continue
                factory = lambda g=groups: make_featurizer(g, point_metrics=point_metrics)
                result = cross_validate(
                    ds, task, factory, model_factory, seeds=seeds, normalize=normalize
                )
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                mean, sd = summarize_folds(result, metric)
                rows.append(dict(dataset=dataset_name, task=task, combination=combination,
                                 scheme="cv", metric=metric, mean=mean, sd=sd, n=len(result)))
                print(f"  {task:24s} cv {metric:12s} {mean:.4f} +/- {sd:.4f}", flush=True)

            for gentest in ds.validation_config.get("generalization_tests", []):
                cell_type_col = gentest.get("cell_type_col", "cell_type")
                for task in gentest.get("tasks", ds.task_ids):
                    if (selected_runs is not None and
                            (dataset_name, task, gentest["name"]) not in selected_runs):
                        continue
                    marker_key = (gentest["name"], task)
                    if normalize and marker_key not in marker_cache:
                        marker_cache[marker_key] = shared_markers(ds, task, gentest)
                    markers = marker_cache.get(marker_key) if normalize else None
                    if normalize and markers is None:
                        continue
                    factory = lambda g=groups, c=cell_type_col, m=markers: make_featurizer(
                        g, cell_type_col=c, fixed_markers=m, point_metrics=point_metrics
                    )
                    result = cohort_split_test(
                        ds, task, gentest, factory, model_factory, seeds=seeds, normalize=normalize
                    )
                    if not result:
                        continue
                    metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                    mean, sd = summarize_folds(result, metric)
                    row = dict(dataset=dataset_name, task=task, combination=combination,
                               scheme=gentest["name"], metric=metric, mean=mean, sd=sd,
                               n=len(result))
                    if markers is not None:
                        row["n_markers"] = len(markers)
                    rows.append(row)
                    print(f"  {task:24s} {gentest['name']:16s} {metric:12s} "
                          f"{mean:.4f} +/- {sd:.4f}", flush=True)
        ds.clear_region_cache()
    return pd.DataFrame(rows)


def parse_combinations(values):
    requested = values or list(COMBINATIONS)
    invalid = sorted(set(requested) - set(COMBINATIONS))
    if invalid:
        raise ValueError(f"Unknown/invalid combinations: {invalid}; choose from {COMBINATIONS}")
    return requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--combinations", nargs="*", default=None, choices=COMBINATIONS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--point-pattern-metrics", nargs="+", default=["K", "L"],
                        choices=["K", "L", "pcf", "variogram"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=str(_CODE / "results" / "pairwise_benchmark.csv"))
    args = parser.parse_args()

    frame = run(args.datasets or list_datasets(), parse_combinations(args.combinations),
                args.seeds, data_root=args.data_root, point_metrics=args.point_pattern_metrics)
    frame["score"] = frame.apply(lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
