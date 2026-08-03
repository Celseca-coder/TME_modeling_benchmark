#!/usr/bin/env python
"""Patch-based composition/expression + MIL aggregation baseline over all datasets, tasks and schemes.

Feature: per region, divide into fixed-size windows, compute composition (or mean expression)
per window, then aggregate with MIL-style pooling (mean, max, std, quantiles).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.features.patch_feats import PatchBasedFeaturizer
from benchmark.models.linear import LinearClassifier, LinearCox
from benchmark.validation.selected_tasks import SELECTED_DATASETS, SELECTED_TRIPLES
from benchmark.validation import (
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    """Execute the model factory operation.
    
        Args:
            task_cfg (Any): Configuration mapping for the current prediction task.
            seed (Any): Random seed used for reproducibility.
    
        Returns:
            Any: The operation result.
    
    Args:
        task_cfg (Any): Configuration mapping for the current prediction task."""
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def run(dataset_names, seeds, data_root=None, *, window_size=100, step=50,
        feature_type="composition", expression_vocab_strategy="union",
        selected_runs=None) -> pd.DataFrame:
    """Run.
    
        Args:
            dataset_names (Any): Names of datasets to process.
            seeds (Any): Random seeds used for repeated benchmark runs.
            data_root (Any): Root directory containing data.
    
        Returns:
            pd.DataFrame: The operation result.
    
    Args:
        dataset_names (Any): Names of datasets to process."""
    rows = []
    selected_runs = set(selected_runs) if selected_runs is not None else None
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        # (A) CV
        for task in ds.task_ids:
            if selected_runs is not None and (name, task, "cv") not in selected_runs:
                continue
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            featurizer = lambda: PatchBasedFeaturizer(
                window_size_um=window_size,
                step_um=step,
                feature_type=feature_type,
                expression_vocab_strategy=expression_vocab_strategy,
                aggregations=("mean", "max", "std", "quantile"),
                quantiles=(0.25, 0.5, 0.75),
            )
            fm = cross_validate(
                ds, task, featurizer, model_factory, seeds=seeds,
                normalize=(feature_type == "expression"),
            )
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        # (B) generalization tests
        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            featurizer = lambda c=cell_type_col: PatchBasedFeaturizer(
                window_size_um=window_size,
                step_um=step,
                feature_type=feature_type,
                expression_vocab_strategy=expression_vocab_strategy,
                cell_type_col=c,
                aggregations=("mean", "max", "std", "quantile"),
                quantiles=(0.25, 0.5, 0.75),
            )
            for task in gt.get("tasks", ds.task_ids):
                if (selected_runs is not None and
                        (name, task, gt["name"]) not in selected_runs):
                    continue
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_split_test(
                    ds, task, gt, featurizer, model_factory, seeds=seeds,
                    normalize=(feature_type == "expression"),
                )
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    """Execute the main operation.

    Returns:
        Any: The operation result."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "patch_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--window-size", type=float, default=100, help="Window side length (µm)")
    ap.add_argument("--step", type=float, default=50, help="Step size (µm)")
    ap.add_argument("--feature-type", choices=["composition", "expression"], default="composition")
    ap.add_argument(
        "--expression-marker-vocab",
        choices=["union", "intersection"],
        default="union",
        help=("Marker vocabulary learned from training regions for patch expression; "
              "ignored for composition"),
    )
    ap.add_argument(
        "--selected-tasks",
        action="store_true",
        help="Run only the shared curated list of 17 dataset/task/scheme entries",
    )
    args = ap.parse_args()

    dataset_names = args.datasets or (list(SELECTED_DATASETS) if args.selected_tasks else list_datasets())
    df = run(
        dataset_names, args.seeds, data_root=args.data_root,
        window_size=args.window_size, step=args.step, feature_type=args.feature_type,
        expression_vocab_strategy=args.expression_marker_vocab,
        selected_runs=SELECTED_TRIPLES if args.selected_tasks else None,
    )
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
