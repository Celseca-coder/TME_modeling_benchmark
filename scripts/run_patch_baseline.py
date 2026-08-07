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
from benchmark.validation import (
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def run(dataset_names, seeds, args) -> pd.DataFrame:
    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=args.data_root)
        print(f"=== {name} ===")

        # (A) CV
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            featurizer = lambda: PatchBasedFeaturizer(
                window_size_um=args.window_size,
                step_um=args.step,
                feature_groups=tuple(args.feature_groups),
                aggregations=tuple(args.aggregations),
                quantiles=tuple(args.quantiles),
                min_cells_per_window=args.min_cells,
                use_tissue_mask=not args.no_tissue_mask,
            )
            fm = cross_validate(ds, task, featurizer, model_factory, seeds=seeds, normalize=False)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        # (B) generalization tests
        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            featurizer = lambda c=cell_type_col: PatchBasedFeaturizer(
                window_size_um=args.window_size,
                step_um=args.step,
                feature_groups=tuple(args.feature_groups),
                cell_type_col=c,
                aggregations=tuple(args.aggregations),
                quantiles=tuple(args.quantiles),
                min_cells_per_window=args.min_cells,
                use_tissue_mask=not args.no_tissue_mask,
            )
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_split_test(ds, task, gt, featurizer, model_factory, seeds=seeds, normalize=False)
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "patch_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--window-size", type=float, default=100, help="Window side length (um)")
    ap.add_argument("--step", type=float, default=50, help="Step size (um)")
    ap.add_argument(
        "--feature-groups", nargs="+", choices=["composition", "expression"],
        default=["composition"],
        help="One or both per-window feature groups.",
    )
    ap.add_argument(
        "--aggregations", nargs="+", choices=["mean", "max", "min", "std", "quantile"],
        default=["mean", "max", "std", "quantile"],
    )
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--no-tissue-mask", action="store_true")
    args = ap.parse_args()

    df = run(args.datasets or list_datasets(), args.seeds, args)
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
