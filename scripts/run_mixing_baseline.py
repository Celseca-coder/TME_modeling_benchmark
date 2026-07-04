#!/usr/bin/env python
"""Mixing / infiltration baseline over all datasets, tasks and schemes.

Feature: per region, cell-type mixing metrics (e.g., mixing score, Shannon entropy, local
mixing indices) computed from cell type labels and spatial coordinates. Aggregated over
the tissue foreground (or within tumour mask). May include both global and local summary
statistics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import time

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.features.mixing import MixingFeaturizer
from benchmark.models.linear import LinearClassifier, LinearCox
from benchmark.validation import (
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def run(dataset_names, seeds, data_root=None) -> pd.DataFrame:
    rows = []
    for name in dataset_names:
        print(f"=== {name} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            featurizer = lambda: MixingFeaturizer()
            fm = cross_validate(ds, task, featurizer, model_factory, seeds=seeds, normalize=False)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            featurizer = lambda c=cell_type_col: MixingFeaturizer(cell_type_col=c)
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
    ap.add_argument("--output", default=str(_CODE / "results" / "mixing_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    df = run(args.datasets or list_datasets(), args.seeds, data_root=args.data_root)
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()