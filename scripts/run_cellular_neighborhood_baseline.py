#!/usr/bin/env python
"""Cellular Neighborhood baseline over all datasets, tasks and schemes.

Feature: unsupervised local cell-type neighbourhood states. For each training
fold, local cell-type composition profiles are clustered into Cellular
Neighborhoods, then each region is summarized by CN abundance and CN content.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.cellular_neighborhood import CellularNeighborhoodFeaturizer
from benchmark.models.linear import LinearClassifier, LinearCox
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation import (
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def run(dataset_names, seeds, k_list, n_list, data_root=None) -> pd.DataFrame:
    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            
            # --- 增加 Grid Search 循环 ---
            for n_val in n_list:
                for k_val in k_list:
                    # 使用 k=k_val, n=n_val 解决 Python 闭包延迟绑定的坑
                    featurizer = lambda k=k_val, n=n_val: CellularNeighborhoodFeaturizer(
                        n_neighborhoods=n, k_neighbors=k, max_cells_per_fit_region=2000,
                    )
                    fm = cross_validate(ds, task, featurizer, model_factory, seeds=seeds, normalize=False)
                    mean, sd = summarize_folds(fm, metric)
                    
                    # 在结果字典中记录 k 和 n
                    rows.append(dict(dataset=name, task=task, scheme="cv",
                                     k_neighbors=k_val, n_neighborhoods=n_val,
                                     metric=metric, mean=mean, sd=sd, n=len(fm)))
                    print(f"  {task:24s} cv (k={k_val:2d}, n={n_val:2d})    {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            
            # --- 同样为 Generalization Test 增加 Grid Search 循环 ---
            for n_val in n_list:
                for k_val in k_list:
                    featurizer = lambda c=cell_type_col, k=k_val, n=n_val: CellularNeighborhoodFeaturizer(
                        cell_type_col=c, n_neighborhoods=n, k_neighbors=k,
                        max_cells_per_fit_region=2000,
                    )
                    for task in gt.get("tasks", ds.task_ids):
                        metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                        res = cohort_split_test(ds, task, gt, featurizer, model_factory, seeds=seeds, normalize=False)
                        if not res:
                            continue
                        mean, sd = summarize_folds(res, metric)
                        
                        # 记录 k 和 n
                        rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                         k_neighbors=k_val, n_neighborhoods=n_val,
                                         metric=metric, mean=mean, sd=sd, n=len(res)))
                        print(f"  {task:24s} {gt['name']:16s} (k={k_val:2d}, n={n_val:2d}) {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    # 增加供命令行调整的 Grid Search 参数
    ap.add_argument("--k-list", type=int, nargs="+", default=[10, 20, 30], help="List of k_neighbors to test")
    ap.add_argument("--n-list", type=int, nargs="+", default=[10, 20, 30], help="List of n_neighborhoods to test")
    
    ap.add_argument("--output", default=str(_CODE / "results" / "cellular_neighborhood_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    df = run(args.datasets or list_datasets(), args.seeds, args.k_list, args.n_list, data_root=args.data_root)
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f}", axis=1)
    
    # 调整列的顺序，把参数放在前面更直观
    cols = ['dataset', 'task', 'scheme', 'k_neighbors', 'n_neighborhoods', 'metric', 'mean', 'sd', 'score', 'n']
    df = df[cols]
    
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()