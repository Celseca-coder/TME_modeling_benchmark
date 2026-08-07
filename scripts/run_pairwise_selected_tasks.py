#!/usr/bin/env python
"""Run all 15 pairwise feature combinations on the 17 selected benchmark tasks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import load_dataset
from benchmark.validation import PRIMARY_METRIC
from benchmark.validation.selected_tasks import SELECTED_DATASETS, SELECTED_RUNS, SELECTED_TRIPLES
from run_pairwise_feature_combinations import COMBINATIONS, parse_combinations, run

def validate_selection(data_root=None):
    """Fail early if a selected task, scheme, or primary metric no longer matches YAML."""
    datasets = {}
    for dataset_name, task, scheme, expected_metric in SELECTED_RUNS:
        if dataset_name not in datasets:
            datasets[dataset_name] = load_dataset(dataset_name, data_root=data_root)
        ds = datasets[dataset_name]
        if task not in ds.task_ids:
            raise ValueError(f"Unknown selected task: {dataset_name}/{task}")
        actual_metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
        if actual_metric != expected_metric:
            raise ValueError(
                f"Metric mismatch for {dataset_name}/{task}: "
                f"selected={expected_metric}, configured={actual_metric}"
            )
        if scheme != "cv":
            generalization = {
                test["name"]: test for test in ds.validation_config.get("generalization_tests", [])
            }
            if scheme not in generalization or task not in generalization[scheme].get("tasks", ds.task_ids):
                raise ValueError(f"Unknown selected scheme/task: {dataset_name}/{task}/{scheme}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combinations", nargs="*", default=None, choices=COMBINATIONS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--point-pattern-metrics", nargs="+", default=["K", "L"],
                        choices=["K", "L", "pcf", "variogram"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output", default=str(_CODE / "results" / "pairwise_selected_tasks_benchmark.csv")
    )
    args = parser.parse_args()

    validate_selection(args.data_root)
    frame = run(
        SELECTED_DATASETS,
        parse_combinations(args.combinations),
        args.seeds,
        data_root=args.data_root,
        point_metrics=args.point_pattern_metrics,
        selected_runs=SELECTED_TRIPLES,
    )
    frame["score"] = frame.apply(lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {output} ({len(frame)} rows; expected up to {len(SELECTED_RUNS) * len(parse_combinations(args.combinations))})")


if __name__ == "__main__":
    main()
