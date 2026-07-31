#!/usr/bin/env python
"""Cell-Graph Signature (GIN-TopK) benchmark over configured datasets/tasks."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.utils.task_filter import should_skip_benchmark_task
from benchmark.validation import cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC

try:
    from benchmark.features.cell_graph_signature import CellGraphSignatureBuilder
    from benchmark.models.cell_graph_signature import (
        CellGraphSignatureClassifier,
        CellGraphSignatureCox,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"torch", "torch_geometric"}:
        raise SystemExit(
            "Cell-Graph Signature requires PyTorch and torch-geometric. "
            "Install the dependencies listed in requirements.txt."
        ) from exc
    raise


def check_runtime_deps() -> None:
    """Check runtime deps."""
    missing = [
        name for name in ("torch", "torch_geometric", "sklearn", "lifelines")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit("Missing runtime dependencies: " + ", ".join(missing))


def model_factory(task_cfg, seed):
    """Execute the model factory operation.
    
        Args:
            task_cfg (Any): Configuration mapping for the current prediction task.
            seed (Any): Random seed used for reproducibility.
    
        Returns:
            Any: The operation result.
    
    Args:
        task_cfg (Any): Configuration mapping for the current prediction task."""
    cls = CellGraphSignatureCox if task_cfg["type"] == "survival" else CellGraphSignatureClassifier
    return cls(seed=seed)


def _builder():
    """Execute the builder operation.

    Returns:
        Any: The operation result."""
    return CellGraphSignatureBuilder(graph_size=100, radius_um=20.0)


def run(dataset_names, seeds, data_root=None) -> pd.DataFrame:
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
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            if should_skip_benchmark_task(name, task, "cv", metric):
                print(f"  {task:24s} cv               {metric:12s} skipped")
                continue
            results = cross_validate(
                ds, task, _builder, model_factory, seeds=seeds, normalize=True
            )
            mean, sd = summarize_folds(results, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv", metric=metric,
                             mean=mean, sd=sd, n=len(results)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for test in ds.validation_config.get("generalization_tests", []):
            for task in test.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                if should_skip_benchmark_task(name, task, test["name"], metric):
                    print(f"  {task:24s} {test['name']:16s} {metric:12s} skipped")
                    continue
                results = cohort_split_test(
                    ds, task, test, _builder, model_factory, seeds=seeds, normalize=True
                )
                if not results:
                    continue
                mean, sd = summarize_folds(results, metric)
                rows.append(dict(dataset=name, task=task, scheme=test["name"], metric=metric,
                                 mean=mean, sd=sd, n=len(results)))
                print(f"  {task:24s} {test['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")
        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main() -> None:
    """Execute the main operation."""
    check_runtime_deps()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default=str(_CODE / "results" / "cell_graph_signature_benchmark.csv"),
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    frame = run(args.datasets or list_datasets(), args.seeds, data_root=args.data_root)
    frame["score"] = frame.apply(
        lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
