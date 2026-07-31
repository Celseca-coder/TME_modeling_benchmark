#!/usr/bin/env python
"""SORBET-style graph neural network baseline over all datasets and tasks.

SORBET is an interpretable sample classification idea for spatial omics: build
cell-neighborhood graphs, embed local neighborhoods with a GNN, and aggregate
local evidence to predict region/sample phenotype.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.sorbet_builder import SORBETGraphBuilder
from benchmark.models.sorbet import SORBETClassifier, SORBETCox
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.utils.task_filter import is_selected_benchmark_task
from benchmark.validation import cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC


def check_runtime_deps() -> None:
    """Check runtime deps."""
    missing = [
        name for name in ("torch", "torch_geometric", "lifelines", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            "SORBET baseline is missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in the graph-model environment first."
        )


def make_model_factory(args):
    """Create model factory.
    
        Args:
            args (Any): Command-line arguments passed to the entry point.
    
        Returns:
            Any: The operation result.
    
    Args:
        args (Any): Command-line arguments passed to the entry point."""
    def model_factory(task_cfg, seed):
        """Execute the model factory operation.
        
                Args:
                    task_cfg (Any): Configuration mapping for the current prediction task.
                    seed (Any): Random seed used for reproducibility.
        
                Returns:
                    Any: The operation result.
        
        Args:
            task_cfg (Any): Configuration mapping for the current prediction task."""
        kwargs = dict(
            seed=seed,
            hidden_dim=args.hidden_dim,
            n_layers=args.layers,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            micro_batch_size=args.micro_batch_size,
            region_batch_size=args.region_batch_size,
            device=args.device,
        )
        return SORBETCox(**kwargs) if task_cfg["type"] == "survival" else SORBETClassifier(**kwargs)
    return model_factory


def make_featurizer(args, cell_type_col: str = "cell_type"):
    """Create featurizer.
    
        Args:
            args (Any): Command-line arguments passed to the entry point.
            cell_type_col (str): Name of the column containing cell type.
    
        Returns:
            Any: The operation result.
    
    Args:
        args (Any): Command-line arguments passed to the entry point."""
    return lambda c=cell_type_col: SORBETGraphBuilder(
        cell_type_col=c,
        radius_um=args.radius_um,
        k_neighbors=args.k_neighbors,
        max_centers=args.max_centers,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
        include_expression=not args.no_expression,
        seed=args.builder_seed,
    )


def should_run_task(args, dataset: str, task: str, scheme: str, metric: str) -> bool:
    """Execute the should run task operation.
    
        Args:
            args (Any): Command-line arguments passed to the entry point.
            dataset (str): Dataset name used to filter benchmark records.
            task (str): Benchmark task name used to filter results.
            scheme (str): Validation scheme name used to filter results.
            metric (str): Evaluation metric name used to filter results.
    
        Returns:
            bool: The operation result.
    
    Args:
        args (Any): Command-line arguments passed to the entry point."""
    return (not args.only_selected_tasks) or is_selected_benchmark_task(dataset, task, scheme, metric)


def run(dataset_names, seeds, args) -> pd.DataFrame:
    """Run.
    
        Args:
            dataset_names (Any): Names of datasets to process.
            seeds (Any): Random seeds used for repeated benchmark runs.
            args (Any): Command-line arguments passed to the entry point.
    
        Returns:
            pd.DataFrame: The operation result.
    
    Args:
        dataset_names (Any): Names of datasets to process."""
    rows = []
    model_factory = make_model_factory(args)
    for name in dataset_names:
        ds = load_dataset(name, data_root=args.data_root)
        print(f"=== {name} ===")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            if not should_run_task(args, name, task, "cv", metric):
                print(f"  {task:24s} cv               {metric:12s} skipped")
                continue
            fm = cross_validate(
                ds, task, make_featurizer(args), model_factory,
                seeds=seeds, normalize=True,
            )
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                if not should_run_task(args, name, task, gt["name"], metric):
                    print(f"  {task:24s} {gt['name']:16s} {metric:12s} skipped")
                    continue
                res = cohort_split_test(
                    ds, task, gt, make_featurizer(args, cell_type_col), model_factory,
                    seeds=seeds, normalize=True,
                )
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main() -> None:
    """Execute the main operation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "sorbet_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=os.environ.get("SORBET_DEVICE", "cuda"))
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--radius-um", type=float, default=50.0)
    ap.add_argument("--k-neighbors", type=int, default=12)
    ap.add_argument("--max-centers", type=int, default=192)
    ap.add_argument("--max-nodes-per-subgraph", type=int, default=96)
    ap.add_argument("--micro-batch-size", type=int, default=64)
    ap.add_argument("--region-batch-size", type=int, default=4)
    ap.add_argument("--builder-seed", type=int, default=0)
    ap.add_argument("--no-expression", action="store_true", help="Use cell-type one-hot features only.")
    ap.add_argument("--only-selected-tasks", action="store_true",
                    help="Run only the 17 selected benchmark tasks in benchmark.utils.task_filter.")
    ap.add_argument("--debug", action="store_true",
                    help="Raise fold-level exceptions instead of recording NaN metrics.")
    ap.add_argument("--progress", action="store_true",
                    help="Print fold stages and Cox epoch progress.")
    args = ap.parse_args()

    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"
    if args.progress:
        os.environ["BENCHMARK_PROGRESS"] = "1"

    check_runtime_deps()
    df = run(args.datasets or list_datasets(), args.seeds, args)
    if df.empty:
        df = pd.DataFrame(columns=["dataset", "task", "scheme", "metric", "mean", "sd", "n", "score"])
    else:
        df["score"] = df.apply(lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
