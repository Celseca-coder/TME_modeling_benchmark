#!/usr/bin/env python
"""Run the Cellular Neighborhood secondary-mask ablation benchmark.

Both experiment arms use ``k_neighbors=20`` and ``n_neighborhoods=20``:

* ``with_secondary_mask`` includes within-CN cell-type composition features.
* ``without_secondary_mask`` skips ``profiles[mask]`` and uses CN abundance only.
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
    PRIMARY_METRIC,
    cohort_split_test,
    cross_validate,
    summarize_folds,
)


K_NEIGHBORS = 20
N_NEIGHBORHOODS = 20
ABLATION_MODES = (
    ("with_secondary_mask", True),
    ("without_secondary_mask", False),
)


def model_factory(task_cfg, seed):
    """Create the model appropriate for the configured task.

    Args:
        task_cfg (dict): Configuration for the current prediction task.
        seed (int): Random seed used to initialize the model.

    Returns:
        LinearCox | LinearClassifier: A new, unfitted region-level model.
    """
    if task_cfg["type"] == "survival":
        return LinearCox(seed=seed)
    return LinearClassifier(seed=seed)


def _featurizer_factory(
    include_content: bool,
    cell_type_col: str = "cell_type",
):
    """Create a fold-safe CN featurizer factory for one ablation arm.

    Args:
        include_content (bool): Whether to calculate within-CN cell-type
            composition using ``profiles[mask]``.
        cell_type_col (str): Column containing cell-type labels.

    Returns:
        callable: Factory that creates a new CN featurizer for each fold.
    """
    return lambda: CellularNeighborhoodFeaturizer(
        cell_type_col=cell_type_col,
        k_neighbors=K_NEIGHBORS,
        n_neighborhoods=N_NEIGHBORHOODS,
        max_cells_per_fit_region=2000,
        include_cn_celltype_content=include_content,
    )


def _result_row(
    dataset: str,
    task: str,
    scheme: str,
    mask_mode: str,
    metric: str,
    values: list[dict],
) -> dict:
    """Summarize one ablation arm for the output table.

    Args:
        dataset (str): Dataset name.
        task (str): Prediction-task identifier.
        scheme (str): Validation scheme name.
        mask_mode (str): Name of the secondary-mask ablation arm.
        metric (str): Primary evaluation metric.
        values (list[dict]): Per-fold or per-seed metric dictionaries.

    Returns:
        dict: One benchmark result row.
    """
    mean, sd = summarize_folds(values, metric)
    return {
        "dataset": dataset,
        "task": task,
        "scheme": scheme,
        "mask_mode": mask_mode,
        "k_neighbors": K_NEIGHBORS,
        "n_neighborhoods": N_NEIGHBORHOODS,
        "metric": metric,
        "mean": mean,
        "sd": sd,
        "n": len(values),
    }


def run(dataset_names, seeds, data_root=None) -> pd.DataFrame:
    """Run both ablation arms over cross-validation and cohort tests.

    Args:
        dataset_names (list[str]): Datasets to benchmark.
        seeds (list[int]): Random seeds for repeated validation.
        data_root (str | None): Optional dataset root override.

    Returns:
        pd.DataFrame: One summary row per task, scheme, and ablation arm.
    """
    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            for mask_mode, include_content in ABLATION_MODES:
                values = cross_validate(
                    ds,
                    task,
                    _featurizer_factory(include_content),
                    model_factory,
                    seeds=seeds,
                    normalize=False,
                )
                row = _result_row(
                    name, task, "cv", mask_mode, metric, values,
                )
                rows.append(row)
                print(
                    f"  {task:24s} cv {mask_mode:24s} "
                    f"(k={K_NEIGHBORS}, n={N_NEIGHBORHOODS}) "
                    f"{metric:12s} {row['mean']:.4f} +/- {row['sd']:.4f}"
                )

        for gentest in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gentest.get("cell_type_col", "cell_type")
            for task in gentest.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                for mask_mode, include_content in ABLATION_MODES:
                    values = cohort_split_test(
                        ds,
                        task,
                        gentest,
                        _featurizer_factory(include_content, cell_type_col),
                        model_factory,
                        seeds=seeds,
                        normalize=False,
                    )
                    if not values:
                        continue
                    row = _result_row(
                        name,
                        task,
                        gentest["name"],
                        mask_mode,
                        metric,
                        values,
                    )
                    rows.append(row)
                    print(
                        f"  {task:24s} {gentest['name']:16s} "
                        f"{mask_mode:24s} "
                        f"(k={K_NEIGHBORS}, n={N_NEIGHBORHOODS}) "
                        f"{metric:12s} {row['mean']:.4f} +/- {row['sd']:.4f}"
                    )

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main() -> None:
    """Parse command-line arguments, run the ablation, and write its CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--output",
        default=str(
            _CODE / "results" / "cellular_neighborhood_mask_ablation.csv"
        ),
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    df = run(
        args.datasets or list_datasets(),
        args.seeds,
        data_root=args.data_root,
    )
    df["score"] = df.apply(
        lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}",
        axis=1,
    )
    columns = [
        "dataset",
        "task",
        "scheme",
        "mask_mode",
        "k_neighbors",
        "n_neighborhoods",
        "metric",
        "mean",
        "sd",
        "score",
        "n",
    ]
    df = df[columns]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nWrote {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
