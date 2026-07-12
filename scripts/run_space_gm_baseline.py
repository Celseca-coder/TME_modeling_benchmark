#!/usr/bin/env python
"""SPACE-GM baseline over all datasets, tasks and schemes.

Each centre cell defines a 3-hop, 75-um microenvironment. The model performs
protein-expression pre-training followed by phenotype fine-tuning and averages
microenvironment predictions to obtain one region-level prediction.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation import (
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)

try:
    from benchmark.features.graph_builder import SpaceGMGraphBuilder
    from benchmark.models.space_gm import SpaceGMClassifier, SpaceGMCox
except ModuleNotFoundError as exc:
    if exc.name in {"torch", "torch_geometric"}:
        raise SystemExit(
            "SPACE-GM baseline requires PyTorch and torch-geometric. "
            "Install them first, for example: pip install 'torch>=2.2.2' 'torch-geometric>=2.4.0'"
        ) from exc
    raise


def check_runtime_deps():
    missing = [
        name for name in ("torch", "torch_geometric", "lifelines", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            "SPACE-GM baseline is missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in the SPACE-GM environment before running."
        )


def model_factory(task_cfg, seed):
    return SpaceGMCox(seed=seed) if task_cfg["type"] == "survival" else SpaceGMClassifier(seed=seed)


def run(dataset_names, seeds, data_root=None) -> pd.DataFrame:
    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        # (A) CV
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            featurizer = lambda: SpaceGMGraphBuilder(
                n_hops=3, radius_um=75.0, near_edge_um=20.0, max_centers=256,
            )
            # normalize=True: node里塞的marker表达需要走数据集自己的标准化,
            # 因为SpaceGM*模型没有走_TabularModel那层scaler(见space_gm.py)
            fm = cross_validate(ds, task, featurizer, model_factory, seeds=seeds, normalize=True)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        # (B) generalization tests
        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            featurizer = lambda c=cell_type_col: SpaceGMGraphBuilder(
                cell_type_col=c, n_hops=3, radius_um=75.0,
                near_edge_um=20.0, max_centers=256,
            )
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_split_test(ds, task, gt, featurizer, model_factory, seeds=seeds, normalize=True)
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    check_runtime_deps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "space_gm_benchmark.csv"))
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
