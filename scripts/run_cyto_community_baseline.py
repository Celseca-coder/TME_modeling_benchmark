#!/usr/bin/env python
"""Cyto-Community baseline over all datasets, tasks and schemes.

Each region is represented as one spatial cell graph. A GraphSAGE encoder embeds
cells, a soft assignment layer pools them into latent communities, and a region
head predicts classification labels or Cox survival risk.
"""
from __future__ import annotations

import argparse
import os
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
    from benchmark.features.cyto_community_builder import CytoCommunityGraphBuilder
    from benchmark.models.cyto_community import CytoCommunityClassifier, CytoCommunityCox
except ModuleNotFoundError as exc:
    if exc.name in {"torch", "torch_geometric"}:
        raise SystemExit(
            "Cyto-Community baseline requires PyTorch and torch-geometric. "
            "Install them first, for example: pip install 'torch>=2.2.2' 'torch-geometric>=2.4.0'"
        ) from exc
    raise


def make_model_factory(hidden_dim: int, batch_size: int, epochs: int, device: str):
    def model_factory(task_cfg, seed):
        kwargs = dict(
            seed=seed,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
        )
        return CytoCommunityCox(**kwargs) if task_cfg["type"] == "survival" else CytoCommunityClassifier(**kwargs)

    return model_factory


def run(
    dataset_names,
    seeds,
    data_root=None,
    *,
    hidden_dim: int = 32,
    batch_size: int = 2,
    epochs: int = 50,
    device: str = "cpu",
    max_cells: int = 1024,
) -> pd.DataFrame:
    model_factory = make_model_factory(hidden_dim, batch_size, epochs, device)
    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            featurizer = lambda: CytoCommunityGraphBuilder(
                radius_um=50.0, k_neighbors=8, max_cells=max_cells, include_expression=True,
            )
            fm = cross_validate(ds, task, featurizer, model_factory, seeds=seeds, normalize=True)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for gt in ds.validation_config.get("generalization_tests", []):
            cell_type_col = gt.get("cell_type_col", "cell_type")
            featurizer = lambda c=cell_type_col: CytoCommunityGraphBuilder(
                cell_type_col=c, radius_um=50.0, k_neighbors=8,
                max_cells=max_cells, include_expression=True,
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "cyto_community_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=os.environ.get("CYTO_COMMUNITY_DEVICE", "cpu"))
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-cells", type=int, default=1024)
    ap.add_argument("--debug", action="store_true",
                    help="Raise fold-level exceptions instead of recording NaN metrics.")
    args = ap.parse_args()

    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"

    df = run(
        args.datasets or list_datasets(),
        args.seeds,
        data_root=args.data_root,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        max_cells=args.max_cells,
    )
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
