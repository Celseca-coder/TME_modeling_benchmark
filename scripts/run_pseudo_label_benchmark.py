#!/usr/bin/env python
"""Verify existing featurizers on motif pseudo-label tasks (patient-level CV).

    python scripts/run_pseudo_label_benchmark.py \\
        --dataset hnc_wu2022 --data-root "$DATA_ROOT" \\
        --feature-sources density point-pattern \\
        --primary-tasks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.basic_feats import (  # noqa: E402
    CompositionFeaturizer,
    MeanExpressionFeaturizer,
)
from benchmark.features.combined import CombinedCompositionExpressionFeaturizer  # noqa: E402
from benchmark.features.density_feats import CellTypeDensityFeaturizer  # noqa: E402
from benchmark.features.point_pattern import PointPatternFeaturizer  # noqa: E402
from benchmark.features.spatial_distance import SpatialDistanceFeaturizer  # noqa: E402
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.overlay import attach_pseudo_labels, motif_task_ids  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation import PRIMARY_METRIC, cross_validate, summarize_folds  # noqa: E402

FEATURE_ALIASES = {
    "composition+expression": "composition-expression",
    "composition_expression": "composition-expression",
    "distance": "spatial-distance",
    "point_pattern": "point-pattern",
}

# Composition controls + spatial tasks composition could not recover.
PRIMARY_MOTIF_TASKS = (
    "motif_tumor_high",
    "motif_cd8_high",
    "motif_cd8_clustering",
    "motif_tls_like",
    "motif_immune_exclusion",
    "motif_interface_immune",
    "motif_tumor_stroma_mixing",
)


def _normalize_source(name: str) -> str:
    return FEATURE_ALIASES.get(name, name)


def _featurizer_factory(name: str, cell_type_col: str, by_type: bool = False):
    name = _normalize_source(name)
    if name == "composition":
        return lambda: CompositionFeaturizer(cell_type_col=cell_type_col)
    if name == "expression":
        return lambda: MeanExpressionFeaturizer()
    if name == "composition-expression":
        return lambda: CombinedCompositionExpressionFeaturizer(
            cell_type_col=cell_type_col,
            feature_groups=("composition", "expression"),
        )
    if name == "density":
        return lambda: CellTypeDensityFeaturizer(cell_type_col=cell_type_col)
    if name == "point-pattern":
        return lambda: PointPatternFeaturizer(
            cell_type_col=cell_type_col,
            radii=[10, 20, 50, 100, 200],
            metrics=("K", "L"),
            by_type=by_type,
        )
    if name == "spatial-distance":
        return lambda: SpatialDistanceFeaturizer(cell_type_col=cell_type_col, k=1)
    raise ValueError(f"Unknown feature source {name!r}")


def model_factory(task_cfg, seed):
    return LinearClassifier(seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument(
        "--feature-sources",
        nargs="+",
        default=["composition", "expression", "composition-expression"],
        choices=[
            "composition",
            "expression",
            "composition-expression",
            "composition+expression",
            "density",
            "point-pattern",
            "point_pattern",
            "distance",
            "spatial-distance",
        ],
    )
    ap.add_argument("--tasks", nargs="*", default=None, help="Subset of motif_* task ids")
    ap.add_argument(
        "--primary-tasks",
        action="store_true",
        help="Run the 2 composition controls plus 5 spatial tasks composition could not recover.",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument(
        "--label-version",
        default="v1",
        choices=["v1", "v2"],
        help="v1 = original tertile labels; v2 = bootstrap-confident labels.",
    )
    ap.add_argument(
        "--by-type",
        action="store_true",
        help="For point-pattern: compute K/L separately per cell type.",
    )
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--append",
        action="store_true",
        help="Keep existing rows in --output and replace matching dataset/task/feature_source/scheme.",
    )
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    labels_path = Path(args.labels or (_CODE / "results" / "pseudo_labels" / f"{args.dataset}.csv"))
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Pseudo-label table not found: {labels_path}\n"
            "Run scripts/generate_pseudo_labels.py first."
        )

    dataset = load_dataset(args.dataset, data_root=args.data_root)
    attach_pseudo_labels(dataset, labels_path, catalog, label_version=args.label_version)
    if args.primary_tasks:
        available = set(motif_task_ids(dataset))
        missing = [task for task in PRIMARY_MOTIF_TASKS if task not in available]
        if missing:
            raise ValueError(f"Primary motif tasks missing from labels/catalog: {missing}")
        tasks = list(PRIMARY_MOTIF_TASKS)
    else:
        tasks = args.tasks or motif_task_ids(dataset)
    cell_type_col = catalog.cell_type_col

    out = Path(args.output or (_CODE / "results" / "pseudo_label_benchmark.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = [_normalize_source(name) for name in args.feature_sources]
    needs_expression = {"expression", "composition-expression"}
    rows = []
    for source in sources:
        print(f"=== {args.dataset} / {source} ===", flush=True)
        featurizer = _featurizer_factory(source, cell_type_col, by_type=args.by_type)
        normalize = source in needs_expression
        for task in tasks:
            metric = PRIMARY_METRIC[dataset.get_task_config(task)["type"]]
            print(f"  start {task} / {source}", flush=True)
            folds = cross_validate(
                dataset, task, featurizer, model_factory,
                seeds=args.seeds, normalize=normalize,
            )
            mean, sd = summarize_folds(folds, metric)
            rows.append(dict(
                dataset=args.dataset, task=task, feature_source=source,
                scheme="cv", metric=metric, mean=mean, sd=sd, n=len(folds),
            ))
            print(f"  {task:28s} {source:18s} {metric:12s} {mean:.4f} +/- {sd:.4f}", flush=True)
            _write_results(out, rows, append_existing=args.append)
        dataset.clear_region_cache()

    print(f"\nWrote {out}  ({len(rows)} new rows)", flush=True)


def _write_results(path: Path, rows: list[dict], append_existing: bool) -> None:
    df = pd.DataFrame(rows)
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    if append_existing and path.exists():
        prev = pd.read_csv(path)
        keys = ["dataset", "task", "feature_source", "scheme"]
        prev = prev[~prev.set_index(keys).index.isin(df.set_index(keys).index)]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
