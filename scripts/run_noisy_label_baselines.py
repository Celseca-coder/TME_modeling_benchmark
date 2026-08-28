#!/usr/bin/env python
"""Run baselines on noisy motif labels from PseudoNoisyDataset.

Imaging data still comes from TME_benchmark_data. Only the *targets* are the
noisy v2 labels. SPACE-GM is not included.

    python scripts/run_noisy_label_baselines.py \
        --method composition-expression \
        --data-root "$DATA_ROOT" \
        --label-root /autofs/nas8/tywang/tjzou/PseudoNoisyDataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.attention_mil import HandcraftedAttentionMILFeaturizer  # noqa: E402
from benchmark.features.basic_feats import (  # noqa: E402
    CompositionFeaturizer,
    MeanExpressionFeaturizer,
)
from benchmark.features.combined import CombinedCompositionExpressionFeaturizer  # noqa: E402
from benchmark.features.density_feats import CellTypeDensityFeaturizer  # noqa: E402
from benchmark.features.mixing import MixingFeaturizer  # noqa: E402
from benchmark.features.patch_feats import PatchBasedFeaturizer  # noqa: E402
from benchmark.features.point_pattern import PointPatternFeaturizer  # noqa: E402
from benchmark.features.spatial_distance import SpatialDistanceFeaturizer  # noqa: E402
from benchmark.models.attention_mil import AttentionMILModel  # noqa: E402
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.noisy_overlay import attach_noisy_labels  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation import PRIMARY_METRIC, cross_validate, summarize_folds  # noqa: E402

DATASETS = (
    "bc_jackson2020",
    "hnc_wu2022",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)

# Selected 10 motifs plus matched controls on every cohort.
TASKS_BY_DATASET = {
    "bc_jackson2020": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_t_tumor_mixing", "motif_cd8_tumor_contact",
        "motif_macrophage_tumor_niche", "motif_apc_t_contact",
    ),
    "hnc_wu2022": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_cd8_clustering", "motif_immune_exclusion",
    ),
    "bc_metabric_ali2020": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_tumor_stroma_mixing", "motif_interface_immune",
    ),
    "tnbc_wang2023": (
        "motif_tumor_high", "motif_cd8_high",
    ),
}

ATTENTION_GROUPS = {
    "attention-composition": ("composition",),
    "attention-expression": ("expression",),
    "attention-composition-expression": ("composition", "expression"),
}

LINEAR_METHODS = {
    "composition", "expression", "composition-expression",
    "density", "spatial-distance", "point-pattern", "mixing",
    "patch-composition", "patch-expression", "patch-composition-expression",
}

ALL_METHODS = tuple(sorted(LINEAR_METHODS | set(ATTENTION_GROUPS)))

LABEL_FILES = {
    "bc_jackson2020": "per_dataset/bc_jackson2020_v2_noisy.csv",
    "hnc_wu2022": "per_dataset/hnc_wu2022_v2_noisy.csv",
    "bc_metabric_ali2020": "per_dataset/bc_metabric_ali2020_v2_noisy.csv",
    "tnbc_wang2023": "per_dataset/tnbc_wang2023_v2_noisy.csv",
}


def _featurizer(method: str, cell_type_col: str):
    if method == "composition":
        return lambda: CompositionFeaturizer(cell_type_col=cell_type_col)
    if method == "expression":
        return lambda: MeanExpressionFeaturizer()
    if method == "density":
        return lambda: CellTypeDensityFeaturizer(cell_type_col=cell_type_col)
    if method == "spatial-distance":
        return lambda: SpatialDistanceFeaturizer(cell_type_col=cell_type_col, k=1)
    if method == "point-pattern":
        return lambda: PointPatternFeaturizer(
            cell_type_col=cell_type_col, radii=[10, 20, 50, 100, 200],
            metrics=("K", "L"),
        )
    if method == "mixing":
        return lambda: MixingFeaturizer(cell_type_col=cell_type_col)
    if method == "patch-composition":
        return lambda: PatchBasedFeaturizer(
            window_size_um=100, step_um=50, feature_type="composition",
            cell_type_col=cell_type_col,
            aggregations=("mean", "max", "std", "quantile"),
            quantiles=(0.25, 0.5, 0.75),
        )
    if method == "patch-expression":
        return lambda: PatchBasedFeaturizer(
            window_size_um=100, step_um=50, feature_type="expression",
            aggregations=("mean", "max", "std", "quantile"),
            quantiles=(0.25, 0.5, 0.75),
        )
    if method == "patch-composition-expression":
        # Naive MIL mean: concatenate local composition + expression, then mean-pool.
        return lambda: PatchBasedFeaturizer(
            window_size_um=100, step_um=50,
            feature_groups=("composition", "expression"),
            cell_type_col=cell_type_col,
            aggregations=("mean",),
        )
    if method == "composition-expression":
        return lambda: CombinedCompositionExpressionFeaturizer(
            cell_type_col=cell_type_col,
            feature_groups=("composition", "expression"),
        )
    if method in ATTENTION_GROUPS:
        groups = ATTENTION_GROUPS[method]
        return lambda: HandcraftedAttentionMILFeaturizer(
            window_size_um=100, step_um=50, feature_groups=groups,
            cell_type_col=cell_type_col, min_cells_per_window=10,
        )
    raise ValueError(f"Unknown method {method!r}")


def _uses_expression(method: str) -> bool:
    return method in {
        "expression", "composition-expression", "patch-expression",
        "patch-composition-expression",
        "attention-expression", "attention-composition-expression",
    }


def _model_factory(method: str):
    if method in ATTENTION_GROUPS:
        def factory(task_cfg, seed):
            task_type = task_cfg["type"]
            if task_type in {"binary", "binary_classification"}:
                task_type = "binary"
            return AttentionMILModel(task_type=task_type, seed=seed, device="auto")
        return factory
    return lambda task_cfg, seed: LinearClassifier(seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-root", default="/autofs/nas8/tywang/tjzou/PseudoNoisyDataset")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    names = args.datasets or list(DATASETS)
    rows = []
    make_model = _model_factory(args.method)
    for name in names:
        ds = load_dataset(name, data_root=args.data_root)
        label_path = Path(args.label_root) / LABEL_FILES[name]
        attach_noisy_labels(ds, label_path)
        cell_type_col = ds.config.get("cell_type_col", "cell_type")
        factory = _featurizer(args.method, cell_type_col)
        normalize = _uses_expression(args.method)
        print(f"=== {name}  method={args.method}  labels={label_path} ===", flush=True)
        wanted = args.tasks or TASKS_BY_DATASET[name]
        for task in wanted:
            if task not in ds.task_ids:
                print(f"  skip {task}: not registered", flush=True)
                continue
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            folds = cross_validate(
                ds, task, factory, make_model, seeds=args.seeds, normalize=normalize,
            )
            if not folds:
                print(f"  skip {task}: no labelled folds", flush=True)
                continue
            mean, sd = summarize_folds(folds, metric)
            rows.append(dict(
                dataset=name, task=task, method=args.method, scheme="cv",
                metric=metric, mean=mean, sd=sd, n=len(folds),
            ))
            print(f"  {task:32s} {metric:10s} {mean:.4f} +/- {sd:.4f}", flush=True)
        ds.clear_region_cache()

    out = Path(args.output or (
        _CODE / "results" / "noisy_label_baselines" / f"{args.method}.csv"
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
