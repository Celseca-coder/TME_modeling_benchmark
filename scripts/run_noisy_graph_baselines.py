#!/usr/bin/env python
"""Run in-repo CytoCommunity / Cell-Graph Signature / Cellular Neighborhood / SORBET
on PseudoNoisyDataset labels.

These are the rewritten adapters under ``benchmark/features`` and
``benchmark/models``, not the original GitHub pipelines.

Imaging still comes from TME_benchmark_data. Only the *targets* are noisy v2
motif labels. Clinical generalization tests are skipped.

    python scripts/run_noisy_graph_baselines.py \
        --method cellular-neighborhood \
        --data-root "$DATA_ROOT" \
        --label-root /autofs/nas8/tywang/tjzou/PseudoNoisyDataset
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.local_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_LABEL_ROOT,
    PSEUDO_DATASETS,
    TASKS_BY_DATASET,
    attach_pseudo_tasks,
    resolve_label_path,
)
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation import PRIMARY_METRIC, cross_validate, summarize_folds  # noqa: E402

METHODS = (
    "cyto-community",
    "cell-graph-signature",
    "cellular-neighborhood",
    "sorbet",
)

CN_ABLATION = (
    ("with_secondary_mask", True),
    ("without_secondary_mask", False),
)


def _method_stack(method: str, cell_type_col: str, args):
    if method == "cyto-community":
        from benchmark.features.cyto_community_builder import CytoCommunityGraphBuilder
        from benchmark.models.cyto_community import CytoCommunityClassifier

        def featurizer():
            return CytoCommunityGraphBuilder(
                cell_type_col=cell_type_col,
                radius_um=50.0,
                k_neighbors=8,
                max_cells=args.max_cells,
                include_expression=True,
            )

        def model_factory(task_cfg, seed):
            return CytoCommunityClassifier(
                seed=seed,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=args.device,
            )

        return featurizer, model_factory, True, None

    if method == "cell-graph-signature":
        from benchmark.features.cell_graph_signature import CellGraphSignatureBuilder
        from benchmark.models.cell_graph_signature import CellGraphSignatureClassifier

        def featurizer():
            return CellGraphSignatureBuilder(graph_size=100, radius_um=20.0)

        def model_factory(task_cfg, seed):
            return CellGraphSignatureClassifier(seed=seed)

        return featurizer, model_factory, True, None

    if method == "sorbet":
        from benchmark.features.sorbet_builder import SORBETGraphBuilder
        from benchmark.models.sorbet import SORBETClassifier

        def featurizer():
            return SORBETGraphBuilder(
                cell_type_col=cell_type_col,
                radius_um=50.0,
                k_neighbors=12,
                max_centers=192,
                max_nodes_per_subgraph=96,
                include_expression=True,
                seed=0,
            )

        def model_factory(task_cfg, seed):
            return SORBETClassifier(
                seed=seed,
                hidden_dim=128,
                n_layers=2,
                epochs=60,
                lr=1e-3,
                weight_decay=1e-4,
                dropout=0.15,
                micro_batch_size=64,
                region_batch_size=4,
                device=args.device,
            )

        return featurizer, model_factory, True, None

    if method != "cellular-neighborhood":
        raise ValueError(f"Unknown method {method!r}")

    from benchmark.features.cellular_neighborhood import CellularNeighborhoodFeaturizer
    from benchmark.models.linear import LinearClassifier

    def model_factory(task_cfg, seed):
        return LinearClassifier(seed=seed)

    def make_cn(include_content: bool):
        return lambda: CellularNeighborhoodFeaturizer(
            cell_type_col=cell_type_col,
            k_neighbors=20,
            n_neighborhoods=20,
            max_cells_per_fit_region=2000,
            include_cn_celltype_content=include_content,
        )

    return make_cn, model_factory, False, CN_ABLATION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--label-root", default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default=os.environ.get("CYTO_COMMUNITY_DEVICE", "cpu"))
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-cells", type=int, default=1024)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"

    if not str(args.data_root or "").strip():
        args.data_root = DEFAULT_DATA_ROOT
    if not str(args.label_root or "").strip():
        args.label_root = DEFAULT_LABEL_ROOT

    names = args.datasets or list(PSEUDO_DATASETS)
    rows = []
    for name in names:
        ds = load_dataset(name, data_root=args.data_root)
        tasks = attach_pseudo_tasks(ds, name, args.label_root, args.tasks)
        label_path = resolve_label_path(name, args.label_root)
        cell_type_col = ds.config.get("cell_type_col", "cell_type")
        print(
            f"=== {name}  method={args.method}  images={ds._root}  labels={label_path} ===",
            flush=True,
        )
        print(f"  pseudo tasks: {tasks}", flush=True)
        feat, make_model, normalize, cn_arms = _method_stack(
            args.method, cell_type_col, args,
        )
        wanted = args.tasks or TASKS_BY_DATASET[name]
        for task in wanted:
            if task not in ds.task_ids:
                print(f"  skip {task}: not registered", flush=True)
                continue
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            arms = cn_arms or (("main", None),)
            for mask_mode, include_content in arms:
                factory = feat(include_content) if cn_arms else feat
                folds = cross_validate(
                    ds, task, factory, make_model, seeds=args.seeds, normalize=normalize,
                )
                if not folds:
                    print(f"  skip {task}: no labelled folds", flush=True)
                    continue
                mean, sd = summarize_folds(folds, metric)
                row = dict(
                    dataset=name, task=task, method=args.method, scheme="cv",
                    metric=metric, mean=mean, sd=sd, n=len(folds),
                )
                if args.method == "cellular-neighborhood":
                    row["mask_mode"] = mask_mode
                    row["k_neighbors"] = 20
                    row["n_neighborhoods"] = 20
                    print(
                        f"  {task:32s} {mask_mode:24s} {metric:10s} {mean:.4f} +/- {sd:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {task:32s} {metric:10s} {mean:.4f} +/- {sd:.4f}",
                        flush=True,
                    )
                rows.append(row)
        ds.clear_region_cache()

    out = Path(args.output or (
        _CODE / "results" / "noisy_label_baselines" / f"{args.method}.csv"
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
