#!/usr/bin/env python
"""Handcrafted local-window features + gated-attention MIL benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from benchmark.features import HandcraftedAttentionMILFeaturizer
from benchmark.models import AttentionMILModel
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation import cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC
from benchmark.validation.selected_tasks import SELECTED_DATASETS, SELECTED_TRIPLES


def run(args) -> pd.DataFrame:
    rows = []
    if args.selected_runs:
        requested = set(args.datasets) if args.datasets else None
        names = [
            name for name in SELECTED_DATASETS
            if requested is None or name in requested
        ]
    else:
        names = args.datasets or list_datasets()
    groups = tuple(args.feature_groups)
    normalize = args.normalize or "expression" in groups
    selected = SELECTED_TRIPLES if args.selected_runs else None
    for dataset_i, name in enumerate(names, 1):
        ds = load_dataset(name, data_root=args.data_root)
        print(f"[{dataset_i}/{len(names)}] === {name} ===", flush=True)

        def make_feat(cell_type_col="cell_type"):
            return HandcraftedAttentionMILFeaturizer(
                window_size_um=args.window_size,
                step_um=args.step,
                feature_groups=groups,
                cell_type_col=cell_type_col,
                min_cells_per_window=args.min_cells,
                use_tissue_mask=not args.no_tissue_mask,
                max_markers=args.max_markers,
            )

        def make_model(task_cfg, seed):
            return AttentionMILModel(
                task_type=task_cfg["type"], seed=seed,
                hidden_dim=args.hidden_dim, attention_dim=args.attention_dim,
                dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
                epochs=args.epochs, patience=args.patience,
                device=args.device, max_instances=args.max_instances,
            )

        for task in ds.task_ids:
            if selected is not None and (name, task, "cv") not in selected:
                continue
            cfg = ds.get_task_config(task)
            if args.only_c_index and cfg["type"] != "survival":
                continue
            metric = PRIMARY_METRIC[cfg["type"]]
            fm = cross_validate(ds, task, make_feat, make_model, seeds=args.seeds,
                                normalize=normalize)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv", metric=metric,
                             mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}",
                  flush=True)

        for gt in ds.validation_config.get("generalization_tests", []):
            for task in gt.get("tasks", ds.task_ids):
                if selected is not None and (name, task, gt["name"]) not in selected:
                    continue
                cfg = ds.get_task_config(task)
                if args.only_c_index and cfg["type"] != "survival":
                    continue
                metric = PRIMARY_METRIC[cfg["type"]]
                column = gt.get("cell_type_col", "cell_type")
                feat = lambda c=column: make_feat(c)
                fm = cohort_split_test(ds, task, gt, feat, make_model,
                                       seeds=args.seeds, normalize=normalize)
                if not fm:
                    continue
                mean, sd = summarize_folds(fm, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"], metric=metric,
                                 mean=mean, sd=sd, n=len(fm)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}",
                      flush=True)
        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--selected-runs",
        action="store_true",
        help="Run only the 17 curated dataset/task/validation-scheme combinations.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--window-size", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=50.0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--feature-groups", nargs="+",
                        choices=sorted(HandcraftedAttentionMILFeaturizer.VALID_GROUPS),
                        default=["composition", "expression"],
                        help=(
                            "Local feature groups. The default directly concatenates "
                            "global-style composition and mean expression."
                        ))
    parser.add_argument("--max-markers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-instances", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize regions; enabled automatically when expression is selected.",
    )
    parser.add_argument("--no-tissue-mask", action="store_true")
    parser.add_argument("--only-c-index", action="store_true")
    parser.add_argument("--output", default=str(CODE_DIR / "results" / "attention_mil_benchmark.csv"))
    args = parser.parse_args()

    result = run(args)
    result["score"] = result.apply(
        lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {output} ({len(result)} rows)", flush=True)


if __name__ == "__main__":
    main()
