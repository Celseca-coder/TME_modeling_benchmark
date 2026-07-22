#!/usr/bin/env python
"""Run the genuine SPACE-GM model on **every binary-classification CV task**.

Sweeps all datasets (or a chosen subset), picks out the ``binary_classification``
tasks, and runs patient-level cross-validation for each via the amortized
:func:`benchmark.models.space_gm_real_cv.cross_validate_fast` (one
``CellularGraphDataset`` built per task, sliced by fold). Survival and
multiclass tasks, and cohort-generalization schemes, are skipped.

Per task it writes weights + a live ``cv_metrics.csv`` under ``--work-root`` /
``--weights-root`` (organised as ``<dataset>/<task>/``); at the end it writes a
summary CSV (one row per dataset/task) and an all-folds CSV.

Requires the ``spacegm`` package (and torch / torch_geometric / torch_scatter /
scikit-learn) in the active environment, e.g. conda env ``p3``.

Example
-------
    # everything, GPU, manuscript-ish budget
    python scripts/run_space_gm_binary_cv.py \
        --device cuda --num-iterations 50000 --emb-dim 512 \
        --out-root results/space_gm_binary_cv

    # one dataset, quick smoke test
    python scripts/run_space_gm_binary_cv.py --datasets hnc_wu2022 --seeds 0 \
        --num-iterations 300 --device cpu
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
from benchmark.validation import summarize_folds, PRIMARY_METRIC

BINARY_TYPES = {"binary_classification", "binary"}

DATA_ROOT = Path("/autofs/bal14/zqwu/CellularTables/TME_benchmark_data")
MODEL_SAVE_ROOT = Path(f'/autofs/bal14/zqwu/projects/TME_modeling_benchmark')

def check_runtime_deps():
    missing = [
        name for name in ("torch", "torch_geometric", "torch_scatter", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if importlib.util.find_spec("spacegm") is None:
        missing.append("spacegm")
    if missing:
        raise SystemExit(
            "SPACE-GM (real) is missing runtime dependencies: " + ", ".join(missing)
            + ".\nInstall them in the SPACE-GM environment (e.g. conda env 'p3') first."
        )


def build_config(args):
    from benchmark.models.space_gm_real_cv import SpaceGMConfig
    return SpaceGMConfig(
        subgraph_size=args.subgraph_size,
        subgraph_radius_limit_um=args.radius_um,
        near_edge_um=args.near_edge_um,
        emb_dim=args.emb_dim,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_subsample_ratio=args.eval_subsample_ratio,
        device=args.device,
    )


def binary_tasks(ds, only_tasks=None) -> list[str]:
    tasks = [t for t in ds.task_ids if ds.get_task_config(t)["type"] in BINARY_TYPES]
    if only_tasks:
        tasks = [t for t in tasks if t in set(only_tasks)]
    return tasks


def run(dataset_names, seeds, config, only_tasks=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from benchmark.models.space_gm_real_cv import cross_validate_fast

    summary_rows = []
    summary_path = MODEL_SAVE_ROOT / "summary.csv"
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict(orient="records")

    for name in dataset_names:
        ds = load_dataset(name, data_root=DATA_ROOT)
        tasks = binary_tasks(ds, only_tasks)
        if not tasks:
            continue

        for task in tasks:
            flags = [entry for entry in summary_rows if entry["dataset"] == name and entry["task"] == task]
            if flags:
                print(f"%%%  {task:24s} cv   (already done) %%%", flush=True)
                continue
            print(f"=== {name}  (binary task: {task}) ===", flush=True)
            
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]

            work_root = MODEL_SAVE_ROOT / f'{name}_{task}_cv'
            work_root.mkdir(parents=True, exist_ok=True)
            model_dir = work_root

            fm = cross_validate_fast(
                ds, task, config, seeds=seeds,
                model_dir=str(model_dir), work_root=str(work_root),
            )
            if not fm:
                print(f"  {task:24s} cv   (no evaluable folds)", flush=True)
                continue

            mean, sd = summarize_folds(fm, metric)
            summary_rows.append(dict(dataset=name, task=task, scheme="cv",
                                     metric=metric, mean=mean, sd=sd, n=len(fm)))
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        ds.clear_region_cache()
    return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset config stems (default: all)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these task ids (still filtered to binary)")
    
    # seeds to run (default: 0,1,2)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    
    # hyper-parameters (see benchmark.models.space_gm_real_cv.SpaceGMConfig)
    ap.add_argument("--device", default="cuda:1", help="cuda / cpu (default: auto)")
    ap.add_argument("--subgraph-size", type=int, default=3)
    ap.add_argument("--radius-um", type=float, default=75.0)
    ap.add_argument("--near-edge-um", type=float, default=20.0)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-subsample-ratio", type=float, default=0.1)
    args = ap.parse_args()

    check_runtime_deps()
    config = build_config(args)

    run(
        args.datasets or list_datasets(), args.seeds, config, only_tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
