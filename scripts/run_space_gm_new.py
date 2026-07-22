#!/usr/bin/env python
"""Run the genuine SPACE-GM model on **every modeled scenario**.

Sweeps all datasets (or a chosen subset), picks out the ``binary_classification``,
``survival`` and ``multiclass_classification`` tasks, and for each runs *both*
validation schemes with the amortized helpers in
:mod:`benchmark.models.space_gm_real_cv` (each ``CellularGraphDataset`` is built
once and sliced, instead of re-featurized per fold):

* **cv** — patient-level cross-validation via
  :func:`~benchmark.models.space_gm_real_cv.cross_validate_fast` (one dataset over
  all CV regions, a fresh model per fold).
* **gentest** — every cohort-generalization test declared in a dataset's
  ``validation_config['generalization_tests']`` via
  :func:`~benchmark.models.space_gm_real_cv.cohort_generalization_fast` (two
  datasets — train / test cohort — sharing the training cohort's vocabulary).

Binary tasks use a single BCE head (primary metric ``auc_roc``); survival tasks
use the Cox SGD head (primary metric ``c_index``); multiclass tasks use one BCE
head per class (one-vs-rest, primary metric ``balanced_acc``).

Per scenario it writes weights + a live ``cv_metrics.csv`` / ``run_metrics.csv``
under a ``<dataset>_<task>_cv`` (cv) or ``<gentest>_<task>_cv`` (gentest) work
directory below ``MODEL_SAVE_ROOT``, and appends one aggregated row (mean/sd of
the primary metric across folds/runs) to a resumable ``summary.csv``.

Requires the ``spacegm`` package (and torch / torch_geometric / torch_scatter /
scikit-learn) in the active environment, e.g. conda env ``p3``.

Example
-------
    # everything, GPU, manuscript-ish budget
    python scripts/run_space_gm_binary_cv.py \
        --device cuda --num-iterations 50000 --emb-dim 512

    # one dataset, quick smoke test
    python scripts/run_space_gm_binary_cv.py --datasets hnc_wu2022 --seeds 0 \
        --num-iterations 300 --device cpu

    # cross-validation only (skip cohort-generalization tests)
    python scripts/run_space_gm_binary_cv.py --schemes cv
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
SURVIVAL_TYPES = {"survival"}
MULTICLASS_TYPES = {"multiclass_classification", "multiclass"}
MODELED_TYPES = BINARY_TYPES | SURVIVAL_TYPES | MULTICLASS_TYPES

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


def modeled_tasks(ds, only_tasks=None) -> list[str]:
    """Binary-classification, survival and multiclass-classification tasks."""
    tasks = [t for t in ds.task_ids if ds.get_task_config(t)["type"] in MODELED_TYPES]
    if only_tasks:
        tasks = [t for t in tasks if t in set(only_tasks)]
    return tasks


def _already_done(summary_rows, dataset, task, scheme) -> bool:
    return any(
        entry["dataset"] == dataset and entry["task"] == task
        and entry.get("scheme") == scheme
        for entry in summary_rows
    )


def run(dataset_names, seeds, config, only_tasks=None, schemes=("cv", "gentest")):
    from benchmark.models.space_gm_real_cv import (
        cross_validate_fast, cohort_generalization_fast)

    summary_rows = []
    summary_path = MODEL_SAVE_ROOT / "summary.csv"
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict(orient="records")

    def record(dataset, task, scheme, metric, fold_metrics):
        mean, sd = summarize_folds(fold_metrics, metric)
        summary_rows.append(dict(dataset=dataset, task=task, scheme=scheme,
                                 metric=metric, mean=mean, sd=sd, n=len(fold_metrics)))
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"  {task:24s} {scheme:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}", flush=True)

    for name in dataset_names:
        ds = load_dataset(name, data_root=DATA_ROOT)
        tasks = modeled_tasks(ds, only_tasks)
        if not tasks:
            continue

        # (A) cross-validation
        if "cv" in schemes:
            for task in tasks:
                if _already_done(summary_rows, name, task, "cv"):
                    print(f"%%%  {task:24s} cv   (already done) %%%", flush=True)
                    continue
                ttype = ds.get_task_config(task)["type"]
                print(f"=== {name}  ({ttype} task: {task}, scheme: cv) ===", flush=True)
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]

                work_root = MODEL_SAVE_ROOT / f"{name}_{task}_cv"
                work_root.mkdir(parents=True, exist_ok=True)

                fm = cross_validate_fast(
                    ds, task, config, seeds=seeds,
                    model_dir=str(work_root), work_root=str(work_root),
                )
                if not fm:
                    print(f"  {task:24s} cv   (no evaluable folds)", flush=True)
                    continue
                record(name, task, "cv", metric, fm)

        # (B) cohort-generalization tests
        if "gentest" in schemes:
            modeled_set = set(tasks)
            for gt in ds.validation_config.get("generalization_tests", []):
                gt_tasks = [t for t in gt.get("tasks", tasks) if t in modeled_set]
                for task in gt_tasks:
                    scheme = gt["name"]
                    if _already_done(summary_rows, name, task, scheme):
                        print(f"%%%  {task:24s} {scheme:16s} (already done) %%%", flush=True)
                        continue
                    ttype = ds.get_task_config(task)["type"]
                    print(f"=== {name}  ({ttype} task: {task}, scheme: {scheme}) ===", flush=True)
                    metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]

                    work_root = MODEL_SAVE_ROOT / f"{scheme}_{task}_cv"
                    work_root.mkdir(parents=True, exist_ok=True)

                    rm = cohort_generalization_fast(
                        ds, task, gt, config, seeds=seeds,
                        model_dir=str(work_root), work_root=str(work_root),
                    )
                    if not rm:
                        print(f"  {task:24s} {scheme:16s} (no evaluable regions)", flush=True)
                        continue
                    record(name, task, scheme, metric, rm)

        ds.clear_region_cache()
    return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset config stems (default: all)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these task ids (still filtered to modeled task types)")
    ap.add_argument("--schemes", nargs="*", default=["cv", "gentest"],
                    choices=["cv", "gentest"],
                    help="which validation schemes to run (default: both)")

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
        args.datasets or list_datasets(), args.seeds, config,
        only_tasks=args.tasks, schemes=args.schemes,
    )


if __name__ == "__main__":
    main()
