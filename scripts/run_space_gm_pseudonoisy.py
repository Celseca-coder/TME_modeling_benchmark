"""SPACE-GM on PseudoNoisyDataset labels.

Imaging still comes from TME_benchmark_data. Only the *targets* are the
noisy v2 ``*_label_v2`` / ``pseudo_label`` columns — never ``score_used``.
Clinical generalization tests are skipped.

Uses the genuine SPACE-GM runner
(:func:`benchmark.models.space_gm_real_cv.cross_validate_fast`), not the
hand-rolled GraphSAGE adapter in ``benchmark.models.space_gm``.

Output matches ``results/noisy_label_baselines/*.csv``::

    dataset,task,method,scheme,metric,mean,sd,n

Default task grid is the same 16 motif tasks as composition / sorbet
(selected motifs plus matched ``tumor_high`` / ``cd8_high`` controls on
every cohort).

    python scripts/run_space_gm_pseudonoisy.py \\
        --data-root /autofs/bal14/zqwu/CellularTables/TME_benchmark_data \\
        --label-root /autofs/nas8/tywang/tjzou/PseudoNoisyDataset \\
        --device cuda
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

from benchmark.motifs.local_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_LABEL_ROOT,
    PSEUDO_DATASETS,
    TASKS_BY_DATASET,
    attach_pseudo_tasks,
    resolve_label_path,
)
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation import PRIMARY_METRIC, summarize_folds  # noqa: E402


def check_runtime_deps() -> None:
    missing = [
        name for name in ("torch", "torch_geometric", "torch_scatter", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if importlib.util.find_spec("spacegm") is None:
        missing.append("spacegm")
    if missing:
        raise SystemExit(
            "SPACE-GM (real) is missing runtime dependencies: " + ", ".join(missing)
            + ".\nInstall them in the SPACE-GM environment first."
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


def _already_done(rows, dataset: str, task: str) -> bool:
    return any(
        row.get("dataset") == dataset and row.get("task") == task
        and row.get("scheme") == "cv"
        for row in rows
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--label-root", default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument(
        "--output",
        default=str(_CODE / "results" / "noisy_label_baselines" / "space_gm.csv"),
    )
    ap.add_argument(
        "--work-root",
        default=str(_CODE / "model_results" / "SPACE_GM_pseudo"),
        help="Per-task graph cache, weights, and cv_metrics.csv.",
    )
    ap.add_argument("--device", default=os.environ.get("SPACEGM_DEVICE", "cuda"))
    ap.add_argument("--subgraph-size", type=int, default=3)
    ap.add_argument("--radius-um", type=float, default=75.0)
    ap.add_argument("--near-edge-um", type=float, default=20.0)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-subsample-ratio", type=float, default=0.1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"
    if not str(args.data_root or "").strip():
        args.data_root = DEFAULT_DATA_ROOT
    if not str(args.label_root or "").strip():
        args.label_root = DEFAULT_LABEL_ROOT

    check_runtime_deps()
    from benchmark.models.space_gm_real_cv import cross_validate_fast

    config = build_config(args)
    names = args.datasets or list(PSEUDO_DATASETS)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(out).to_dict(orient="records") if out.exists() else []
    for name in names:
        ds = load_dataset(name, data_root=args.data_root)
        tasks = attach_pseudo_tasks(ds, name, args.label_root, args.tasks)
        label_path = resolve_label_path(name, args.label_root)
        print(
            f"=== {name}  method=space_gm  images={ds._root}  labels={label_path} ===",
            flush=True,
        )
        print(f"  pseudo tasks: {tasks}", flush=True)
        wanted = args.tasks or TASKS_BY_DATASET[name]
        for task in wanted:
            if task not in ds.task_ids:
                print(f"  skip {task}: not registered", flush=True)
                continue
            if _already_done(rows, name, task):
                print(f"  {task:32s} cv (already done)", flush=True)
                continue
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            task_root = work_root / f"{name}_{task}_cv"
            task_root.mkdir(parents=True, exist_ok=True)
            folds = cross_validate_fast(
                ds, task, config, seeds=args.seeds,
                model_dir=str(task_root), work_root=str(task_root),
                normalize=True,
            )
            if not folds:
                print(f"  skip {task}: no labelled folds", flush=True)
                continue
            mean, sd = summarize_folds(folds, metric)
            rows.append(dict(
                dataset=name, task=task, method="space_gm", scheme="cv",
                metric=metric, mean=mean, sd=sd, n=len(folds),
            ))
            pd.DataFrame(rows).to_csv(out, index=False)
            print(
                f"  {task:32s} {metric:10s} {mean:.4f} +/- {sd:.4f}",
                flush=True,
            )
        ds.clear_region_cache()

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
