#!/usr/bin/env python
"""Run the genuine SPACE-GM model over the benchmark with **amortized datasets**.

Same tasks / schemes / metrics as ``run_space_gm_real.py``, but each
``spacegm.data.CellularGraphDataset`` is built and processed **once** per CV run
(and once per cohort for generalization tests) instead of once per fold — see
:mod:`benchmark.models.space_gm_real_cv` for why that is leakage-free. This is
dramatically faster when the featurization / graph processing dominates.

Requires the ``spacegm`` package to be installed in the active environment.

Example
-------
    # everything, GPU, manuscript-ish budget, saving weights
    python scripts/run_space_gm_real_cv.py \
        --device cuda --num-iterations 50000 --emb-dim 512 \
        --model-dir results/space_gm_real_weights

    # quick smoke test on one dataset/seed
    python scripts/run_space_gm_real_cv.py --datasets hnc_wu2022 --seeds 0 \
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


def check_runtime_deps():
    missing = [
        name for name in ("torch", "torch_geometric", "torch_scatter", "lifelines", "sklearn")
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
        emb_dim=args.emb_dim,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_subsample_ratio=args.eval_subsample_ratio,
        device=args.device,
        model_dir=args.model_dir,
    )


def run(dataset_names, seeds, config, *, data_root=None, model_dir=None, work_root=None) -> pd.DataFrame:
    from benchmark.models.space_gm_real_cv import cross_validate_fast, cohort_generalization_fast

    rows = []
    for name in dataset_names:
        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===", flush=True)

        # (A) patient-level cross-validation (one dataset per task, processed once)
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            fm = cross_validate_fast(ds, task, config, seeds=seeds,
                                     model_dir=model_dir, work_root=work_root)
            if not fm:
                continue
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}", flush=True)

        # (B) cohort -> cohort generalization tests (two datasets per test)
        for gt in ds.validation_config.get("generalization_tests", []):
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_generalization_fast(ds, task, gt, config, seeds=seeds,
                                                 model_dir=model_dir, work_root=work_root)
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}", flush=True)

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset config stems")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "space_gm_real_cv_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    # hyper-parameters (see benchmark.models.space_gm_real.SpaceGMConfig)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--subgraph-size", type=int, default=3)
    ap.add_argument("--radius-um", type=float, default=75.0)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--num-iterations", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-subsample-ratio", type=float, default=0.3)
    ap.add_argument("--model-dir", default=None,
                    help="if set, save trained weights (+vocab.json) under this dir, "
                         "one unique subfolder per (task, fold, seed)")
    ap.add_argument("--work-root", default=None,
                    help="where to build/process the temporary datasets "
                         "(default: system temp); point at fast local disk for big cohorts")
    args = ap.parse_args()

    check_runtime_deps()
    config = build_config(args)
    df = run(args.datasets or list_datasets(), args.seeds, config,
             data_root=args.data_root, model_dir=args.model_dir, work_root=args.work_root)

    if len(df):
        df["score"] = df.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
