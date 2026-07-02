#!/usr/bin/env python
"""Global mean biomarker-expression baseline over all datasets, tasks and schemes.

Feature: per region, the mean (over cells) of each marker on the arcsinh-normalised
expression. For each dataset it runs patient-level cross-validation for every task,
plus every declared cohort-split generalization test for its transferable tasks.

Cohort-split: only **overlapping biomarkers** may be used, since the two cohorts can
have different panels. We get this for free from `MeanExpressionFeaturizer`, which
learns the marker vocabulary as the *intersection* of the regions it is fit on — so we
pre-fit it on `train ∪ test` regions (the shared panel) and hand that pre-fitted
featurizer to `cohort_split_test`. Within-cohort CV just fits per fold (one panel).

    conda activate p3
    python scripts/run_global_expression_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset  # noqa: E402
from benchmark.features.basic_feats import MeanExpressionFeaturizer  # noqa: E402
from benchmark.models.linear import LinearClassifier, LinearCox  # noqa: E402
from benchmark.validation import (  # noqa: E402
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def shared_marker_featurizer(ds, task, gentest):
    """A MeanExpressionFeaturizer pre-fit on the SHARED (train ∩ test) marker panel.

    Returns None if either cohort has no labelled regions for the task. Fitting on the
    union of train+test regions makes ``markers_`` their intersection, so only
    overlapping biomarkers are used for the transfer.
    """
    cc = ds.validation_config["cohort_col"]
    meta = ds.get_task_metadata(task)
    tr = meta[meta[cc].isin(gentest["train"])]["region_id"].tolist()
    te = meta[meta[cc].isin(gentest["test"])]["region_id"].tolist()
    if not tr or not te:
        return None
    return MeanExpressionFeaturizer().fit(ds.load_regions(tr + te))


def run(dataset_names, seeds) -> pd.DataFrame:
    expr = lambda: MeanExpressionFeaturizer()     # CV: fit per fold (one cohort, one panel)
    rows = []
    for name in dataset_names:
        ds = load_dataset(name)
        print(f"=== {name} ===")

        # (A) cross-validation for every task
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            fm = cross_validate(ds, task, expr, model_factory, seeds=seeds)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        # (B) every cohort-split test, on the shared marker panel
        for gt in ds.validation_config.get("generalization_tests", []):
            for task in gt.get("tasks", ds.task_ids):
                feat = shared_marker_featurizer(ds, task, gt)   # pre-fit on shared markers
                if feat is None:
                    continue
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_split_test(ds, task, gt, feat, model_factory, seeds=seeds)
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"],
                                 metric=metric, mean=mean, sd=sd, n=len(res),
                                 n_markers=len(feat.markers_)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}"
                      f"  ({len(feat.markers_)} shared markers)")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "expression_benchmark.csv"))
    args = ap.parse_args()

    summary = run(args.datasets or list_datasets(), args.seeds)
    summary["score"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
