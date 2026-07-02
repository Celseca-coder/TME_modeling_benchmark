#!/usr/bin/env python
"""Cell-type density baseline over all datasets, tasks and schemes.

Feature: per region, physical **densities (cells per mm^2)** computed from the tissue and
tumour masks carried on each RegionData (the exported ``tissue_polygons.geojson``):

  * ``tissue_density::<cell_type>`` — each cell type's density in the tissue foreground;
  * ``tumor_area_ratio``           — tumour area / tissue area;
  * ``tumor_density::<cell_type>`` — each cell type's density inside the tumour mask
                                     (only cells within that mask).

For each dataset it runs patient-level cross-validation (`cross_validate`) for every task,
plus every declared cohort-split generalization test (`cohort_split_test`) for its
transferable tasks (using each test's `cell_type_col`, e.g. `cell_type_uniform`, so the
features transfer across cohorts). Writes a summary CSV.

    conda activate p3
    python scripts/run_celltype_density_baseline.py
    python scripts/run_celltype_density_baseline.py --datasets nsclc_aung2025 luad_sorin2023

Requires the polygon files to have been exported (section 8 of the
tissue_area_estimation notebook). Regions missing a tissue mask contribute NaN features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset  # noqa: E402
from benchmark.features.density_feats import CellTypeDensityFeaturizer  # noqa: E402
from benchmark.models.linear import LinearClassifier, LinearCox  # noqa: E402
from benchmark.validation import (  # noqa: E402
    cross_validate, cohort_split_test, summarize_folds, PRIMARY_METRIC,
)


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def run(dataset_names, seeds) -> pd.DataFrame:
    dens = lambda: CellTypeDensityFeaturizer()      # CV: fine cell_type, fit per fold
    rows = []
    for name in dataset_names:
        ds = load_dataset(name)
        print(f"=== {name} ===")

        # (A) cross-validation for every task
        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            fm = cross_validate(ds, task, dens, model_factory, seeds=seeds, normalize=False)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        # (B) every cohort-split generalization test
        for gt in ds.validation_config.get("generalization_tests", []):
            feat = lambda c=gt.get("cell_type_col", "cell_type"): CellTypeDensityFeaturizer(cell_type_col=c)
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                res = cohort_split_test(ds, task, gt, feat, model_factory, seeds=seeds, normalize=False)
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
    ap.add_argument("--output", default=str(_CODE / "results" / "celltype_density_benchmark.csv"))
    args = ap.parse_args()

    summary = run(args.datasets or list_datasets(), args.seeds)
    summary["score"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
