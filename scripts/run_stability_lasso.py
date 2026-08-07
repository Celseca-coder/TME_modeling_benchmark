#!/usr/bin/env python
"""Run fold/seed/bootstrap stability Lasso on region-level feature tables.

Examples
--------
# Any precomputed local, UTAG, CN, or MIL-aggregated region features:
python scripts/run_stability_lasso.py --datasets hnc_wu2022 --tasks response \
  --feature-source precomputed --features-csv results/my_region_features.csv

# Existing global composition or patch-MIL features:
python scripts/run_stability_lasso.py --datasets hnc_wu2022 \
  --feature-source composition --cell-type-col cell_type_uniform
python scripts/run_stability_lasso.py --datasets hnc_wu2022 \
  --feature-source patch --window-size 100 --step 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.basic_feats import CompositionFeaturizer
from benchmark.features.patch_feats import PatchBasedFeaturizer
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation.stability_lasso import stability_lasso_cv


def _load_precomputed(path: str, region_id_col: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    if region_id_col not in table.columns:
        raise ValueError(
            f"{path} has no region ID column {region_id_col!r}; "
            f"available columns: {list(table.columns)}"
        )
    return table.set_index(region_id_col)


def _featurizer_factory(args):
    if args.feature_source == "composition":
        return lambda: CompositionFeaturizer(cell_type_col=args.cell_type_col)
    if args.feature_source == "patch":
        return lambda: PatchBasedFeaturizer(
            window_size_um=args.window_size,
            step_um=args.step,
            feature_groups=tuple(args.feature_groups),
            cell_type_col=args.cell_type_col,
            aggregations=tuple(args.aggregations),
            quantiles=tuple(args.quantiles),
            min_cells_per_window=args.min_cells,
            use_tissue_mask=not args.no_tissue_mask,
        )
    return None


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def run(args) -> dict[str, pd.DataFrame]:
    precomputed = None
    if args.feature_source == "precomputed":
        if not args.features_csv:
            raise ValueError("--features-csv is required for precomputed features")
        precomputed = _load_precomputed(args.features_csv, args.region_id_col)

    fold_tables = []
    seed_tables = []
    summary_tables = []
    bootstrap_tables = []
    for dataset_name in args.datasets or list_datasets():
        dataset = load_dataset(dataset_name, data_root=args.data_root)
        tasks = args.tasks or dataset.task_ids
        print(f"=== {dataset_name} ===", flush=True)
        for task in tasks:
            task_cfg = dataset.get_task_config(task)
            if task_cfg["type"] == "survival":
                print(f"  {task}: skipped (L1 Cox is not implemented)", flush=True)
                continue
            print(f"  {task}: fitting stability Lasso", flush=True)
            kwargs = {"features": precomputed} if precomputed is not None else {
                "featurizer": _featurizer_factory(args)
            }
            result = stability_lasso_cv(
                dataset,
                task,
                seeds=args.seeds,
                n_folds=args.n_folds,
                n_bootstrap=args.n_bootstrap,
                lambda_value=args.lambda_value,
                patient_col=args.patient_col,
                cv_filter=args.cv_filter,
                normalize=not args.no_normalize,
                class_weight=None if args.no_class_weight else "balanced",
                max_iter=args.max_iter,
                selected_tolerance=args.selected_tolerance,
                ci_level=args.ci_level,
                seed_fold_frequency_threshold=args.seed_fold_frequency_threshold,
                **kwargs,
            )
            fold_tables.append(result.fold_coefficients)
            seed_tables.append(result.seed_summary)
            summary_tables.append(result.feature_summary)
            if args.save_bootstrap_detail:
                bootstrap_tables.append(result.bootstrap_coefficients)
        dataset.clear_region_cache()

    return {
        "fold_coefficients": _concat(fold_tables),
        "seed_summary": _concat(seed_tables),
        "feature_summary": _concat(summary_tables),
        "bootstrap_coefficients": _concat(bootstrap_tables),
    }


def _write_results(tables: dict[str, pd.DataFrame], output_prefix: str) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.empty and name == "bootstrap_coefficients":
            continue
        output = prefix.parent / f"{prefix.name}_{name}.csv"
        table.to_csv(output, index=False)
        print(f"Wrote {output} ({len(table)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument(
        "--lambda-value", type=float, default=1.0,
        help="L1 strength; the sklearn inverse strength is C=1/lambda.",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--selected-tolerance", type=float, default=1e-10)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--seed-fold-frequency-threshold", type=float, default=0.5,
        help="Within-seed fold frequency required to call a feature selected.",
    )
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--cv-filter", default=None)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--feature-source",
        choices=["precomputed", "composition", "patch"],
        default="precomputed",
    )
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--region-id-col", default="region_id")
    parser.add_argument(
        "--cell-type-col", default="cell_type",
        help="Use an UTAG/domain column here for domain composition.",
    )
    parser.add_argument("--window-size", type=float, default=100)
    parser.add_argument("--step", type=float, default=50)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=["composition", "expression"],
        default=["composition"],
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=["mean", "max", "min", "std", "quantile"],
        default=["mean", "max", "std", "quantile"],
    )
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--no-tissue-mask", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default=str(_CODE / "results" / "stability_lasso"),
        help="Produces *_fold_coefficients.csv, *_seed_summary.csv and *_feature_summary.csv.",
    )
    parser.add_argument(
        "--save-bootstrap-detail",
        action="store_true",
        help="Also save every bootstrap coefficient (can be a large file).",
    )
    args = parser.parse_args()
    _write_results(run(args), args.output_prefix)


if __name__ == "__main__":
    main()
