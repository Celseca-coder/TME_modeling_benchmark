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
from datetime import datetime
import logging
import re
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.basic_feats import CompositionFeaturizer, MeanExpressionFeaturizer
from benchmark.features.combined import CombinedCompositionExpressionFeaturizer
from benchmark.features.density_feats import CellTypeDensityFeaturizer
from benchmark.features.mixing import MixingFeaturizer
from benchmark.features.patch_feats import PatchBasedFeaturizer
from benchmark.features.point_pattern import PointPatternFeaturizer
from benchmark.features.spatial_distance import SpatialDistanceFeaturizer
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation.stability_lasso import stability_lasso_cv

LOGGER = logging.getLogger("stability_lasso")


def _configure_logging(
    log_file: str | None,
    datasets: list[str] | None,
    tasks: list[str] | None,
) -> Path:
    if log_file:
        path = Path(log_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_label = "-".join(datasets or ["all-datasets"])
        task_label = "-".join(tasks or ["all-tasks"])
        analysis_label = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", f"{dataset_label}_{task_label}"
        ).strip("-")
        analysis_label = analysis_label[:120]
        path = _CODE / "log" / f"{analysis_label}_lasso_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    return path


def _load_precomputed(path: str, region_id_col: str) -> pd.DataFrame:
    feature_path = Path(path).expanduser()
    if not feature_path.is_absolute():
        feature_path = Path.cwd() / feature_path
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Precomputed feature table not found: {feature_path}\n"
            "The name 'features.csv' in the documentation is only an example. "
            "Supply the real region-by-feature CSV path, or use "
            "'--feature-source composition' / '--feature-source patch' "
            "to generate features directly from the configured dataset."
        )
    LOGGER.info("Loading precomputed features from %s", feature_path)
    table = pd.read_csv(feature_path)
    if region_id_col not in table.columns:
        raise ValueError(
            f"{feature_path} has no region ID column {region_id_col!r}; "
            f"available columns: {list(table.columns)}"
        )
    return table.set_index(region_id_col)


def _featurizer_factory(args, dataset):
    if args.feature_source == "composition":
        return lambda: CompositionFeaturizer(cell_type_col=args.cell_type_col)
    if args.feature_source == "expression":
        return lambda: MeanExpressionFeaturizer()
    if args.feature_source == "composition-expression":
        return lambda: CombinedCompositionExpressionFeaturizer(
            cell_type_col=args.cell_type_col
        )
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
    if args.feature_source == "density":
        return lambda: CellTypeDensityFeaturizer(cell_type_col=args.cell_type_col)
    if args.feature_source == "spatial-distance":
        return lambda: SpatialDistanceFeaturizer(
            cell_type_col=args.cell_type_col,
            k=args.distance_k,
            use_tissue_mask=not args.no_tissue_mask,
        )
    if args.feature_source == "point-pattern":
        return lambda: PointPatternFeaturizer(
            radii=args.point_pattern_radii,
            metrics=tuple(args.point_pattern_metrics),
            by_type=args.point_pattern_by_type,
            cell_type_col=args.cell_type_col,
            use_tissue_mask=not args.no_tissue_mask,
        )
    if args.feature_source == "mixing":
        return lambda: MixingFeaturizer(
            cell_type_col=args.cell_type_col,
            k_neighbors=args.mixing_k,
            use_tissue_mask=not args.no_tissue_mask,
        )
    if args.feature_source == "utag":
        from models.utag.process_local_data import DEFAULT_CACHE, UTAGFeaturizer

        utag_args = argparse.Namespace(
            **{**vars(args), "cache_dir": args.cache_dir or str(DEFAULT_CACHE)}
        )
        return lambda: UTAGFeaturizer(dataset, utag_args)
    if args.feature_source == "kronos":
        from models.KRONOS.process_local_data import (
            DEFAULT_CACHE,
            DEFAULT_MODEL_CACHE,
            KronosFeaturizer,
        )

        kronos_args = argparse.Namespace(
            **{
                **vars(args),
                "cache_dir": args.cache_dir or str(DEFAULT_CACHE),
                "model_cache": args.model_cache or str(DEFAULT_MODEL_CACHE),
                "hf_repo": args.hf_repo or "MahmoodLab/KRONOS",
            }
        )
        return lambda: KronosFeaturizer(dataset, kronos_args)
    if args.feature_source == "eva":
        from models.Eva.process_local_data import DEFAULT_CACHE, EvaFeaturizer

        cache_dir = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE
        return lambda: EvaFeaturizer(
            dataset,
            image_mode=args.image_mode,
            device=args.device,
            checkpoint=args.checkpoint,
            hf_repo=args.hf_repo or "yandrewl/Eva",
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            stride=args.image_stride,
            min_foreground=args.min_foreground,
            max_patches=args.max_patches,
            cls=args.eva_cls,
        )
    return None


def _normalize_regions(args) -> bool:
    """Whether RegionData expression should use dataset-level normalization."""
    if args.no_normalize:
        return False
    # These adapters either do not consume expression, or apply their own
    # transform/raster scaling. UTAG in particular must not receive data that
    # have already been arcsinh-normalized by the dataset loader.
    return args.feature_source not in {
        "density",
        "spatial-distance",
        "point-pattern",
        "mixing",
        "utag",
        "kronos",
        "eva",
    }


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _restrict_to_available_regions(dataset) -> None:
    """Drop metadata rows whose region directory is absent (in memory only)."""
    metadata = dataset.get_metadata()
    available = metadata["region_id"].astype(str).map(
        lambda region_id: dataset.region_dir(region_id).is_dir()
    )
    if not available.any():
        raise FileNotFoundError(
            f"No region directories from {dataset._root} match its metadata"
        )
    dropped = int((~available).sum())
    if dropped:
        dataset._metadata = metadata.loc[available].reset_index(drop=True)
        LOGGER.warning(
            "Dataset %s: skipping %d metadata region(s) with no directory",
            dataset.name,
            dropped,
        )


def _load_cytocommunity_features(dataset_name: str, dataset, root: str) -> pd.DataFrame:
    """Aggregate one dataset's cell-level TCN labels into region proportions."""
    from scripts.export_cached_model_features import (
        _region_lookup,
        _validate_table,
        aggregate_cytocommunity,
    )

    rendered = str(root).format(dataset=dataset_name)
    input_root = Path(rendered).expanduser()
    if "{dataset}" not in str(root) and (input_root / dataset_name).is_dir():
        input_root = input_root / dataset_name
    if not input_root.is_dir():
        raise FileNotFoundError(
            f"CytoCommunity result directory not found for {dataset_name}: {input_root}"
        )

    region_ids = dataset.get_metadata()["region_id"].astype(str).tolist()
    features = aggregate_cytocommunity(input_root, _region_lookup(region_ids))
    features = _validate_table(features)
    missing = len(set(region_ids) - set(features.index))
    LOGGER.info(
        "CytoCommunity %s: regions=%d/%d, missing=%d, TCN features=%d, root=%s",
        dataset_name,
        len(features),
        len(region_ids),
        missing,
        features.shape[1],
        input_root,
    )
    return features


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
        dataset_features = precomputed
        if args.feature_source == "cytocommunity":
            try:
                dataset_features = _load_cytocommunity_features(
                    dataset_name, dataset, args.cytocommunity_root
                )
            except Exception:
                if not args.continue_on_error:
                    raise
                LOGGER.exception(
                    "CytoCommunity aggregation failed for %s; continuing",
                    dataset_name,
                )
                continue
        elif precomputed is None:
            _restrict_to_available_regions(dataset)
        tasks = args.tasks or dataset.task_ids
        LOGGER.info("Dataset: %s", dataset_name)
        for task in tasks:
            if task not in dataset.task_ids:
                LOGGER.warning(
                    "Task %s skipped for %s; available tasks: %s",
                    task,
                    dataset_name,
                    dataset.task_ids,
                )
                continue
            task_cfg = dataset.get_task_config(task)
            if task_cfg["type"] == "survival":
                LOGGER.info("Task %s skipped: L1 Cox is not implemented", task)
                continue
            LOGGER.info("Task %s: fitting stability Lasso", task)
            kwargs = {"features": dataset_features} if dataset_features is not None else {
                "featurizer": _featurizer_factory(args, dataset)
            }
            try:
                result = stability_lasso_cv(
                    dataset,
                    task,
                    seeds=args.seeds,
                    n_folds=args.n_folds,
                    n_bootstrap=args.n_bootstrap,
                    lambda_value=args.lambda_value,
                    patient_col=args.patient_col,
                    cv_filter=args.cv_filter,
                    normalize=_normalize_regions(args),
                    class_weight=None if args.no_class_weight else "balanced",
                    max_iter=args.max_iter,
                    selected_tolerance=args.selected_tolerance,
                    ci_level=args.ci_level,
                    seed_fold_frequency_threshold=args.seed_fold_frequency_threshold,
                    **kwargs,
                )
            except Exception:
                if not args.continue_on_error:
                    raise
                LOGGER.exception(
                    "Dataset %s task %s failed; continuing", dataset_name, task
                )
                continue
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
        LOGGER.info("Wrote %s (%d rows)", output, len(table))


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
        "--continue-on-error",
        action="store_true",
        help="Log a failed dataset/task and continue with the remaining analyses.",
    )
    parser.add_argument(
        "--feature-source",
        choices=[
            "precomputed",
            "composition",
            "expression",
            "composition-expression",
            "patch",
            "density",
            "spatial-distance",
            "point-pattern",
            "mixing",
            "utag",
            "kronos",
            "eva",
            "cytocommunity",
        ],
        default="composition",
    )
    parser.add_argument(
        "--features-csv",
        default=None,
        help="Real region-by-feature CSV path; required only for --feature-source precomputed.",
    )
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
    parser.add_argument("--distance-k", type=int, default=1)
    parser.add_argument(
        "--point-pattern-radii",
        nargs="+",
        type=float,
        default=[10, 20, 50, 100, 200],
    )
    parser.add_argument(
        "--point-pattern-metrics",
        nargs="+",
        choices=["K", "L", "pcf", "variogram"],
        default=["K", "L"],
    )
    parser.add_argument("--point-pattern-by-type", action="store_true")
    parser.add_argument("--mixing-k", type=int, default=10)

    # UTAG options. Domain clustering is fitted independently in each CV fold.
    parser.add_argument(
        "--feature-mode",
        choices=["message-passing", "domains", "combined"],
        default="combined",
    )
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument(
        "--normalization-mode", choices=["l1_norm", "sum"], default="l1_norm"
    )
    parser.add_argument(
        "--coordinate-mode", choices=["auto", "um", "native"], default="auto"
    )
    parser.add_argument(
        "--expression-transform",
        choices=["none", "arcsinh", "log1p"],
        default="arcsinh",
    )
    parser.add_argument("--arcsinh-cofactor", type=float, default=5.0)
    parser.add_argument("--n-domains", type=int, default=10)
    parser.add_argument("--max-fit-cells", type=int, default=100000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for train-fold UTAG domain clustering.",
    )

    # Frozen KRONOS/Eva embedding options. Existing caches are reused.
    parser.add_argument(
        "--image-mode",
        choices=["native", "rasterized", "auto"],
        default="auto",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-stride", type=int, default=224)
    parser.add_argument("--min-foreground", type=float, default=0.01)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--eva-cls", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Embedding/message-passing cache root; model-specific default if omitted.",
    )
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--cfg-path", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--marker-metadata", default=None)
    parser.add_argument(
        "--model-type", choices=["vits16", "vitl16"], default="vits16"
    )
    parser.add_argument("--token-overlap", action="store_true")
    parser.add_argument("--max-value", type=float, default=65535.0)
    parser.add_argument("--raster-radius", type=int, default=2)
    parser.add_argument(
        "--cytocommunity-root",
        default=str(
            _CODE
            / "model_results"
            / "CytoCommunity"
            / "native_local_runs_cutoff02"
        ),
        help=(
            "Root containing one subdirectory per registry dataset, or a path "
            "template containing '{dataset}'."
        ),
    )
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
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log path. Default: <project>/log/<dataset>_<task>_lasso_<timestamp>.log",
    )
    args = parser.parse_args()
    log_path = _configure_logging(args.log_file, args.datasets, args.tasks)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info(
        "Feature source: %s; datasets: %s; tasks: %s; seeds: %s",
        args.feature_source,
        args.datasets or "all",
        args.tasks or "all classification tasks",
        args.seeds,
    )
    try:
        _write_results(run(args), args.output_prefix)
    except Exception:
        LOGGER.exception("Stability Lasso failed")
        raise


if __name__ == "__main__":
    main()
