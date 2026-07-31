#!/usr/bin/env python
"""Benchmark validation for native CytoCommunity Step4-derived features.

Input is the region-level CSV produced by ``extract_native_cyto_features.py``.
The script validates those fixed native TCN composition features with the same
patient-level CV and cohort-split protocol used by the other benchmark runners.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.models.linear import LinearClassifier, LinearCox
from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.utils.task_filter import is_selected_benchmark_task, should_skip_benchmark_task
from benchmark.validation.metrics import score_predictions, summarize_folds, PRIMARY_METRIC
from benchmark.validation.splits import safe_patient_kfold, stratify_column


def _nan_metrics(task_type: str) -> dict[str, float]:
    """Execute the nan metrics operation.
    
        Args:
            task_type (str): Prediction task category, such as classification or survival.
    
        Returns:
            dict[str, float]: The operation result.
    
    Args:
        task_type (str): Prediction task category, such as classification or survival."""
    if task_type == "survival":
        return {"c_index": float("nan")}
    if task_type in ("binary", "binary_classification"):
        return {"auc_roc": float("nan"), "avg_precision": float("nan"), "balanced_acc": float("nan")}
    return {"balanced_acc": float("nan"), "macro_auc": float("nan")}


def model_factory(task_cfg: dict, seed: int):
    """Execute the model factory operation.
    
        Args:
            task_cfg (dict): Configuration mapping for the current prediction task.
            seed (int): Random seed used for reproducibility.
    
        Returns:
            Any: The operation result.
    
    Args:
        task_cfg (dict): Configuration mapping for the current prediction task."""
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def select_feature_columns(features: pd.DataFrame, include_counts: bool, include_n_cells: bool) -> list[str]:
    """Select feature columns.
    
        Args:
            features (pd.DataFrame): Feature values used to fit or evaluate the model.
            include_counts (bool): Whether to include counts in the output.
            include_n_cells (bool): Whether to include n cells in the output.
    
        Returns:
            list[str]: The operation result.
    
    Args:
        features (pd.DataFrame): Feature values used to fit or evaluate the model."""
    frac_cols = sorted(c for c in features.columns if c.startswith("tcn_") and c.endswith("_frac"))
    cols = frac_cols + [c for c in ["tcn_entropy"] if c in features.columns]
    if include_counts:
        cols.extend(sorted(c for c in features.columns if c.startswith("tcn_") and c.endswith("_count")))
    if include_n_cells and "n_cells" in features.columns:
        cols.append("n_cells")
    if not cols:
        raise ValueError("No usable native CytoCommunity feature columns found.")
    return cols


def load_feature_table(path: str | Path, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Load feature table.
    
        Args:
            path (str | Path): Path to the input or output resource.
            feature_cols (list[str] | None): Names of columns used as model features.
    
        Returns:
            pd.DataFrame: The operation result.
    
    Args:
        path (str | Path): Filesystem path of the resource to load or save."""
    features = pd.read_csv(path)
    required = {"dataset", "region_id"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {sorted(missing)}")
    features["dataset"] = features["dataset"].astype(str)
    features["region_id"] = features["region_id"].astype(str)
    if features.duplicated(["dataset", "region_id"]).any():
        dup = features.loc[features.duplicated(["dataset", "region_id"], keep=False), ["dataset", "region_id"]]
        raise ValueError("Feature CSV has duplicated dataset/region_id rows, e.g.\n" + dup.head().to_string(index=False))
    if feature_cols is not None:
        keep = ["dataset", "region_id"] + feature_cols
        missing_cols = [c for c in feature_cols if c not in features.columns]
        if missing_cols:
            raise ValueError(f"Requested feature columns not found: {missing_cols}")
        features = features[keep]
    return features


def align_task_metadata(ds, task_id: str, dataset_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the align task metadata operation.
    
        Args:
            ds (Any): Dataset instance that supplies regions and task metadata.
            task_id (str): Unique identifier of the benchmark task.
            dataset_features (pd.DataFrame): Feature table associated with each dataset.
    
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The operation result.
    
    Args:
        ds (Any): Dataset instance that supplies regions and task metadata."""
    meta = ds.get_task_metadata(task_id).copy()
    meta["region_id"] = meta["region_id"].astype(str)
    available = set(dataset_features.index.astype(str))
    meta = meta[meta["region_id"].isin(available)].copy()
    if len(meta) == 0:
        return meta, dataset_features.iloc[0:0]
    X = dataset_features.loc[meta["region_id"]]
    return meta, X


def feature_table_cross_validate(
    ds,
    task_id: str,
    dataset_features: pd.DataFrame,
    seeds: list[int],
) -> list[dict]:
    """Execute the feature table cross validate operation.
    
        Args:
            ds (Any): Dataset instance that supplies regions and task metadata.
            task_id (str): Unique identifier of the benchmark task.
            dataset_features (pd.DataFrame): Feature table associated with each dataset.
            seeds (list[int]): Random seeds used for repeated benchmark runs.
    
        Returns:
            list[dict]: The operation result.
    
    Args:
        ds (Any): Dataset instance that supplies regions and task metadata."""
    vcfg = ds.validation_config
    n_folds = vcfg.get("n_folds", 5)
    patient_col = vcfg.get("patient_col", "patient_id")
    cv_filter = vcfg.get("cv_filter")
    task_cfg = ds.get_task_config(task_id)

    meta, _ = align_task_metadata(ds, task_id, dataset_features)
    if len(meta) == 0:
        return []
    if cv_filter:
        sub = meta.query(cv_filter)
        meta = sub if len(sub) else meta

    fold_metrics: list[dict] = []
    for seed in seeds:
        folds = safe_patient_kfold(meta, n_folds, patient_col, stratify_column(task_cfg), seed)
        if folds is None:
            continue
        for fold_i, (train_ids, val_ids) in enumerate(folds):
            train_ids = [str(x) for x in train_ids if str(x) in dataset_features.index]
            val_ids = [str(x) for x in val_ids if str(x) in dataset_features.index]
            if not train_ids or not val_ids:
                continue
            X_tr = dataset_features.loc[train_ids]
            X_va = dataset_features.loc[val_ids]
            y_tr = ds.build_target(list(X_tr.index), task_id)
            y_va = ds.build_target(list(X_va.index), task_id)

            try:
                model = model_factory(task_cfg, seed).fit(X_tr, y_tr)
                metrics = score_predictions(task_cfg, y_va, model.predict(X_va), getattr(model, "classes_", None))
            except Exception:
                if os.environ.get("BENCHMARK_RAISE_ERRORS"):
                    raise
                metrics = _nan_metrics(task_cfg["type"])
            metrics.update({"seed": seed, "fold": fold_i, "n_train": len(X_tr), "n_val": len(X_va)})
            fold_metrics.append(metrics)
    return fold_metrics


def feature_table_cohort_split_test(
    ds,
    task_id: str,
    gentest: dict,
    dataset_features: pd.DataFrame,
    seeds: list[int],
) -> list[dict]:
    """Execute the feature table cohort split test operation.
    
        Args:
            ds (Any): Dataset instance that supplies regions and task metadata.
            task_id (str): Unique identifier of the benchmark task.
            gentest (dict): Boolean mask identifying the held-out generalization set.
            dataset_features (pd.DataFrame): Feature table associated with each dataset.
            seeds (list[int]): Random seeds used for repeated benchmark runs.
    
        Returns:
            list[dict]: The operation result.
    
    Args:
        ds (Any): Dataset instance that supplies regions and task metadata."""
    cohort_col = ds.validation_config["cohort_col"]
    task_cfg = ds.get_task_config(task_id)
    meta, _ = align_task_metadata(ds, task_id, dataset_features)
    if len(meta) == 0:
        return []

    train_ids = meta[meta[cohort_col].isin(gentest["train"])]["region_id"].astype(str).tolist()
    test_ids = meta[meta[cohort_col].isin(gentest["test"])]["region_id"].astype(str).tolist()
    train_ids = [rid for rid in train_ids if rid in dataset_features.index]
    test_ids = [rid for rid in test_ids if rid in dataset_features.index]
    if not train_ids or not test_ids:
        return []

    X_tr = dataset_features.loc[train_ids]
    X_te = dataset_features.loc[test_ids]
    y_tr = ds.build_target(list(X_tr.index), task_id)
    y_te = ds.build_target(list(X_te.index), task_id)

    results: list[dict] = []
    for seed in seeds:
        try:
            model = model_factory(task_cfg, seed).fit(X_tr, y_tr)
            metrics = score_predictions(task_cfg, y_te, model.predict(X_te), getattr(model, "classes_", None))
        except Exception:
            if os.environ.get("BENCHMARK_RAISE_ERRORS"):
                raise
            metrics = _nan_metrics(task_cfg["type"])
        metrics.update({"seed": seed, "n_train": len(X_tr), "n_test": len(X_te)})
        results.append(metrics)
    return results


def run(
    dataset_names: list[str],
    seeds: list[int],
    features_csv: str | Path,
    data_root: str | None = None,
    include_counts: bool = False,
    include_n_cells: bool = False,
    only_selected_tasks: bool = False,
) -> pd.DataFrame:
    """Run.
    
        Args:
            dataset_names (list[str]): Names of datasets to process.
            seeds (list[int]): Random seeds used for repeated benchmark runs.
            features_csv (str | Path): CSV file containing precomputed features.
            data_root (str | None): Root directory containing data.
            include_counts (bool): Whether to include counts in the output.
            include_n_cells (bool): Whether to include n cells in the output.
            only_selected_tasks (bool): Whether to process only explicitly selected tasks.
    
        Returns:
            pd.DataFrame: The operation result.
    
    Args:
        dataset_names (list[str]): Names of datasets to process."""
    raw_features = load_feature_table(features_csv)
    feature_cols = select_feature_columns(raw_features, include_counts=include_counts, include_n_cells=include_n_cells)
    raw_features = load_feature_table(features_csv, feature_cols=feature_cols)
    print("Using native CytoCommunity feature columns: " + ", ".join(feature_cols))

    rows = []
    for name in dataset_names:
        ds_features = raw_features[raw_features["dataset"] == name].copy()
        if ds_features.empty:
            print(f"=== {name} ===")
            print("  skipped: no native CytoCommunity features")
            continue
        ds_features = ds_features.set_index("region_id")[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        ds = load_dataset(name, data_root=data_root)
        print(f"=== {name} ===")
        print(f"  feature regions: {len(ds_features)}")

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            if only_selected_tasks:
                if not is_selected_benchmark_task(name, task, "cv", metric):
                    print(f"  {task:24s} cv               {metric:12s} skipped")
                    continue
            elif should_skip_benchmark_task(name, task, "cv", metric):
                print(f"  {task:24s} cv               {metric:12s} skipped")
                continue
            fm = feature_table_cross_validate(ds, task, ds_features, seeds)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=name, task=task, scheme="cv", metric=metric, mean=mean, sd=sd, n=len(fm)))
            print(f"  {task:24s} cv               {metric:12s} {mean:.4f} +/- {sd:.4f}")

        for gt in ds.validation_config.get("generalization_tests", []):
            for task in gt.get("tasks", ds.task_ids):
                metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                if only_selected_tasks:
                    if not is_selected_benchmark_task(name, task, gt["name"], metric):
                        print(f"  {task:24s} {gt['name']:16s} {metric:12s} skipped")
                        continue
                elif should_skip_benchmark_task(name, task, gt["name"], metric):
                    print(f"  {task:24s} {gt['name']:16s} {metric:12s} skipped")
                    continue
                res = feature_table_cohort_split_test(ds, task, gt, ds_features, seeds)
                if not res:
                    continue
                mean, sd = summarize_folds(res, metric)
                rows.append(dict(dataset=name, task=task, scheme=gt["name"], metric=metric, mean=mean, sd=sd, n=len(res)))
                print(f"  {task:24s} {gt['name']:16s} {metric:12s} {mean:.4f} +/- {sd:.4f}")

        ds.clear_region_cache()
    return pd.DataFrame(rows)


def main() -> None:
    """Execute the main operation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-csv", required=True, help="CSV from extract_native_cyto_features.py")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output", default=str(_CODE / "results" / "cyto_community_native_benchmark.csv"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--include-counts", action="store_true", help="Also include raw tcn_*_count columns.")
    ap.add_argument("--include-n-cells", action="store_true", help="Also include n_cells as a feature.")
    ap.add_argument(
        "--only-selected-tasks",
        action="store_true",
        help="Run only the selected benchmark tasks listed in benchmark.utils.task_filter.",
    )
    ap.add_argument("--debug", action="store_true", help="Raise fold-level exceptions instead of recording NaN metrics.")
    args = ap.parse_args()

    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"

    df = run(
        args.datasets or list_datasets(),
        args.seeds,
        features_csv=args.features_csv,
        data_root=args.data_root,
        include_counts=args.include_counts,
        include_n_cells=args.include_n_cells,
        only_selected_tasks=args.only_selected_tasks,
    )
    df["score"] = df.apply(lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f}", axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
