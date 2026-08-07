"""Patient-level cross-validated Lasso coefficient stability analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from benchmark.models.linear import LinearClassifier
from benchmark.validation.crossval import _resolve_featurizer
from benchmark.validation.splits import safe_patient_kfold, stratify_column


@dataclass
class LassoStabilityResult:
    """Long-form coefficient tables and their stability summaries."""

    fold_coefficients: pd.DataFrame
    seed_summary: pd.DataFrame
    feature_summary: pd.DataFrame
    bootstrap_coefficients: pd.DataFrame


def _metadata_with_region_id(dataset, task_id: str) -> pd.DataFrame:
    meta = dataset.get_task_metadata(task_id).copy()
    if "region_id" not in meta.columns:
        meta = meta.reset_index(names="region_id")
    meta["region_id"] = meta["region_id"].astype(str)
    return meta


def _validate_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not features.index.is_unique:
        raise ValueError("features index must contain unique region_id values")
    if not features.columns.is_unique:
        raise ValueError("feature names must be unique")
    out = features.copy()
    out.index = out.index.astype(str)
    out.index.name = "region_id"
    try:
        out = out.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("all feature columns must be numeric") from exc
    return out


def _coefficient_rows(
    model: LinearClassifier,
    all_features: Sequence[str],
    *,
    selected_tolerance: float,
) -> list[dict[str, Any]]:
    """Expand a fitted model to one row per class and original feature."""
    fitted_coefficient = np.asarray(model._clf.coef_, dtype=float)
    coefficient = np.zeros(
        (fitted_coefficient.shape[0], len(all_features)), dtype=float
    )
    positions = {name: i for i, name in enumerate(all_features)}
    for retained_i, name in enumerate(model._keep):
        coefficient[:, positions[name]] = fitted_coefficient[:, retained_i]

    if coefficient.shape[0] == 1:
        class_labels = [model.classes_[1]]
    else:
        class_labels = list(model.classes_)

    rows: list[dict[str, Any]] = []
    for class_i, class_label in enumerate(class_labels):
        for feature_i, feature in enumerate(all_features):
            value = float(coefficient[class_i, feature_i])
            rows.append({
                "class": str(class_label),
                "feature": str(feature),
                "coefficient": value,
                "selected": abs(value) > selected_tolerance,
            })
    return rows


def _fit_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    lambda_value: float,
    class_weight: str | None,
    max_iter: int,
    selected_tolerance: float,
) -> list[dict[str, Any]]:
    if lambda_value <= 0:
        raise ValueError("lambda_value must be > 0")
    from sklearn.linear_model import LogisticRegression

    model = LinearClassifier(
        seed=seed,
        C=1.0 / lambda_value,
        l1_ratio=1.0,
        class_weight=class_weight,
        max_iter=max_iter,
    )
    X_scaled = model._prep_fit(X.loc[list(y.index)])
    model._clf = LogisticRegression(
        C=1.0 / lambda_value,
        penalty="l1",
        solver="saga",
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    ).fit(X_scaled, y.values)
    model.classes_ = model._clf.classes_
    return _coefficient_rows(
        model, list(X.columns), selected_tolerance=selected_tolerance
    )


def _cluster_bootstrap_positions(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample patients with replacement and retain all regions per draw."""
    unique_groups = pd.unique(groups)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    group_positions = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    return np.concatenate([group_positions[group] for group in sampled])


def _summarize(
    fold_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    *,
    ci_level: float,
    seed_fold_frequency_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "task", "class", "feature"]
    seed_keys = keys + ["seed"]

    seed_summary = (
        fold_df.groupby(seed_keys, as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_sd=("coefficient", "std"),
            fold_selection_frequency=("selected", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
    seed_summary["selected"] = (
        seed_summary["fold_selection_frequency"] >= seed_fold_frequency_threshold
    )

    feature_summary = (
        fold_df.groupby(keys, as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_sd=("coefficient", "std"),
            fold_selection_frequency=("selected", "mean"),
            n_fold_models=("coefficient", "size"),
        )
    )
    seed_frequency = (
        seed_summary.groupby(keys, as_index=False)["selected"]
        .mean()
        .rename(columns={"selected": "seed_selection_frequency"})
    )
    feature_summary = feature_summary.merge(seed_frequency, on=keys, how="left")

    if not bootstrap_df.empty:
        alpha = (1.0 - ci_level) / 2.0
        boot_summary = (
            bootstrap_df.groupby(keys)["coefficient"]
            .agg(
                bootstrap_coefficient_mean="mean",
                bootstrap_coefficient_sd="std",
                ci_low=lambda x: x.quantile(alpha),
                ci_high=lambda x: x.quantile(1.0 - alpha),
                n_bootstrap_fits="size",
            )
            .reset_index()
        )
        boot_selection = (
            bootstrap_df.groupby(keys, as_index=False)["selected"]
            .mean()
            .rename(columns={"selected": "bootstrap_selection_frequency"})
        )
        feature_summary = feature_summary.merge(boot_summary, on=keys, how="left")
        feature_summary = feature_summary.merge(boot_selection, on=keys, how="left")

    mean = feature_summary["coefficient_mean"]
    ever_selected = feature_summary["fold_selection_frequency"] > 0
    feature_summary["direction"] = np.select(
        [ever_selected & (mean > 0), ever_selected & (mean < 0)],
        ["positive", "negative"],
        default="not selected",
    )
    if {"ci_low", "ci_high"}.issubset(feature_summary.columns):
        feature_summary["bootstrap_ci_direction"] = np.select(
            [feature_summary["ci_low"] > 0, feature_summary["ci_high"] < 0],
            ["positive", "negative"],
            default="crosses zero",
        )
    return seed_summary, feature_summary


def stability_lasso_cv(
    dataset,
    task_id: str,
    *,
    featurizer=None,
    features: pd.DataFrame | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    n_folds: int | None = None,
    n_bootstrap: int = 200,
    lambda_value: float = 1.0,
    patient_col: str | None = None,
    cv_filter: str | None = None,
    normalize: bool = True,
    class_weight: str | None = "balanced",
    max_iter: int = 5000,
    selected_tolerance: float = 1e-10,
    ci_level: float = 0.95,
    seed_fold_frequency_threshold: float = 0.5,
) -> LassoStabilityResult:
    """Estimate Lasso coefficient stability across folds, seeds and bootstraps.

    Exactly one of ``featurizer`` or ``features`` must be supplied. ``features``
    is a precomputed region-by-feature table indexed by ``region_id`` and is the
    generic entry point for UTAG/CN compositions, handcrafted local features,
    and externally aggregated MIL region embeddings.

    The outer split is patient-level K-fold CV. Each fold model is fit only on
    its training patients. Bootstrap confidence intervals are also computed
    only from the training fold, resampling patients (not individual regions)
    with replacement. Coefficients refer to standardized features.

    ``lambda_value`` is the L1 strength and is passed to scikit-learn as
    ``C = 1 / lambda_value``. Classification tasks are supported; survival
    outcomes require a separate L1 Cox implementation.
    """
    if (featurizer is None) == (features is None):
        raise ValueError("supply exactly one of featurizer or features")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")
    if not 0 <= seed_fold_frequency_threshold <= 1:
        raise ValueError("seed_fold_frequency_threshold must be between 0 and 1")

    task_cfg = dataset.get_task_config(task_id)
    if task_cfg["type"] == "survival":
        raise ValueError("stability_lasso_cv currently supports classification only")

    vcfg = dataset.validation_config
    n_folds = n_folds or vcfg.get("n_folds", 5)
    patient_col = patient_col or vcfg.get("patient_col", "patient_id")
    cv_filter = cv_filter if cv_filter is not None else vcfg.get("cv_filter")

    meta = _metadata_with_region_id(dataset, task_id)
    if cv_filter:
        filtered = meta.query(cv_filter)
        if len(filtered):
            meta = filtered

    precomputed = _validate_features(features) if features is not None else None
    if precomputed is not None:
        meta = meta[meta["region_id"].isin(precomputed.index)]
    if meta.empty:
        raise ValueError(f"no usable regions for task {task_id!r}")
    if patient_col not in meta.columns:
        raise ValueError(f"patient column {patient_col!r} is absent from metadata")

    patient_for_region = meta.set_index("region_id")[patient_col]
    dataset_name = str(getattr(dataset, "name", "dataset"))
    fold_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for seed in seeds:
        folds = safe_patient_kfold(
            meta, n_folds, patient_col, stratify_column(task_cfg), int(seed)
        )
        if folds is None:
            continue
        for fold, (train_ids, val_ids) in enumerate(folds):
            train_ids = [str(x) for x in train_ids]
            if precomputed is None:
                train_regions = dataset.load_regions(train_ids, normalize=normalize)
                fitted_featurizer = _resolve_featurizer(featurizer, train_regions)
                X_train = fitted_featurizer.transform(train_regions)
                X_train = _validate_features(X_train)
            else:
                X_train = precomputed.loc[precomputed.index.intersection(train_ids)]

            y_train = dataset.build_target(list(X_train.index), task_id)
            y_train.index = y_train.index.astype(str)
            X_train = X_train.loc[y_train.index]
            groups = patient_for_region.loc[X_train.index].to_numpy()

            common = {
                "dataset": dataset_name,
                "task": task_id,
                "seed": int(seed),
                "fold": int(fold),
                "lambda": float(lambda_value),
                "n_train": len(X_train),
                "n_val": len(val_ids),
            }
            try:
                rows = _fit_lasso(
                    X_train,
                    y_train,
                    seed=int(seed),
                    lambda_value=lambda_value,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    selected_tolerance=selected_tolerance,
                )
            except Exception as exc:
                warnings.warn(
                    f"Lasso failed for seed={seed}, fold={fold}: {exc}",
                    RuntimeWarning,
                )
                continue
            fold_rows.extend({**common, **row} for row in rows)

            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(fold), 7919])
            )
            for bootstrap in range(n_bootstrap):
                positions = _cluster_bootstrap_positions(groups, rng)
                X_boot = X_train.iloc[positions].copy()
                y_boot = y_train.iloc[positions].copy()
                synthetic_index = pd.Index(
                    [f"bootstrap_{bootstrap}_{i}" for i in range(len(positions))]
                )
                X_boot.index = synthetic_index
                y_boot.index = synthetic_index
                try:
                    rows = _fit_lasso(
                        X_boot,
                        y_boot,
                        seed=int(seed) * 100_000 + fold * 1_000 + bootstrap,
                        lambda_value=lambda_value,
                        class_weight=class_weight,
                        max_iter=max_iter,
                        selected_tolerance=selected_tolerance,
                    )
                except Exception:
                    continue
                bootstrap_common = {
                    **common,
                    "bootstrap": bootstrap,
                    "n_bootstrap_train": len(X_boot),
                }
                bootstrap_rows.extend(
                    {**bootstrap_common, **row} for row in rows
                )

    fold_df = pd.DataFrame(fold_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    if fold_df.empty:
        raise RuntimeError("all Lasso fold fits failed")
    seed_summary, feature_summary = _summarize(
        fold_df,
        bootstrap_df,
        ci_level=ci_level,
        seed_fold_frequency_threshold=seed_fold_frequency_threshold,
    )
    return LassoStabilityResult(
        fold_coefficients=fold_df,
        seed_summary=seed_summary,
        feature_summary=feature_summary,
        bootstrap_coefficients=bootstrap_df,
    )
