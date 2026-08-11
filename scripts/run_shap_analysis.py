#!/usr/bin/env python
"""Patient-level cross-validated SHAP analysis for region feature tables.

Models are trained on each patient-level training fold and SHAP values are
computed only for held-out regions. This keeps feature construction, model
fitting and explanation separated from the validation patients.

Supported models:

* ``logistic``: exact linear SHAP values in log-odds space;
* ``cox``: exact linear SHAP values in log-risk space;
* ``random-forest``: TreeSHAP probability-space values;
* ``xgboost``: XGBoost's exact TreeSHAP contributions in margin space.

All tabular feature sources accepted by ``run_stability_lasso.py`` are
supported, including precomputed MIL bag-level vectors and CytoCommunity TCN
compositions. Raw variable-length MIL bags and neural-network graphs must first
be pooled into a fixed region-by-feature table.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import list_datasets, load_dataset
from benchmark.validation.splits import safe_patient_kfold, stratify_column
from scripts.run_stability_lasso import (
    _featurizer_factory,
    _load_cytocommunity_features,
    _load_precomputed,
    _normalize_regions,
    _restrict_to_available_regions,
)

LOGGER = logging.getLogger("shap_analysis")


def _configure_logging(path: str | None) -> Path:
    if path:
        output = Path(path).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = _CODE / "log" / f"shap_{stamp}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (logging.FileHandler(output, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return output


def _metadata(dataset, task: str, patient_col: str, cv_filter: str | None) -> pd.DataFrame:
    meta = dataset.get_task_metadata(task).copy()
    if "region_id" not in meta:
        meta = meta.reset_index(names="region_id")
    meta["region_id"] = meta["region_id"].astype(str)
    if cv_filter:
        filtered = meta.query(cv_filter)
        if len(filtered):
            meta = filtered
    if patient_col not in meta:
        raise ValueError(f"Patient column {patient_col!r} is absent from metadata")
    return meta


def _fold_features(
    dataset,
    args,
    train_ids: list[str],
    val_ids: list[str],
    precomputed: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if precomputed is not None:
        train = precomputed.loc[precomputed.index.intersection(train_ids)]
        val = precomputed.loc[precomputed.index.intersection(val_ids)]
        return train, val

    normalize = _normalize_regions(args)
    train_regions = dataset.load_regions(train_ids, normalize=normalize)
    val_regions = dataset.load_regions(val_ids, normalize=normalize)
    factory = _featurizer_factory(args, dataset)
    if factory is None:
        raise ValueError(f"Unsupported SHAP feature source: {args.feature_source}")
    featurizer = factory().fit(train_regions)
    train = featurizer.transform(train_regions)
    val = featurizer.transform(val_regions)
    val = val.reindex(columns=train.columns)
    return train, val


def _preprocess(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    from sklearn.preprocessing import StandardScaler

    median = train.median()
    train_filled = train.fillna(median)
    std = train_filled.std()
    keep = std[std > 0].index.tolist()
    if not keep:
        raise ValueError("No non-constant features in the training fold")
    scaler = StandardScaler().fit(train_filled[keep])
    train_scaled = scaler.transform(train_filled[keep])
    val_raw = val[keep].fillna(median[keep])
    val_scaled = scaler.transform(val_raw)
    return train_scaled, val_scaled, val_raw, keep


def _sample_validation(
    val: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum is None or len(val) <= maximum:
        return val
    return val.sample(n=maximum, random_state=seed).sort_index()


def _linear_classification_shap(
    train: np.ndarray,
    val: np.ndarray,
    target: pd.Series,
    args,
) -> tuple[list[str], list[np.ndarray]]:
    from sklearn.linear_model import LogisticRegression

    penalty = args.logistic_penalty
    kwargs: dict[str, Any]
    if penalty == "l1":
        kwargs = {"penalty": "l1", "solver": "saga"}
    elif penalty == "elasticnet":
        kwargs = {
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": args.l1_ratio,
        }
    else:
        kwargs = {"penalty": "l2", "solver": "lbfgs"}
    model = LogisticRegression(
        C=args.C,
        class_weight=None if args.no_class_weight else "balanced",
        max_iter=args.max_iter,
        random_state=args.model_seed,
        **kwargs,
    ).fit(train, target.to_numpy())
    centered = val - train.mean(axis=0)
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.shape[0] == 1:
        return [str(model.classes_[1])], [centered * coefficients[0]]
    return (
        [str(value) for value in model.classes_],
        [centered * coefficients[index] for index in range(len(model.classes_))],
    )


def _cox_shap(
    train: np.ndarray,
    val: np.ndarray,
    target: pd.DataFrame,
    feature_names: list[str],
    args,
) -> tuple[list[str], list[np.ndarray]]:
    from lifelines import CoxPHFitter

    frame = pd.DataFrame(train, columns=feature_names, index=target.index)
    frame["time"] = target["time"].to_numpy()
    frame["event"] = target["event"].to_numpy()
    model = CoxPHFitter(penalizer=args.cox_penalizer)
    model.fit(frame, duration_col="time", event_col="event")
    coefficient = model.params_.reindex(feature_names).to_numpy(dtype=float)
    return ["risk"], [(val - train.mean(axis=0)) * coefficient]


def _tree_shap(
    train: np.ndarray,
    val: np.ndarray,
    target: pd.Series,
    model_kind: str,
    args,
) -> tuple[list[str], list[np.ndarray]]:
    import shap

    if model_kind == "random-forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_features=args.max_features,
            class_weight=None if args.no_class_weight else "balanced",
            random_state=args.model_seed,
            n_jobs=-1,
        ).fit(train, target.to_numpy())
        classes = model.classes_
    else:
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        from xgboost import XGBClassifier

        encoder = LabelEncoder().fit(target.to_numpy())
        encoded = encoder.transform(target.to_numpy())
        n_classes = len(encoder.classes_)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective=objective,
            eval_metric="logloss",
            random_state=args.model_seed,
            n_jobs=args.n_jobs,
        ).fit(train, encoded)
        classes = encoder.classes_
        contributions = model.get_booster().predict(
            xgb.DMatrix(val), pred_contribs=True
        )
        # The final column is the expected-value/bias term, not a feature.
        if contributions.ndim == 2:
            return [str(classes[1])], [contributions[:, :-1]]
        if contributions.ndim == 3:
            return (
                [str(value) for value in classes],
                [
                    contributions[:, index, :-1]
                    for index in range(contributions.shape[1])
                ],
            )
        raise ValueError(
            f"Unexpected XGBoost contribution shape: {contributions.shape}"
        )

    rng = np.random.default_rng(args.model_seed)
    if len(train) > args.background_size:
        background = train[rng.choice(len(train), args.background_size, replace=False)]
    else:
        background = train
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    raw = explainer.shap_values(val)
    arrays = _normalise_tree_output(raw, val.shape[1], len(classes))
    if len(classes) == 2:
        index = 1 if len(arrays) > 1 else 0
        return [str(classes[1])], [arrays[index]]
    return [str(value) for value in classes], arrays


def _normalise_tree_output(
    values: Any,
    n_features: int,
    n_classes: int,
) -> list[np.ndarray]:
    if isinstance(values, list):
        return [np.asarray(value, dtype=float) for value in values]
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        return [array]
    if array.ndim != 3:
        raise ValueError(f"Unexpected TreeSHAP shape: {array.shape}")
    if array.shape[1] == n_features:
        return [array[:, :, index] for index in range(array.shape[2])]
    if array.shape[2] == n_features:
        return [array[:, index, :] for index in range(array.shape[1])]
    raise ValueError(
        f"Cannot identify feature axis in TreeSHAP shape {array.shape}; "
        f"expected {n_features} features and {n_classes} classes"
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    result = spearmanr(x, y, nan_policy="omit")
    return float(result.statistic)


def _explanation_rows(
    dataset_name: str,
    task: str,
    scheme: str,
    model: str,
    seed: int,
    fold: int,
    class_names: list[str],
    shap_arrays: list[np.ndarray],
    raw_values: pd.DataFrame,
    patient_for_region: pd.Series,
    top_k: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    detail: list[pd.DataFrame] = []
    fold_summary: list[dict[str, Any]] = []
    for class_name, shap_values in zip(class_names, shap_arrays):
        importance = np.mean(np.abs(shap_values), axis=0)
        top_positions = set(np.argsort(-importance)[: min(top_k, len(importance))])
        for position, feature in enumerate(raw_values.columns):
            values = raw_values.iloc[:, position].to_numpy(dtype=float)
            effects = shap_values[:, position]
            correlation = _spearman(values, effects)
            fold_summary.append({
                "dataset": dataset_name,
                "task": task,
                "scheme": scheme,
                "model": model,
                "class": class_name,
                "seed": seed,
                "fold": fold,
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(effects))),
                "mean_shap": float(np.mean(effects)),
                "value_shap_spearman": correlation,
                "top_feature": position in top_positions,
                "n_explained_regions": len(raw_values),
            })

        shap_frame = pd.DataFrame(
            shap_values, index=raw_values.index, columns=raw_values.columns
        )
        value_long = raw_values.rename_axis("region_id").reset_index().melt(
            id_vars="region_id", var_name="feature", value_name="feature_value"
        )
        shap_long = shap_frame.rename_axis("region_id").reset_index().melt(
            id_vars="region_id", var_name="feature", value_name="shap_value"
        )
        frame = value_long.merge(shap_long, on=["region_id", "feature"])
        frame["patient_id"] = frame["region_id"].map(patient_for_region)
        frame.insert(0, "fold", fold)
        frame.insert(0, "seed", seed)
        frame.insert(0, "class", class_name)
        frame.insert(0, "model", model)
        frame.insert(0, "scheme", scheme)
        frame.insert(0, "task", task)
        frame.insert(0, "dataset", dataset_name)
        detail.append(frame)
    return detail, fold_summary


def _summarise_features(fold_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "scheme", "model", "class", "feature"]
    grouped = fold_summary.groupby(keys, dropna=False)
    summary = grouped.agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        sd_abs_shap=("mean_abs_shap", "std"),
        mean_shap=("mean_shap", "mean"),
        value_shap_spearman=("value_shap_spearman", "mean"),
        top_fold_frequency=("top_feature", "mean"),
        n_fold_models=("fold", "size"),
        n_explained_regions=("n_explained_regions", "sum"),
    ).reset_index()

    def consistency(values: pd.Series) -> float:
        signs = np.sign(values.dropna())
        signs = signs[signs != 0]
        if not len(signs):
            return float("nan")
        return float(max((signs > 0).mean(), (signs < 0).mean()))

    direction = grouped["value_shap_spearman"].apply(consistency).rename(
        "direction_consistency"
    ).reset_index()
    summary = summary.merge(direction, on=keys, how="left")
    summary["direction"] = np.select(
        [
            summary["value_shap_spearman"] > 0,
            summary["value_shap_spearman"] < 0,
        ],
        ["higher value increases prediction", "higher value decreases prediction"],
        default="unclear",
    )
    return summary.sort_values(
        ["dataset", "task", "scheme", "model", "class", "mean_abs_shap"],
        ascending=[True, True, True, True, True, False],
    )


def _summarise_patients(detail: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "task", "scheme", "model", "class", "patient_id", "feature"
    ]
    result = detail.assign(abs_shap=detail["shap_value"].abs()).groupby(
        keys, dropna=False
    ).agg(
        mean_abs_shap=("abs_shap", "mean"),
        mean_shap=("shap_value", "mean"),
        max_abs_shap=("abs_shap", "max"),
        n_region_explanations=("region_id", "size"),
    ).reset_index()
    return result.sort_values(
        ["dataset", "task", "scheme", "model", "class", "mean_abs_shap"],
        ascending=[True, True, True, True, True, False],
    )


def _cross_dataset_summary(feature_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "scheme", "model", "class", "feature"]
    grouped = feature_summary.groupby(keys, dropna=False)
    result = grouped.agg(
        n_datasets=("dataset", "nunique"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        median_abs_shap=("mean_abs_shap", "median"),
        mean_value_shap_spearman=("value_shap_spearman", "mean"),
    ).reset_index()

    def agreement(values: pd.Series) -> float:
        signs = np.sign(values.dropna())
        signs = signs[signs != 0]
        if not len(signs):
            return float("nan")
        return float(max((signs > 0).mean(), (signs < 0).mean()))

    agreement_table = grouped["value_shap_spearman"].apply(agreement).rename(
        "dataset_direction_agreement"
    ).reset_index()
    return result.merge(agreement_table, on=keys, how="left").sort_values(
        ["task", "scheme", "model", "class", "mean_abs_shap"],
        ascending=[True, True, True, True, False],
    )


def run_dataset_task(
    dataset_name: str,
    dataset,
    task: str,
    args,
    precomputed: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_cfg = dataset.get_task_config(task)
    is_survival = task_cfg["type"] == "survival"
    model_kind = (
        "cox" if is_survival else "logistic"
    ) if args.model == "auto" else args.model
    if is_survival != (model_kind == "cox"):
        LOGGER.info(
            "Skipping %s/%s: model=%s is %s survival task",
            dataset_name,
            task,
            model_kind,
            "not compatible with" if is_survival else "only compatible with",
        )
        return pd.DataFrame(), pd.DataFrame()

    patient_col = args.patient_col or dataset.validation_config.get(
        "patient_col", "patient_id"
    )
    cv_filter = (
        args.cv_filter
        if args.cv_filter is not None
        else dataset.validation_config.get("cv_filter")
    )
    meta = _metadata(dataset, task, patient_col, cv_filter)
    if precomputed is not None:
        meta = meta[meta["region_id"].isin(precomputed.index)]
    if meta.empty:
        raise ValueError(f"No usable regions for {dataset_name}/{task}")
    patient_for_region = meta.set_index("region_id")[patient_col]
    n_folds = args.n_folds or dataset.validation_config.get("n_folds", 5)
    details: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        splits = safe_patient_kfold(
            meta, n_folds, patient_col, stratify_column(task_cfg), seed
        )
        if splits is None:
            continue
        for fold, (train_ids, val_ids) in enumerate(splits):
            fold_args = argparse.Namespace(
                **{**vars(args), "model_seed": int(seed)}
            )
            LOGGER.info(
                "%s/%s/cv seed=%d fold=%d: train=%d val=%d",
                dataset_name, task, seed, fold, len(train_ids), len(val_ids),
            )
            train_features, val_features = _fold_features(
                dataset, args, list(map(str, train_ids)), list(map(str, val_ids)),
                precomputed,
            )
            val_features = _sample_validation(
                val_features, args.max_explain_regions_per_fold, seed * 1000 + fold
            )
            train_target = dataset.build_target(list(train_features.index), task)
            train_features = train_features.loc[train_target.index]
            train_scaled, val_scaled, val_raw, feature_names = _preprocess(
                train_features, val_features
            )
            if model_kind == "logistic":
                class_names, shap_arrays = _linear_classification_shap(
                    train_scaled, val_scaled, train_target, fold_args
                )
            elif model_kind == "cox":
                class_names, shap_arrays = _cox_shap(
                    train_scaled, val_scaled, train_target, feature_names, fold_args
                )
            else:
                class_names, shap_arrays = _tree_shap(
                    train_scaled, val_scaled, train_target, model_kind, fold_args
                )
            detail, summaries = _explanation_rows(
                dataset_name, task, "cv", model_kind, seed, fold, class_names,
                shap_arrays, val_raw, patient_for_region, args.top_k,
            )
            details.extend(detail)
            fold_rows.extend(summaries)
            dataset.clear_region_cache()
    return (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.DataFrame(fold_rows),
    )


def run_dataset_generalization(
    dataset_name: str,
    dataset,
    task: str,
    scheme: str,
    args,
    precomputed: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = {
        test["name"]: test
        for test in dataset.validation_config.get("generalization_tests", [])
    }
    if scheme not in tests:
        raise KeyError(
            f"Generalization scheme {scheme!r} not found for {dataset_name}"
        )
    test = tests[scheme]
    task_cfg = dataset.get_task_config(task)
    is_survival = task_cfg["type"] == "survival"
    model_kind = (
        "cox" if is_survival else "logistic"
    ) if args.model == "auto" else args.model
    if is_survival != (model_kind == "cox"):
        return pd.DataFrame(), pd.DataFrame()

    patient_col = args.patient_col or dataset.validation_config.get(
        "patient_col", "patient_id"
    )
    cohort_col = dataset.validation_config["cohort_col"]
    meta = _metadata(dataset, task, patient_col, None)
    if precomputed is not None:
        meta = meta[meta["region_id"].isin(precomputed.index)]
    train_ids = meta.loc[
        meta[cohort_col].isin(test["train"]), "region_id"
    ].astype(str).tolist()
    test_ids = meta.loc[
        meta[cohort_col].isin(test["test"]), "region_id"
    ].astype(str).tolist()
    if not train_ids or not test_ids:
        raise ValueError(
            f"No train/test regions for {dataset_name}/{task}/{scheme}"
        )
    patient_for_region = meta.set_index("region_id")[patient_col]
    details: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    scheme_args = argparse.Namespace(
        **{
            **vars(args),
            "cell_type_col": test.get("cell_type_col", args.cell_type_col),
        }
    )
    for seed in args.seeds:
        fold_args = argparse.Namespace(
            **{**vars(scheme_args), "model_seed": int(seed)}
        )
        LOGGER.info(
            "%s/%s/%s seed=%d: train=%d test=%d",
            dataset_name, task, scheme, seed, len(train_ids), len(test_ids),
        )
        train_features, test_features = _fold_features(
            dataset, fold_args, train_ids, test_ids, precomputed
        )
        test_features = _sample_validation(
            test_features, args.max_explain_regions_per_fold, seed
        )
        train_target = dataset.build_target(list(train_features.index), task)
        train_features = train_features.loc[train_target.index]
        train_scaled, test_scaled, test_raw, feature_names = _preprocess(
            train_features, test_features
        )
        if model_kind == "logistic":
            class_names, shap_arrays = _linear_classification_shap(
                train_scaled, test_scaled, train_target, fold_args
            )
        elif model_kind == "cox":
            class_names, shap_arrays = _cox_shap(
                train_scaled, test_scaled, train_target, feature_names, fold_args
            )
        else:
            class_names, shap_arrays = _tree_shap(
                train_scaled, test_scaled, train_target, model_kind, fold_args
            )
        detail, summaries = _explanation_rows(
            dataset_name, task, scheme, model_kind, seed, 0, class_names,
            shap_arrays, test_raw, patient_for_region, args.top_k,
        )
        details.extend(detail)
        fold_rows.extend(summaries)
        dataset.clear_region_cache()
    return (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.DataFrame(fold_rows),
    )


def _safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "group"


def _write_plots(
    detail: pd.DataFrame,
    feature_summary: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> None:
    if detail.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    group_keys = ["dataset", "task", "scheme", "model", "class"]
    for key, summary_group in feature_summary.groupby(group_keys, dropna=False):
        mask = np.ones(len(detail), dtype=bool)
        for column, value in zip(group_keys, key):
            mask &= detail[column].astype(str).to_numpy() == str(value)
        group = detail.loc[mask]
        if group.empty:
            continue
        top = summary_group.nlargest(top_k, "mean_abs_shap")["feature"].tolist()
        group = group[group["feature"].isin(top)]
        observation = ["seed", "fold", "region_id"]
        shap_wide = group.pivot_table(
            index=observation, columns="feature", values="shap_value", aggfunc="mean"
        ).reindex(columns=top)
        value_wide = group.pivot_table(
            index=observation, columns="feature", values="feature_value", aggfunc="mean"
        ).reindex(index=shap_wide.index, columns=top)
        if shap_wide.empty:
            continue
        label = "__".join(_safe_label(value) for value in key)
        shap.summary_plot(
            shap_wide.to_numpy(),
            value_wide,
            feature_names=top,
            max_display=min(top_k, len(top)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{label}__summary.png", dpi=180)
        plt.close()
        for feature in top[:3]:
            shap.dependence_plot(
                feature,
                shap_wide.to_numpy(),
                value_wide,
                feature_names=top,
                interaction_index=None,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{label}__dependence__{_safe_label(feature)}.png",
                dpi=180,
            )
            plt.close()


def run(args) -> dict[str, pd.DataFrame]:
    global_features = None
    if args.feature_source == "precomputed":
        if not args.features_csv:
            raise ValueError("--features-csv is required for precomputed SHAP")
        global_features = _load_precomputed(args.features_csv, args.region_id_col)

    all_detail: list[pd.DataFrame] = []
    all_fold: list[pd.DataFrame] = []
    selected_by_dataset: dict[str, list[tuple[str, str]]] = {}
    if args.selected_runs:
        from benchmark.validation.selected_tasks import SELECTED_RUNS

        for dataset_name, task, scheme, _metric in SELECTED_RUNS:
            selected_by_dataset.setdefault(dataset_name, []).append((task, scheme))
        dataset_names = list(selected_by_dataset)
    else:
        dataset_names = args.datasets or list_datasets()

    for dataset_name in dataset_names:
        try:
            dataset = load_dataset(dataset_name, data_root=args.data_root)
            dataset_features = global_features
            if args.feature_source == "cytocommunity":
                dataset_features = _load_cytocommunity_features(
                    dataset_name, dataset, args.cytocommunity_root
                )
            elif dataset_features is None:
                _restrict_to_available_regions(dataset)
        except Exception:
            if not args.continue_on_error:
                raise
            LOGGER.exception(
                "Dataset preparation failed for %s; continuing", dataset_name
            )
            continue
        requested = (
            selected_by_dataset[dataset_name]
            if args.selected_runs
            else [(task, "cv") for task in (args.tasks or dataset.task_ids)]
        )
        for task, scheme in requested:
            if task not in dataset.task_ids:
                LOGGER.warning("Skipping absent task %s for %s", task, dataset_name)
                continue
            try:
                if scheme == "cv":
                    detail, fold = run_dataset_task(
                        dataset_name, dataset, task, args, dataset_features
                    )
                else:
                    detail, fold = run_dataset_generalization(
                        dataset_name, dataset, task, scheme, args, dataset_features
                    )
            except Exception:
                if not args.continue_on_error:
                    raise
                LOGGER.exception(
                    "SHAP failed for %s/%s/%s; continuing",
                    dataset_name, task, scheme,
                )
                continue
            if len(detail):
                all_detail.append(detail)
            if len(fold):
                all_fold.append(fold)
        dataset.clear_region_cache()

    detail = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    fold_summary = (
        pd.concat(all_fold, ignore_index=True) if all_fold else pd.DataFrame()
    )
    feature_summary = (
        _summarise_features(fold_summary) if len(fold_summary) else pd.DataFrame()
    )
    patient_summary = (
        _summarise_patients(detail) if len(detail) else pd.DataFrame()
    )
    cross_dataset = (
        _cross_dataset_summary(feature_summary)
        if len(feature_summary)
        else pd.DataFrame()
    )
    return {
        "observation_values": detail,
        "fold_summary": fold_summary,
        "feature_summary": feature_summary,
        "patient_summary": patient_summary,
        "cross_dataset_summary": cross_dataset,
    }


def _write_results(tables: dict[str, pd.DataFrame], prefix: str) -> None:
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        output = path.parent / f"{path.name}_{name}.csv"
        table.to_csv(output, index=False)
        LOGGER.info("Wrote %s (%d rows)", output, len(table))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--cv-filter", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--selected-runs",
        action="store_true",
        help=(
            "Run only benchmark.validation.selected_tasks.SELECTED_RUNS, "
            "including fixed cohort-transfer schemes."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["auto", "logistic", "cox", "random-forest", "xgboost"],
        default="logistic",
    )
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--logistic-penalty", choices=["l1", "l2", "elasticnet"], default="l1"
    )
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--cox-penalizer", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--max-explain-regions-per-fold", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument(
        "--feature-source",
        choices=[
            "precomputed", "composition", "expression",
            "composition-expression", "patch", "density",
            "spatial-distance", "point-pattern", "mixing", "utag",
            "kronos", "eva", "cytocommunity",
        ],
        default="composition",
    )
    parser.add_argument("--features-csv", default=None)
    parser.add_argument("--region-id-col", default="region_id")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-tissue-mask", action="store_true")
    parser.add_argument("--window-size", type=float, default=100)
    parser.add_argument("--step", type=float, default=50)
    parser.add_argument(
        "--feature-groups", nargs="+",
        choices=["composition", "expression"], default=["composition"],
    )
    parser.add_argument(
        "--aggregations", nargs="+",
        choices=["mean", "max", "min", "std", "quantile"],
        default=["mean", "max", "std", "quantile"],
    )
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--distance-k", type=int, default=1)
    parser.add_argument(
        "--point-pattern-radii", nargs="+", type=float,
        default=[10, 20, 50, 100, 200],
    )
    parser.add_argument(
        "--point-pattern-metrics", nargs="+",
        choices=["K", "L", "pcf", "variogram"], default=["K", "L"],
    )
    parser.add_argument("--point-pattern-by-type", action="store_true")
    parser.add_argument("--mixing-k", type=int, default=10)

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
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--image-mode", choices=["native", "rasterized", "auto"], default="auto"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-stride", type=int, default=224)
    parser.add_argument("--min-foreground", type=float, default=0.01)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--eva-cls", action="store_true")
    parser.add_argument("--cache-dir", default=None)
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
            _CODE / "model_results" / "CytoCommunity" / "native_local_runs_cutoff02"
        ),
    )
    parser.add_argument(
        "--output-prefix", default=str(_CODE / "results" / "shap_analysis")
    )
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = _configure_logging(args.log_file)
    LOGGER.info(
        "Log=%s model=%s source=%s datasets=%s tasks=%s",
        log_path, args.model, args.feature_source,
        args.datasets or "all", args.tasks or "all compatible",
    )
    tables = run(args)
    _write_results(tables, args.output_prefix)
    if not args.no_plots:
        plots_dir = (
            Path(args.plots_dir)
            if args.plots_dir
            else Path(args.output_prefix).parent
            / f"{Path(args.output_prefix).name}_plots"
        )
        _write_plots(
            tables["observation_values"],
            tables["feature_summary"],
            plots_dir,
            args.top_k,
        )
        LOGGER.info("Wrote SHAP plots to %s", plots_dir)


if __name__ == "__main__":
    main()
