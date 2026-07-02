"""Generic validation runners for the region-level pipeline:
patient-level cross-validation and train-cohort -> test-cohort generalization."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from benchmark.validation.metrics import score_predictions
from benchmark.validation.splits import safe_patient_kfold, stratify_column

# A ``featurizer`` argument may be either:
#   * a factory (callable) -> a fresh extractor created and fit on the training
#     regions of each fold (the no-leakage default), or
#   * a pre-fitted / stateless extractor instance -> used as-is (transform only),
#     e.g. one fit on a fixed reference vocabulary or shared across datasets.
Featurizer = Any


def _resolve_featurizer(featurizer: Featurizer, train_regions: list):
    """Return a ready-to-``transform`` featurizer: call+fit a factory, or use a
    pre-specified instance unchanged."""
    return featurizer().fit(train_regions) if callable(featurizer) else featurizer


def _nan_metrics(task_type: str) -> dict[str, float]:
    if task_type == "survival":
        return {"c_index": float("nan")}
    if task_type in ("binary", "binary_classification"):
        return {"auc_roc": float("nan"), "avg_precision": float("nan"),
                "balanced_acc": float("nan")}
    return {"balanced_acc": float("nan"), "macro_auc": float("nan")}


def cross_validate(
    dataset,
    task_id: str,
    featurizer: Featurizer,
    model_factory: Callable[[dict, int], object],
    *,
    seeds: Sequence[int] = (0,),
    n_folds: int | None = None,
    patient_col: str | None = None,
    cv_filter: str | None = None,
    normalize: bool = True,
) -> list[dict]:
    """Run patient-level K-fold cross-validation; return one metric dict per fold.

    Generic over the featurizer and model — it only needs:

    * ``featurizer`` -> an extractor with ``fit(regions).transform(regions) ->
      DataFrame indexed by region_id`` (e.g. ``benchmark.features.basic_feats``).
      Pass either a **factory** (callable) — re-created and fit on each *training*
      fold so the feature vocabulary never leaks from val — or a **pre-fitted /
      stateless instance**, which is used as-is on every fold.
    * ``model_factory(task_cfg, seed)`` -> a model with ``fit(features, target)`` and
      ``predict(features)`` (e.g. ``benchmark.models``). Re-created per fold.

    Splitting is at the **patient** level (no patient crosses train/val); every region
    is an independent sample and metrics are computed over regions. The whole run is
    repeated once per entry of ``seeds`` (each a fresh CV partition). Folds where the
    model cannot be fit (e.g. a single class in the training split) yield NaN metrics
    so the run continues.

    ``n_folds`` / ``patient_col`` / ``cv_filter`` default to the dataset's
    ``validation`` config. Each returned dict holds the metric values plus
    ``seed``, ``fold``, ``n_train``, ``n_val``.
    """
    vcfg = dataset.validation_config
    n_folds = n_folds or vcfg.get("n_folds", 5)
    patient_col = patient_col or vcfg.get("patient_col", "patient_id")
    cv_filter = cv_filter if cv_filter is not None else vcfg.get("cv_filter")
    task_cfg = dataset.get_task_config(task_id)

    meta = dataset.get_task_metadata(task_id)
    if len(meta) == 0:
        return []
    if cv_filter:
        sub = meta.query(cv_filter)
        meta = sub if len(sub) else meta            # ignore a filter that empties the task

    fold_metrics: list[dict] = []
    for seed in seeds:
        folds = safe_patient_kfold(meta, n_folds, patient_col, stratify_column(task_cfg), seed)
        if folds is None:
            continue
        for fold_i, (train_ids, val_ids) in enumerate(folds):
            train_regions = dataset.load_regions(train_ids, normalize=normalize)
            val_regions = dataset.load_regions(val_ids, normalize=normalize)

            feat = _resolve_featurizer(featurizer, train_regions)
            X_tr, X_va = feat.transform(train_regions), feat.transform(val_regions)
            y_tr = dataset.build_target(list(X_tr.index), task_id)
            y_va = dataset.build_target(list(X_va.index), task_id)

            try:
                model = model_factory(task_cfg, seed).fit(X_tr, y_tr)
                metrics = score_predictions(task_cfg, y_va, model.predict(X_va),
                                            getattr(model, "classes_", None))
            except Exception:
                metrics = _nan_metrics(task_cfg["type"])

            metrics.update({"seed": seed, "fold": fold_i,
                            "n_train": len(X_tr), "n_val": len(X_va)})
            fold_metrics.append(metrics)
    return fold_metrics


def cohort_split_test(
    dataset,
    task_id: str,
    gentest: dict,
    featurizer: Featurizer,
    model_factory: Callable[[dict, int], object],
    *,
    seeds: Sequence[int] = (0,),
    normalize: bool = True,
) -> list[dict]:
    """Train on the train cohort(s), evaluate on the held-out test cohort(s).

    ``gentest`` is one entry of the dataset's ``validation.generalization_tests``
    (a dict with ``train`` / ``test`` cohort lists, e.g. ``{"train": ["UPMC_HNC"],
    "test": ["DFCI_HNC"], ...}``). The data split is fixed; ``seeds`` only re-seed the
    model (so a deterministic model gives identical rows).

    Same ``featurizer`` / ``model_factory`` contract as :func:`cross_validate`. A
    featurizer **factory** is fit on the train cohort (so the feature space is the
    train cohort's — bake any cross-cohort column, e.g.
    ``cell_type_col=gentest["cell_type_col"]``, into it); a **pre-fitted instance** is
    used as-is.

    Returns one metric dict per seed (with ``seed`` / ``n_train`` / ``n_test``), or
    ``[]`` if either cohort has no labelled regions for the task.
    """
    cohort_col = dataset.validation_config["cohort_col"]
    task_cfg = dataset.get_task_config(task_id)
    meta = dataset.get_task_metadata(task_id)

    train_ids = meta[meta[cohort_col].isin(gentest["train"])]["region_id"].tolist()
    test_ids = meta[meta[cohort_col].isin(gentest["test"])]["region_id"].tolist()
    if not train_ids or not test_ids:
        return []

    train_regions = dataset.load_regions(train_ids, normalize=normalize)
    feat = _resolve_featurizer(featurizer, train_regions)
    X_tr = feat.transform(train_regions)
    X_te = feat.transform(dataset.load_regions(test_ids, normalize=normalize))
    y_tr = dataset.build_target(list(X_tr.index), task_id)
    y_te = dataset.build_target(list(X_te.index), task_id)

    results: list[dict] = []
    for seed in seeds:
        try:
            model = model_factory(task_cfg, seed).fit(X_tr, y_tr)
            metrics = score_predictions(task_cfg, y_te, model.predict(X_te),
                                        getattr(model, "classes_", None))
        except Exception:
            metrics = _nan_metrics(task_cfg["type"])
        metrics.update({"seed": seed, "n_train": len(X_tr), "n_test": len(X_te)})
        results.append(metrics)
    return results
