"""Scoring: turn model predictions into a metric dict (one function)."""
from __future__ import annotations

import numpy as np

# Primary metric reported per task type (the rest are kept as secondary columns).
PRIMARY_METRIC = {
    "survival": "c_index",
    "binary": "auc_roc",
    "binary_classification": "auc_roc",
    "multiclass": "balanced_acc",
    "multiclass_classification": "balanced_acc",
}


def score_predictions(task_cfg: dict, y_true, y_pred, classes=None) -> dict[str, float]:
    """Compute the metrics for one fold from a task config, targets and predictions.

    Parameters
    ----------
    task_cfg : task dict (uses ``type`` and, for binary, the 0/1 positive encoding).
    y_true   : survival -> DataFrame with ``time`` / ``event``; classification ->
               array/Series of labels (0/1 for binary, raw labels for multiclass).
    y_pred   : classification -> probability matrix ``(n, n_classes)``;
               survival -> 1-D risk score (higher = worse prognosis).
    classes  : model class labels (classification), giving ``y_pred`` column order.

    Degenerate folds (one class present, no events) return NaN rather than raising.
    Metrics are computed by calling sklearn / lifelines directly.
    """
    ttype = task_cfg["type"]

    if ttype == "survival":
        from lifelines.utils import concordance_index
        times = y_true["time"].values
        events = y_true["event"].values
        if events.sum() < 1:
            return {"c_index": float("nan")}
        # Cox risk: higher = worse; lifelines wants higher = longer survival -> negate.
        return {"c_index": float(concordance_index(times, -np.asarray(y_pred), events))}

    y_true = np.asarray(y_true)

    if ttype in ("binary", "binary_classification"):
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, balanced_accuracy_score,
        )
        if classes is None or 1 not in list(classes):
            return {"auc_roc": float("nan"), "avg_precision": float("nan"),
                    "balanced_acc": float("nan")}
        score = np.asarray(y_pred)[:, list(classes).index(1)]
        if len(np.unique(y_true)) < 2:
            auc = ap = float("nan")
        else:
            auc = float(roc_auc_score(y_true, score))
            ap = float(average_precision_score(y_true, score))
        return {"auc_roc": auc, "avg_precision": ap,
                "balanced_acc": float(balanced_accuracy_score(y_true, (score > 0.5).astype(int)))}

    # multiclass
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = np.asarray(classes)[np.argmax(np.asarray(y_pred), axis=1)]
    out = {"balanced_acc": float(balanced_accuracy_score(y_true, pred))}
    try:
        out["macro_auc"] = float(roc_auc_score(
            y_true, y_pred, multi_class="ovr", average="macro", labels=classes))
    except Exception:
        out["macro_auc"] = float("nan")
    return out


def summarize_folds(fold_metrics: list[dict], metric: str) -> tuple[float, float]:
    """Mean and sample SD of ``metric`` across the folds from :func:`cross_validate`
    (NaN folds dropped)."""
    if not fold_metrics:
        return float("nan"), float("nan")
    vals = np.array([fm.get(metric, np.nan) for fm in fold_metrics], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, sd
