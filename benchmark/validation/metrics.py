"""Score region-level predictions and summarize repeated validation results."""

from __future__ import annotations

import numpy as np


PRIMARY_METRIC = {
    "survival": "c_index",
    "binary": "auc_roc",
    "binary_classification": "auc_roc",
    "multiclass": "balanced_acc",
    "multiclass_classification": "balanced_acc",
}


def score_predictions(task_cfg: dict, y_true, y_pred, classes=None) -> dict[str, float]:
    """Calculate task-appropriate metrics for one validation fold.

    Degenerate folds, such as binary folds with one class or survival folds
    without observed events, return ``NaN``.

    Args:
        task_cfg: Task configuration containing the prediction type.
        y_true: Survival ``time``/``event`` table or classification labels.
        y_pred: Survival-risk vector or class-probability matrix.
        classes: Class labels giving the probability-column order.

    Returns:
        Metric mapping for the configured task type.
    """
    task_type = task_cfg["type"]

    if task_type == "survival":
        from lifelines.utils import concordance_index

        times = y_true["time"].values
        events = y_true["event"].values
        if events.sum() < 1:
            return {"c_index": float("nan")}
        # Cox risk increases as prognosis worsens, whereas lifelines expects
        # larger scores to indicate longer survival.
        return {
            "c_index": float(
                concordance_index(times, -np.asarray(y_pred), events)
            )
        }

    y_true = np.asarray(y_true)

    if task_type in ("binary", "binary_classification"):
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            roc_auc_score,
        )

        if classes is None or 1 not in list(classes):
            return {
                "auc_roc": float("nan"),
                "avg_precision": float("nan"),
                "balanced_acc": float("nan"),
            }
        score = np.asarray(y_pred)[:, list(classes).index(1)]
        if len(np.unique(y_true)) < 2:
            auc = average_precision = float("nan")
        else:
            auc = float(roc_auc_score(y_true, score))
            average_precision = float(average_precision_score(y_true, score))
        return {
            "auc_roc": auc,
            "avg_precision": average_precision,
            "balanced_acc": float(
                balanced_accuracy_score(y_true, (score > 0.5).astype(int))
            ),
        }

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    predicted_labels = np.asarray(classes)[np.argmax(np.asarray(y_pred), axis=1)]
    result = {
        "balanced_acc": float(
            balanced_accuracy_score(y_true, predicted_labels)
        )
    }
    try:
        result["macro_auc"] = float(
            roc_auc_score(
                y_true,
                y_pred,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except Exception:
        result["macro_auc"] = float("nan")
    return result


def summarize_folds(
    fold_metrics: list[dict],
    metric: str,
) -> tuple[float, float]:
    """Summarize a metric across validation folds or repeated seeds.

    Args:
        fold_metrics: Per-fold or per-seed metric dictionaries.
        metric: Metric key to summarize.

    Returns:
        Mean and sample standard deviation after dropping ``NaN`` values.
    """
    if not fold_metrics:
        return float("nan"), float("nan")
    values = np.array(
        [fold.get(metric, np.nan) for fold in fold_metrics],
        dtype=float,
    )
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    standard_deviation = (
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    )
    return mean, standard_deviation
