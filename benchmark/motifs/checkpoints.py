"""Save one trained fold so a later run can reload it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


SCORE_TAG = "full+loo"


def score_tag_matches(directory: str | Path | None) -> bool:
    """True when a saved fold was trained for the current full-window and LOO scores."""
    if directory is None:
        return False
    files = sorted(Path(directory).glob("fold*.pt"))
    if not files:
        return False
    try:
        payload = torch.load(files[0], map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(files[0], map_location="cpu")
    return isinstance(payload, dict) and payload.get("score") == SCORE_TAG


def save_fold(directory: str | Path | None, fold: int, payload: dict) -> Path | None:
    """Write ``fold{k}.pt``. ``directory is None`` leaves the model unsaved."""
    if directory is None:
        return None
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"fold{int(fold)}.pt"
    torch.save(payload, path)
    return path


def mil_payload(kind: str, model) -> dict:
    """Weights plus the train-set scaler for an attention-family model."""
    return {
        "kind": kind,
        "state_dict": {key: value.detach().cpu() for key, value in model._net.state_dict().items()},
        "median": np.asarray(model._median),
        "mean": np.asarray(model._mean),
        "std": np.asarray(model._std),
        "input_dim": int(model._input_dim),
        "classes": np.asarray(model.classes_),
        "hidden_dim": int(model.hidden_dim),
        "attention_dim": int(model.attention_dim),
        "dropout": float(model.dropout),
        "seed": int(model.seed),
        "lr": float(model.lr),
        "weight_decay": float(getattr(model, "weight_decay", 0.0)),
        "l1_lambda": float(getattr(model, "l1_lambda", 0.0)),
        "loss": "bce_l1" if kind == "l1_mil" else "bce",
        "score": "full+loo",
    }


def mlp_payload(model) -> dict:
    """Weights plus the columns kept after the variance filter."""
    return {
        "kind": "mlp",
        "state_dict": {key: value.detach().cpu() for key, value in model._net.state_dict().items()},
        "columns": list(model._columns),
        "keep": np.asarray(model._keep),
        "median": np.asarray(model._median),
        "mean": np.asarray(model._mean),
        "std": np.asarray(model._std),
        "classes": np.asarray(model.classes_),
        "hidden_dim": int(model.hidden_dim),
        "dropout": float(model.dropout),
        "seed": int(model.seed),
        "lr": float(model.lr),
        "l1_lambda": float(model.l1_lambda),
        "loss": "bce",
        "score": "full+loo",
    }


def rf_payload(model) -> dict:
    """The fitted forest and the tabular preprocessor. Stored with torch.save."""
    return {
        "kind": "rf",
        "classifier": model._clf,
        "median": model._median,
        "keep": list(model._keep),
        "scaler_mean": np.asarray(model._scaler.mean_),
        "scaler_scale": np.asarray(model._scaler.scale_),
        "classes": np.asarray(model.classes_),
        "seed": int(model.seed),
        "n_estimators": int(model.n_estimators),
        "max_features": model.max_features,
        "class_weight": model.class_weight,
    }
