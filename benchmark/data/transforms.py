"""Expression normalization.

The benchmark applies a single, **stateless arcsinh variance-stabilising transform**
to raw expression, with a per-dataset cofactor matched to the imaging platform's
dynamic range::

    x = arcsinh(x / cofactor)

This is a data-level transform you can inspect directly — a "quick view" of normalised
expression — and it needs no fitting, so there is no train/val leakage to worry about.

Feature **standardisation (z-scoring) is intentionally not done here.** It is a
model-side concern applied to the patient/region feature matrix and fit on the training
split only — see the ``StandardScaler`` step in ``benchmark/models/`` (LogisticRegression,
RandomForest, CoxPH, CoxNet all do it). Tree models are invariant to it; linear / Cox
models need it. Keeping it in the model avoids leakage and keeps the data path simple.

Cofactor modes:

* **fixed** — a single scalar applied to every marker.
* **adaptive** (``cofactor: quantile``) — a per-marker ``quantile_scale * p_q + eps``
  (default ``5 * p20 + 1e-5``, the Wu2022 pipeline); self-scales each marker to its own
  low-signal floor. Only appropriate when the bottom quantile is strictly positive.

See ``docs/normalization_rationale.md`` for the per-dataset cofactor decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame, exclude_cols: list[str] | None) -> list[str]:
    """Execute the numeric cols operation.
    
        Args:
            df (pd.DataFrame): Input data frame.
            exclude_cols (list[str] | None): Columns excluded from numeric feature selection.
    
        Returns:
            list[str]: The operation result.
    
    Args:
        df (pd.DataFrame): Data frame whose columns are inspected or transformed."""
    skip = set(exclude_cols or ["cell_id"])
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _resolve_cofactor(
    sub: pd.DataFrame,
    cofactor: float | str,
    quantile: float,
    quantile_scale: float,
    eps: float,
) -> np.ndarray | float:
    """Return a scalar (fixed) or per-marker (adaptive) cofactor array.
    
    Args:
        sub (pd.DataFrame): Subset of values currently being transformed.
        cofactor (float | str): Cofactor used by the arcsinh transformation.
        quantile (float): Reference quantile used to estimate the scaling factor.
        quantile_scale (float): Value controlling or representing quantile scale.
        eps (float): Small positive constant used to avoid numerical instability."""
    if isinstance(cofactor, str):
        mode = cofactor.lower()
        if mode in ("quantile", "adaptive", "auto"):
            return quantile_scale * np.quantile(sub.to_numpy(), quantile, axis=0) + eps
        raise ValueError(
            f"Unknown cofactor spec {cofactor!r}; expected a number or 'quantile'."
        )
    return float(cofactor)


def apply_normalization(
    region_data: "RegionData",  # noqa: F821  — avoid circular import
    norm_cfg: dict | None = None,
) -> "RegionData":
    """Return a copy of *region_data* with arcsinh-transformed expression.
    
        Stateless and per-region: nothing is fit across regions. Feature standardisation
        is handled model-side (see module docstring).
    
        ``norm_cfg`` keys
        -----------------
        method : {arcsinh, none}
        cofactor : float or "quantile"            (for arcsinh)
        quantile, quantile_scale, eps             (for the adaptive "quantile" cofactor)
    
    Args:
        region_data ('RegionData'): Region object containing coordinates, labels, and expression data.
        norm_cfg (dict | None): Normalization settings applied to region measurements."""
    from .dataset import RegionData

    norm_cfg = norm_cfg or {}
    method = norm_cfg.get("method", "arcsinh")

    if method in ("none", "raw"):
        return region_data
    if method != "arcsinh":
        raise ValueError(
            f"Unknown normalization method: {method!r}. Expected 'arcsinh' or 'none' "
            f"(feature z-scoring is done model-side, not here)."
        )

    cofactor = norm_cfg.get("cofactor", 1.0)
    quantile = float(norm_cfg.get("quantile", 0.2))
    quantile_scale = float(norm_cfg.get("quantile_scale", 5.0))
    eps = float(norm_cfg.get("eps", 1e-5))

    df = region_data.expression
    cols = _numeric_cols(df, None)
    result = df.copy()
    sub = df[cols].astype(float)
    cf = _resolve_cofactor(sub, cofactor, quantile, quantile_scale, eps)
    result[cols] = np.arcsinh(sub / cf)

    return RegionData(
        region_id=region_data.region_id,
        coordinates=region_data.coordinates,
        expression=result,
        cell_types=region_data.cell_types,
        microns_per_pixel=region_data.microns_per_pixel,
        polygons=region_data.polygons,
    )
