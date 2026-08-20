"""Turn motif scores into region-level binary pseudo labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from benchmark.motifs.spec import MotifCatalog, MotifSpec


def discovery_mask(table: pd.DataFrame, cv_filter: str | None) -> pd.Series:
    """Boolean mask of discovery-cohort rows used to fit motif cut-points.

    ``cv_filter`` is a pandas ``eval`` expression (e.g. ``cohort == 'Basel'``).
    If it is missing, refers to absent columns, or matches nothing, all rows
    are used.
    """
    if not cv_filter:
        return pd.Series(True, index=table.index)
    try:
        mask = table.eval(cv_filter)
    except Exception:
        return pd.Series(True, index=table.index)
    if isinstance(mask, pd.DataFrame):
        mask = mask.all(axis=1)
    mask = pd.Series(mask, index=table.index).fillna(False).astype(bool)
    if not bool(mask.any()):
        return pd.Series(True, index=table.index)
    return mask


def residualize(score: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    """Return OLS residual of ``score`` on ``covariates`` (NaNs left in place)."""
    cov = covariates.reindex(score.index)
    mask = score.notna() & cov.notna().all(axis=1)
    out = score.copy()
    if int(mask.sum()) < max(10, cov.shape[1] + 2):
        return out
    model = LinearRegression()
    x = cov.loc[mask].to_numpy(float)
    y = score.loc[mask].to_numpy(float)
    pred = model.fit(x, y).predict(x)
    out.loc[mask] = y - pred
    return out


def _median_label(values: pd.Series) -> pd.Series:
    mid = float(values.median())
    label = pd.Series(np.nan, index=values.index, dtype="float")
    valid = values.notna()
    label.loc[valid] = (values.loc[valid] >= mid).astype(float)
    return label


def _tertile_extreme_label(values: pd.Series) -> pd.Series:
    """Top tertile = 1, bottom tertile = 0, middle dropped (NaN)."""
    label = pd.Series(np.nan, index=values.index, dtype="float")
    valid = values.dropna()
    if len(valid) < 6:
        return label
    lo, hi = np.nanquantile(valid.to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return _median_label(values)
    label.loc[values <= lo] = 0.0
    label.loc[values >= hi] = 1.0
    return label


def assign_labels(values: pd.Series, rule: str) -> pd.Series:
    if rule == "median":
        return _median_label(values)
    if rule == "tertile_extremes":
        return _tertile_extreme_label(values)
    raise ValueError(f"Unknown label rule {rule!r}")


def working_score(table: pd.DataFrame, spec: MotifSpec, fit_index: pd.Index | None = None) -> pd.Series:
    """Score used for thresholding: raw or residualized on composition fractions."""
    score = table[spec.score_col]
    if not spec.residual_on:
        return score
    cols = [f"frac__{name}" for name in spec.residual_on]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"{spec.id}: residual covariates missing: {missing}")
    cov = table[cols]
    if fit_index is None:
        return residualize(score, cov)
    raw_fit = score.loc[fit_index]
    mask_fit = raw_fit.notna() & cov.loc[fit_index].notna().all(axis=1)
    if int(mask_fit.sum()) < max(10, len(cols) + 2):
        return score
    model = LinearRegression().fit(
        cov.loc[fit_index].loc[mask_fit].to_numpy(float),
        raw_fit.loc[mask_fit].to_numpy(float),
    )
    out = score.copy()
    apply_mask = score.notna() & cov.notna().all(axis=1)
    out.loc[apply_mask] = (
        score.loc[apply_mask].to_numpy(float)
        - model.predict(cov.loc[apply_mask].to_numpy(float))
    )
    return out


def add_pseudo_labels(
    table: pd.DataFrame,
    catalog: MotifCatalog,
    fit_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Add ``<id>_score_used`` and ``<id>_label`` columns.

    Thresholds (median / tertiles) and residual models are estimated on
    ``fit_mask`` rows (the discovery cohort) and then applied to the full table.
    """
    out = table.copy()
    if fit_mask is None:
        fit_index = out.index
    else:
        fit_index = out.index[fit_mask.reindex(out.index).fillna(False)]

    for spec in catalog.motifs:
        used = working_score(out, spec, fit_index=fit_index)
        out[f"{spec.id}_score_used"] = used
        fit_values = used.loc[fit_index]
        if spec.label_rule == "median":
            cut = float(fit_values.median())
            label = pd.Series(np.nan, index=out.index, dtype="float")
            valid = used.notna()
            label.loc[valid] = (used.loc[valid] >= cut).astype(float)
        else:
            valid_fit = fit_values.dropna()
            label = pd.Series(np.nan, index=out.index, dtype="float")
            if len(valid_fit) >= 6:
                lo, hi = np.nanquantile(valid_fit.to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
                if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                    label.loc[used <= lo] = 0.0
                    label.loc[used >= hi] = 1.0
                else:
                    label = assign_labels(used, "median")
            else:
                label = assign_labels(used, "median")
        out[spec.label_col] = label
    return out


def bootstrap_tertile_cuts(
    scores: pd.Series,
    patient_ids: pd.Series,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level bootstrap of tertile cuts on ``scores``.

    Returns ``(low_cuts, high_cuts)`` of length ``n_boot``. Rows with missing
    scores are ignored. If ``patient_ids`` is empty/all-NaN, resampling is at
    the region level.
    """
    table = pd.DataFrame({"score": scores, "patient_id": patient_ids})
    table = table.dropna(subset=["score"])
    if table.empty:
        return np.full(n_boot, np.nan), np.full(n_boot, np.nan)

    rng = np.random.default_rng(seed)
    if table["patient_id"].notna().any():
        groups = [g.to_numpy(float) for _, g in table.groupby("patient_id")["score"]]
    else:
        groups = [np.array([v], dtype=float) for v in table["score"].to_numpy(float)]
    n_groups = len(groups)
    low_cuts = np.empty(n_boot, dtype=float)
    high_cuts = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, n_groups, size=n_groups)
        values = np.concatenate([groups[j] for j in sampled])
        low_cuts[i], high_cuts[i] = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return low_cuts, high_cuts


def bootstrap_confident_labels(
    scores: pd.Series,
    patient_ids: pd.Series,
    n_boot: int = 1000,
    confidence: float = 0.90,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keep a 0/1 label only if it is stable across patient bootstraps.

    ``p_low`` / ``p_high`` are the fraction of bootstraps in which the score
    falls in the bottom / top tertile. A region is labeled 0 or 1 only if that
    fraction is at least ``confidence``; otherwise the label is NaN.
    """
    low_cuts, high_cuts = bootstrap_tertile_cuts(
        scores, patient_ids, n_boot=n_boot, seed=seed
    )
    values = scores.to_numpy(float)
    finite = np.isfinite(values)
    p_low = np.full(len(scores), np.nan)
    p_high = np.full(len(scores), np.nan)
    if finite.any() and np.isfinite(low_cuts).all():
        p_low[finite] = np.mean(values[finite, None] <= low_cuts[None, :], axis=1)
        p_high[finite] = np.mean(values[finite, None] >= high_cuts[None, :], axis=1)

    labels = np.full(len(scores), np.nan)
    low_ok = finite & (p_low >= confidence)
    high_ok = finite & (p_high >= confidence)
    labels[low_ok & ~high_ok] = 0.0
    labels[high_ok & ~low_ok] = 1.0
    return (
        pd.Series(labels, index=scores.index, dtype="float"),
        pd.Series(p_low, index=scores.index, dtype="float"),
        pd.Series(p_high, index=scores.index, dtype="float"),
    )


def add_bootstrap_labels(
    table: pd.DataFrame,
    catalog: MotifCatalog,
    fit_mask: pd.Series | None = None,
    n_boot: int = 1000,
    confidence: float = 0.90,
    seed: int = 0,
    motifs: list[str] | None = None,
) -> pd.DataFrame:
    """Add ``<id>_label_v2``, ``<id>_p_low`` and ``<id>_p_high`` columns.

    Median-rule controls are copied unchanged. Spatial / tertile motifs are
    relabeled with patient-bootstrap-stable tertiles estimated on ``fit_mask``.
    """
    out = table.copy()
    if fit_mask is None:
        fit_mask = pd.Series(True, index=out.index)
    else:
        fit_mask = fit_mask.reindex(out.index).fillna(False)

    wanted = set(motifs) if motifs is not None else None
    patient_col = "patient_id" if "patient_id" in out.columns else None

    for i, spec in enumerate(catalog.motifs):
        if wanted is not None and spec.id not in wanted:
            out[f"{spec.id}_p_low"] = np.nan
            out[f"{spec.id}_p_high"] = np.nan
            if spec.label_col in out.columns:
                out[f"{spec.id}_label_v2"] = out[spec.label_col]
            continue
        used_col = f"{spec.id}_score_used"
        if used_col not in out.columns:
            raise KeyError(f"Missing {used_col}; run generate_pseudo_labels.py first")
        if spec.label_rule != "tertile_extremes":
            out[f"{spec.id}_p_low"] = np.nan
            out[f"{spec.id}_p_high"] = np.nan
            out[f"{spec.id}_label_v2"] = out[spec.label_col]
            continue
        fit_scores = out.loc[fit_mask, used_col]
        fit_patients = (
            out.loc[fit_mask, patient_col] if patient_col else pd.Series(np.nan, index=fit_scores.index)
        )
        # Cuts must be estimated on the discovery cohort, then applied to all rows.
        low_cuts, high_cuts = bootstrap_tertile_cuts(
            fit_scores, fit_patients, n_boot=n_boot, seed=seed + i
        )
        values = out[used_col].to_numpy(float)
        finite = np.isfinite(values)
        p_low = np.full(len(out), np.nan)
        p_high = np.full(len(out), np.nan)
        if finite.any() and np.isfinite(low_cuts).all():
            p_low[finite] = np.mean(values[finite, None] <= low_cuts[None, :], axis=1)
            p_high[finite] = np.mean(values[finite, None] >= high_cuts[None, :], axis=1)
        labels = np.full(len(out), np.nan)
        low_ok = finite & (p_low >= confidence)
        high_ok = finite & (p_high >= confidence)
        labels[low_ok & ~high_ok] = 0.0
        labels[high_ok & ~low_ok] = 1.0
        out[f"{spec.id}_p_low"] = p_low
        out[f"{spec.id}_p_high"] = p_high
        out[f"{spec.id}_label_v2"] = labels
    return out
