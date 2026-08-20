"""Quality checks that a motif label is spatial, balanced, and not just composition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from benchmark.data.dataset import RegionData
from benchmark.motifs.detect import score_motif, shuffle_coordinates
from benchmark.motifs.spec import MotifCatalog, MotifSpec


def label_prevalence(labels: pd.Series) -> dict[str, float]:
    valid = labels.dropna()
    n = int(len(valid))
    n_pos = int((valid == 1).sum()) if n else 0
    return {
        "n_labeled": float(n),
        "n_positive": float(n_pos),
        "positive_rate": (n_pos / n) if n else np.nan,
        "n_dropped": float(labels.isna().sum()),
    }


def composition_correlation(score: pd.Series, frac: pd.Series) -> dict[str, float]:
    mask = score.notna() & frac.notna()
    if int(mask.sum()) < 8:
        return {"spearman_rho": np.nan, "spearman_p": np.nan, "n": float(mask.sum())}
    rho, p = spearmanr(score.loc[mask], frac.loc[mask])
    return {"spearman_rho": float(rho), "spearman_p": float(p), "n": float(mask.sum())}


def spatial_shuffle_delta(
    region: RegionData,
    catalog: MotifCatalog,
    spec: MotifSpec,
    rng: np.random.Generator,
    n_shuffles: int = 1,
) -> dict[str, float]:
    """Compare the real score with coordinate-shuffled scores on one region."""
    real = score_motif(region, catalog, spec)
    shuffled = [
        score_motif(shuffle_coordinates(region, rng), catalog, spec)
        for _ in range(n_shuffles)
    ]
    shuffled_arr = np.asarray(shuffled, float)
    return {
        "real_score": float(real) if np.isfinite(real) else np.nan,
        "shuffled_mean": float(np.nanmean(shuffled_arr)),
        "abs_delta": float(np.abs(real - np.nanmean(shuffled_arr)))
        if np.isfinite(real) and np.isfinite(np.nanmean(shuffled_arr))
        else np.nan,
    }


def spatial_null_report(
    regions: list[RegionData],
    catalog: MotifCatalog,
    spec: MotifSpec,
    seed: int = 0,
) -> dict[str, float]:
    """KS test of real vs shuffled scores across a panel of regions."""
    rng = np.random.default_rng(seed)
    real = []
    fake = []
    for region in regions:
        real.append(score_motif(region, catalog, spec))
        fake.append(score_motif(shuffle_coordinates(region, rng), catalog, spec))
    real_s = pd.Series(real, dtype=float).dropna()
    fake_s = pd.Series(fake, dtype=float).dropna()
    out = {
        "n_regions": float(len(regions)),
        "real_mean": float(real_s.mean()) if len(real_s) else np.nan,
        "shuffled_mean": float(fake_s.mean()) if len(fake_s) else np.nan,
        "mean_abs_delta": float(np.nanmean(np.abs(np.asarray(real, float) - np.asarray(fake, float)))),
        "ks_stat": np.nan,
        "ks_p": np.nan,
    }
    if len(real_s) >= 5 and len(fake_s) >= 5:
        stat, p = ks_2samp(real_s, fake_s)
        out["ks_stat"] = float(stat)
        out["ks_p"] = float(p)
    return out


def qc_table(
    labeled: pd.DataFrame,
    catalog: MotifCatalog,
    *,
    fit_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-motif prevalence and correlation with the closest composition fraction."""
    rows = []
    subset = labeled if fit_mask is None else labeled.loc[fit_mask.reindex(labeled.index).fillna(False)]
    for spec in catalog.motifs:
        prev = label_prevalence(subset[spec.label_col])
        frac_col = None
        if spec.cell_set:
            frac_col = f"frac__{spec.cell_set}"
        elif spec.immune_set:
            frac_col = f"frac__{spec.immune_set}"
        elif spec.source_set:
            frac_col = f"frac__{spec.source_set}"
        elif spec.required_sets:
            frac_col = f"frac__{spec.required_sets[0]}"
        corr = {"spearman_rho": np.nan, "spearman_p": np.nan, "n": np.nan}
        if frac_col and frac_col in subset.columns:
            corr = composition_correlation(subset[spec.score_col], subset[frac_col])
        rows.append({
            "motif_id": spec.id,
            "kind": spec.kind,
            "spatial": spec.spatial,
            "label_rule": spec.label_rule,
            "frac_col": frac_col,
            **prev,
            **corr,
        })
    return pd.DataFrame(rows)
