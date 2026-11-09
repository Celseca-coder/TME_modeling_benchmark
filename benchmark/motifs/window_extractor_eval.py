"""Train bag models on the region label; score windows against y_w."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmark.features.local_window import columns_for_families
from benchmark.models.l1_mil import L1MILModel
from benchmark.validation.splits import safe_patient_kfold

PASS_WINDOW = 0.60
MIN_LABELED = 6
TOPK_FRAC = 0.20
TOP_FRAC = 0.20
# Chance precision@20% among labeled windows is the regional prevalence (~0.5).
# 0.70 is clearly above chance and reachable when ranking is useful.
PASS_PRECISION_AT = 0.70


def _row_id(region_id, window_id) -> str:
    return f"{region_id}::{int(window_id)}"


def join_features_and_labels(
    features: pd.DataFrame,
    windows: pd.DataFrame,
    motif: str,
    bags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inner-join x_w to window GT. Metrics use y_w; bag training uses Y_r."""
    feat = features.copy()
    feat["region_id"] = feat["region_id"].astype(str)
    feat["window_id"] = feat["window_id"].astype(int)
    lab = windows[windows["motif_id"].astype(str) == motif].copy()
    lab["region_id"] = lab["region_id"].astype(str)
    lab["window_id"] = lab["window_id"].astype(int)
    lab["y_w"] = pd.to_numeric(lab["label"], errors="coerce")
    keep = ["region_id", "window_id", "y_w"]
    if "score_used" in lab.columns:
        keep.append("score_used")
    out = feat.merge(lab[keep], on=["region_id", "window_id"], how="inner")
    if bags is not None and "patient_id" in bags.columns:
        meta = bags[["region_id", "patient_id"]].copy()
        meta["region_id"] = meta["region_id"].astype(str)
        out = out.merge(meta.drop_duplicates("region_id"), on="region_id", how="left")
    out["row_id"] = [_row_id(r, w) for r, w in zip(out["region_id"], out["window_id"])]
    return out


def feature_columns(table: pd.DataFrame, families: tuple[str, ...] | None = None) -> list[str]:
    skip = {
        "region_id", "window_id", "cx_um", "cy_um", "y_w", "score_used",
        "patient_id", "row_id", "motif_id", "n_cells",
    }
    cols = [c for c in table.columns if c not in skip]
    if families:
        cols = columns_for_families(cols, families)
        if not cols:
            raise ValueError(f"no feature columns for families {families}")
    return cols


def within_region_auroc(y: pd.Series, score: pd.Series, region: pd.Series) -> dict:
    """Mean AUROC(p_hat_w, y_w) inside each region. Do not pool windows."""
    aucs = []
    for _, idx in region.groupby(region).groups.items():
        yy = y.loc[idx].to_numpy(float)
        ss = score.loc[idx].to_numpy(float)
        mask = np.isfinite(yy) & np.isfinite(ss)
        yy, ss = yy[mask], ss[mask]
        if len(yy) < MIN_LABELED or yy.min() == yy.max():
            continue
        aucs.append(float(roc_auc_score(yy, ss)))
    arr = np.asarray(aucs, float)
    return {
        "n_regions_scored": int(arr.size),
        "auroc_window_mean": float(arr.mean()) if arr.size else np.nan,
        "auroc_window_median": float(np.median(arr)) if arr.size else np.nan,
        "pass_window": bool(arr.size and float(arr.mean()) >= PASS_WINDOW),
    }


def topk_labeled_positions(
    y: np.ndarray,
    score: np.ndarray,
    frac: float = TOPK_FRAC,
) -> np.ndarray:
    """Positions of the top ``frac`` finite 0/1 labels, ranked by ``score``.

    Positions index the original arrays. Empty when ``within_region_topk``
    would skip the slice (too few labels, or a single class).
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score) & np.isin(y, (0.0, 1.0))
    labeled = np.flatnonzero(mask)
    yy, ss = y[labeled], score[labeled]
    if len(yy) < MIN_LABELED or yy.min() == yy.max():
        return np.array([], dtype=int)
    k = max(1, int(np.ceil(frac * len(yy))))
    local = np.argpartition(-ss, kth=min(k - 1, len(ss) - 1))[:k]
    return labeled[local]


def within_region_topk(
    y: pd.Series,
    score: pd.Series,
    region: pd.Series,
    frac: float = TOPK_FRAC,
) -> dict:
    """Precision / recall of y_w=1 among the top ``frac`` labeled windows.

    Rank only windows with a finite 0/1 label, inside each region. Chance
    precision is the labeled-window prevalence (≈0.5 under median / tertile
    extremes). Chance recall equals ``frac``.
    """
    precs, recs, prevs, ks, score_means = [], [], [], [], []
    for _, idx in region.groupby(region).groups.items():
        yy = y.loc[idx].to_numpy(float)
        ss = score.loc[idx].to_numpy(float)
        top = topk_labeled_positions(yy, ss, frac)
        if top.size == 0:
            continue
        labeled = np.isfinite(yy) & np.isfinite(ss) & np.isin(yy, (0.0, 1.0))
        yy_l = yy[labeled]
        k = int(top.size)
        hit = float((yy[top] == 1.0).sum())
        n_pos = float((yy_l == 1.0).sum())
        precs.append(hit / k)
        recs.append(hit / n_pos if n_pos else np.nan)
        prevs.append(n_pos / len(yy_l))
        ks.append(k)
        score_means.append(float(np.mean(ss[top])))
    prec = np.asarray(precs, float)
    rec = np.asarray(recs, float)
    prev = np.asarray(prevs, float)
    return {
        "n_regions_topk": int(prec.size),
        "precision_at_20": float(prec.mean()) if prec.size else np.nan,
        "recall_at_20": float(np.nanmean(rec)) if rec.size else np.nan,
        "labeled_prev": float(prev.mean()) if prev.size else np.nan,
        "topk_k_mean": float(np.mean(ks)) if ks else np.nan,
        "score_top20_mean": float(np.mean(score_means)) if score_means else np.nan,
    }


def within_region_precision_at_frac(
    y: pd.Series,
    score: pd.Series,
    region: pd.Series,
    frac: float = TOPK_FRAC,
) -> dict:
    """Alias used by tests: same ranking, keys ``n_regions_scored`` / ``precision_at_20_mean``."""
    rec = within_region_topk(y, score, region, frac=frac)
    return {
        "n_regions_scored": rec["n_regions_topk"],
        "precision_at_20_mean": rec["precision_at_20"],
        "recall_at_20_mean": rec["recall_at_20"],
        "labeled_prev": rec["labeled_prev"],
    }


def region_label(bags: pd.DataFrame, motif: str) -> pd.Series:
    """Region label ``{motif}_label_v2``, indexed by ``region_id``."""
    col = f"{motif}_label_v2"
    meta = bags.drop_duplicates("region_id").copy()
    meta["region_id"] = meta["region_id"].astype(str)
    if col not in meta.columns:
        raise KeyError(f"{col} is not in the region table")
    return pd.to_numeric(meta.set_index("region_id")[col], errors="coerce")


def bags_for_regions(table: pd.DataFrame, feat_cols: list[str], region_ids) -> pd.DataFrame:
    """One row per region. ``bag`` is the window matrix in ``window_id`` order."""
    rows = []
    wanted = set(map(str, region_ids))
    part = table[table["region_id"].astype(str).isin(wanted)]
    for rid, group in part.groupby("region_id", sort=False):
        group = group.sort_values("window_id")
        rows.append({
            "region_id": str(rid),
            "bag": group[feat_cols].to_numpy(np.float32),
            "index": group.index.to_numpy(),
        })
    out = pd.DataFrame(rows)
    return out.set_index("region_id") if len(out) else out

# 分训练折和验证折
def patient_region_folds(table: pd.DataFrame, y_r: pd.Series, n_folds: int = 5, seed: int = 0):
    """Patient folds stratified by the region label."""
    if "patient_id" not in table.columns:
        raise ValueError("need patient_id for clustered CV")
    meta = (
        table[["region_id", "patient_id"]]
        .dropna(subset=["patient_id"])
        .drop_duplicates("region_id")
        .copy()
    )
    meta["region_id"] = meta["region_id"].astype(str)
    labels = y_r.reindex(meta["region_id"])
    meta = meta.loc[labels.isin([0.0, 1.0]).to_numpy()].copy()
    meta["Y_r"] = y_r.reindex(meta["region_id"]).to_numpy(float)
    folds = safe_patient_kfold(meta, n_folds, "patient_id", stratify_col="Y_r", seed=seed)
    if not folds:
        raise ValueError("not enough patients for CV")
    return folds


def evaluate_fold(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feat_cols: list[str],
    y_r: pd.Series,
    seed: int = 0,
    l1_lambda: float = 1e-5,
    lr: float = 1e-4,
    epochs: int = 80,
    patience: int = 15,
    checkpoint_dir: str | Path | None = None,
    fold: int = 0,
) -> dict:
    """Fit mean-pooled L1 MIL on Y_r. Score validation windows with p_w."""
    tr = bags_for_regions(train, feat_cols, train["region_id"].unique())
    ytr = y_r.reindex(tr.index)
    keep = ytr.isin([0.0, 1.0])
    tr, ytr = tr.loc[keep], ytr.loc[keep]
    va = val.copy()
    empty = {"n_train": int(len(tr)), "auroc_window_mean": np.nan, "pass_window": False}
    if tr.empty or ytr.nunique() < 2:
        return empty, dict(empty)
    model = L1MILModel(
        seed=seed, lr=lr, l1_lambda=l1_lambda, epochs=epochs, patience=patience, max_instances=None,
    ).fit(tr[["bag"]], ytr.astype(int))
    from benchmark.motifs.checkpoints import mil_payload, save_fold
    save_fold(checkpoint_dir, fold, mil_payload("l1_mil", model))
    full = pd.Series(np.nan, index=va.index, dtype=float)
    loo = pd.Series(np.nan, index=va.index, dtype=float)
    for rid, group in va.groupby("region_id", sort=False):
        group = group.sort_values("window_id")
        bag = group[feat_cols].to_numpy(np.float32)
        full.loc[group.index] = model.window_probabilities(bag)
        loo.loc[group.index] = model.leave_one_out_scores(bag)
    return _score_record(va, full, tr), _score_record(va, loo, tr)


def evaluate_table(
    table: pd.DataFrame,
    y_r: pd.Series,
    n_folds: int = 5,
    seed: int = 0,
    families: tuple[str, ...] | None = None,
    l1_lambda: float = 1e-5,
    lr: float = 1e-4,
    epochs: int = 80,
    patience: int = 15,
    checkpoint_dir: str | Path | None = None,
    with_loo: bool = False,
):
    """Patient-level CV. Train on the region label. Score windows with p_w."""
    feat_cols = feature_columns(table, families=families)
    folds = patient_region_folds(table, y_r, n_folds=n_folds, seed=seed)
    rows, loo_rows = [], []
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = table[table["region_id"].astype(str).isin(set(map(str, tr_regs)))]
        va = table[table["region_id"].astype(str).isin(set(map(str, va_regs)))]
        rec, loo = evaluate_fold(
            tr, va, feat_cols, y_r,
            seed=seed + k, l1_lambda=l1_lambda, lr=lr, epochs=epochs, patience=patience,
            checkpoint_dir=checkpoint_dir, fold=k,
        )
        rec["fold"] = k
        loo["fold"] = k
        rows.append(rec)
        loo_rows.append(loo)
    full = pd.DataFrame(rows)
    if with_loo:
        return full, pd.DataFrame(loo_rows)
    return full


def _score_record(va: pd.DataFrame, score: pd.Series, train: pd.DataFrame) -> dict:
    loc = within_region_auroc(va["y_w"], score, va["region_id"])
    loc.update(within_region_topk(va["y_w"], score, va["region_id"]))
    loc["n_train"] = int(len(train))
    loc["n_val"] = int(len(va))
    loc["n_val_labeled"] = int(va["y_w"].isin([0.0, 1.0]).sum())
    return loc