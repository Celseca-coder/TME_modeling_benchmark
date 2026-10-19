#!/usr/bin/env python
"""Train gated-attention MIL on Y_r; score windows; AUROC and PWR@20 vs y_w.

Three scores share each validation fold: softmax attention, Shapley phi_w,
and Integrated Gradients s_w. Metrics stay inside each region.

    python scripts/eval_ig_mil_window_importance.py \
        --dataset tnbc_wang2023 --motif cd8_high \
        --features results/pseudo_labels/mil_windows/features/tnbc_wang2023_panel.csv \
        --families composition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import WINDOW_PANEL_COMBOS, combo_name
from benchmark.models.attention_mil import AttentionMILModel
from benchmark.motifs.ig_mil import ig_completeness_error, ig_window_importance
from benchmark.motifs.shap_mil import shap_window_importance
from benchmark.motifs.source_select import selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    MIN_LABELED,
    PASS_PRECISION_AT,
    PASS_WINDOW,
    feature_columns,
    join_features_and_labels,
    within_region_auroc,
    within_region_topk,
)
from benchmark.validation.splits import safe_patient_kfold

LABEL_ROOT = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset")
WINDOW_DIR = LABEL_ROOT / "mil_windows" / "per_dataset"
BAG_DIR = LABEL_ROOT / "per_dataset"

IG_STEPS = 32
IG_STEPS_RETRY = 64
COMPLETENESS_RETRY = 0.05
METHODS = ("attention", "shap_mil", "ig")


def _bags_for_split(table: pd.DataFrame, feat_cols: list[str], region_ids) -> pd.DataFrame:
    rows = []
    for rid, g in table[table["region_id"].isin(set(map(str, region_ids)))].groupby("region_id"):
        g = g.sort_values("window_id")
        rows.append({"region_id": str(rid), "bag": g[feat_cols].to_numpy(np.float32)})
    out = pd.DataFrame(rows)
    return out.set_index("region_id") if len(out) else out


def _ig_scores(model: AttentionMILModel, bag: np.ndarray, steps: int) -> tuple[np.ndarray, float, int]:
    """Right-Riemann IG. Recompute at 64 steps when the completeness gap stays large."""
    scores = ig_window_importance(model, bag, steps=steps)
    err = ig_completeness_error(model, bag, scores)
    used = steps
    if steps < IG_STEPS_RETRY and err > COMPLETENESS_RETRY:
        scores = ig_window_importance(model, bag, steps=IG_STEPS_RETRY)
        err = ig_completeness_error(model, bag, scores)
        used = IG_STEPS_RETRY
    return scores, err, used


def _region_metrics(y: pd.Series, score: np.ndarray, region: pd.Series) -> dict:
    score_s = pd.Series(score, index=y.index)
    loc = within_region_auroc(y, score_s, region)
    loc.update(within_region_topk(y, score_s, region))
    return loc


def evaluate_gated_localisation(
    table: pd.DataFrame,
    y_r: pd.Series,
    families,
    steps: int = IG_STEPS,
    n_samples: int = 32,
    seed: int = 0,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, pd.DataFrame]:
    """Patient-stratified CV. One gated-attention model per fold, three window scores."""
    feat_cols = feature_columns(table, families=families)
    meta = (
        table[["region_id", "patient_id"]]
        .dropna(subset=["patient_id"])
        .drop_duplicates("region_id")
    )
    ycol = y_r.reindex(meta["region_id"].astype(str))
    meta = meta.loc[ycol.isin([0.0, 1.0]).to_numpy()].copy()
    meta["Y_r"] = y_r.reindex(meta["region_id"].astype(str)).to_numpy(float)
    folds = safe_patient_kfold(meta, 5, "patient_id", stratify_col="Y_r", seed=seed)
    fold_rows = {method: [] for method in methods}
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = _bags_for_split(table, feat_cols, tr_regs)
        ytr = y_r.reindex(tr.index)
        keep = ytr.isin([0.0, 1.0])
        tr, ytr = tr.loc[keep], ytr.loc[keep]
        if ytr.nunique() < 2:
            continue
        model = AttentionMILModel(
            seed=seed + k, epochs=80, patience=15, max_instances=None,
        ).fit(tr, ytr.astype(int))
        va = table[table["region_id"].isin(set(map(str, va_regs)))].copy()
        held = {method: np.full(len(va), np.nan) for method in methods}
        errors: list[float] = []
        steps_used: list[int] = []
        for _, g in va.groupby("region_id"):
            idx = g.sort_values("window_id").index
            yw = g.loc[idx, "y_w"].to_numpy(float)
            labeled = np.isfinite(yw) & np.isin(yw, (0.0, 1.0))
            if labeled.sum() < MIN_LABELED or np.unique(yw[labeled]).size < 2:
                continue
            bag = table.loc[idx, feat_cols].to_numpy(np.float32)
            slot = va.index.get_indexer(idx)
            if "attention" in methods:
                attn = model.attention_weights(pd.DataFrame({"bag": [bag]}))[0]
                held["attention"][slot] = attn
            if "shap_mil" in methods:
                held["shap_mil"][slot] = shap_window_importance(
                    model, bag, n_samples=n_samples, seed=seed + k,
                )
            if "ig" in methods:
                scores, err, used = _ig_scores(model, bag, steps)
                held["ig"][slot] = scores
                errors.append(err)
                steps_used.append(used)
        for method in methods:
            loc = _region_metrics(va["y_w"], held[method], va["region_id"])
            loc["fold"] = k
            if method == "ig":
                loc["completeness_error_mean"] = float(np.mean(errors)) if errors else np.nan
                loc["ig_steps_mean"] = float(np.mean(steps_used)) if steps_used else np.nan
            fold_rows[method].append(loc)
    return {method: pd.DataFrame(rows) for method, rows in fold_rows.items()}


def _summary_row(dataset: str, motif: str, group: str, method: str, fold: pd.DataFrame) -> dict:
    if fold.empty:
        auroc = pwr = prev = np.nan
        n_regions = n_folds = 0
    else:
        auroc = float(fold["auroc_window_mean"].mean())
        pwr = float(fold["precision_at_20"].mean())
        prev = float(fold["labeled_prev"].mean())
        n_regions = int(fold["n_regions_scored"].sum())
        n_folds = int(len(fold))
    row = {
        "dataset": dataset,
        "motif": motif,
        "feature_group": group,
        "method": method,
        "auroc_window_mean": auroc,
        "precision_at_20": pwr,
        "labeled_prev": prev,
        "n_regions_scored": n_regions,
        "n_folds": n_folds,
        "pass_window": bool(np.isfinite(auroc) and auroc >= PASS_WINDOW),
        "pass_precision_at_20": bool(np.isfinite(pwr) and pwr >= PASS_PRECISION_AT),
        "completeness_error_mean": np.nan,
        "ig_steps_mean": np.nan,
    }
    if method == "ig" and not fold.empty and "completeness_error_mean" in fold.columns:
        row["completeness_error_mean"] = float(fold["completeness_error_mean"].mean())
        row["ig_steps_mean"] = float(fold["ig_steps_mean"].mean())
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--motif", default="")
    p.add_argument("--features", default="")
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--steps", type=int, default=IG_STEPS)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--window-dir", default=str(WINDOW_DIR))
    p.add_argument("--bag-dir", default=str(BAG_DIR))
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/gated_attention_localisation.csv")
    args = p.parse_args()

    feat_path = Path(args.features) if args.features else Path(
        f"results/pseudo_labels/mil_windows/features/{args.dataset}_panel.csv"
    )
    feats = pd.read_csv(feat_path)
    wins = pd.read_csv(Path(args.window_dir) / f"{args.dataset}_windows_noisy.csv")
    bags = pd.read_csv(Path(args.bag_dir) / f"{args.dataset}_v2_noisy.csv")
    motifs = [args.motif] if args.motif else list(selected_motifs_for(args.dataset))
    combos = (tuple(args.families),) if args.families else WINDOW_PANEL_COMBOS
    methods = tuple(args.methods)

    rows = []
    for families in combos:
        group = combo_name(families)
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(
                bags.set_index(bags["region_id"].astype(str))[f"{motif}_label_v2"],
                errors="coerce",
            )
            folds = evaluate_gated_localisation(
                tab, y_r, families,
                steps=args.steps, n_samples=args.n_samples, methods=methods,
            )
            for method in methods:
                summary = _summary_row(args.dataset, motif, group, method, folds[method])
                rows.append(summary)
                print(summary, flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
