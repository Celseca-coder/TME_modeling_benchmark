#!/usr/bin/env python
"""Train Additive MIL on Y_r.

Attention is non-gated. The bag probability is sigmoid(Wz + b) and the loss
is BCE. Dropout applies to the pooled vector during training. The full-window
score is the attention weight. Leave-one-out is the drop in bag probability.

    python scripts/eval_additive_mil_window_importance.py \
        --families composition density mixing point_pattern
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import combo_name  # noqa: E402
from benchmark.models.additive_mil import AdditiveMILModel  # noqa: E402
from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES, selected_motifs_for  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    MIN_LABELED,
    PASS_PRECISION_AT,
    PASS_WINDOW,
    feature_columns,
    join_features_and_labels,
    within_region_auroc,
    within_region_topk,
)
from benchmark.validation.splits import safe_patient_kfold  # noqa: E402

LABEL_ROOT = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset")
WINDOW_DIR = LABEL_ROOT / "mil_windows" / "per_dataset"
BAG_DIR = LABEL_ROOT / "per_dataset"
FEATURE_DIR = Path(
    "/autofs/nas8/tywang/tjzou/TME_modeling_benchmark/results/pseudo_labels/mil_windows/features"
)
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")


def _bags_for_split(table: pd.DataFrame, feat_cols: list[str], region_ids) -> pd.DataFrame:
    rows = []
    wanted = set(map(str, region_ids))
    for rid, g in table[table["region_id"].isin(wanted)].groupby("region_id"):
        g = g.sort_values("window_id")
        rows.append({"region_id": str(rid), "bag": g[feat_cols].to_numpy(np.float32)})
    out = pd.DataFrame(rows)
    return out.set_index("region_id") if len(out) else out


def evaluate_additive_localisation(
    table: pd.DataFrame,
    y_r: pd.Series,
    families,
    seed: int = 0,
    epochs: int = 80,
    patience: int = 15,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-stratified CV. One Additive MIL model per fold."""
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
    rows = []
    loo_rows = []
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = _bags_for_split(table, feat_cols, tr_regs)
        ytr = y_r.reindex(tr.index)
        keep = ytr.isin([0.0, 1.0])
        tr, ytr = tr.loc[keep], ytr.loc[keep]
        if ytr.nunique() < 2:
            continue
        model = AdditiveMILModel(
            seed=seed + k, epochs=epochs, patience=patience, max_instances=None,
            lr=1e-4, weight_decay=0.0,
        ).fit(tr, ytr.astype(int))
        from benchmark.motifs.checkpoints import mil_payload, save_fold
        save_fold(checkpoint_dir, k, mil_payload("additive_mil", model))
        va = table[table["region_id"].isin(set(map(str, va_regs)))].copy()
        held = np.full(len(va), np.nan)
        held_loo = np.full(len(va), np.nan)
        for _, g in va.groupby("region_id"):
            idx = g.sort_values("window_id").index
            yw = g.loc[idx, "y_w"].to_numpy(float)
            labeled = np.isfinite(yw) & np.isin(yw, (0.0, 1.0))
            if labeled.sum() < MIN_LABELED or np.unique(yw[labeled]).size < 2:
                continue
            bag = table.loc[idx, feat_cols].to_numpy(np.float32)
            slot = va.index.get_indexer(idx)
            held[slot] = model.attention_weights(pd.DataFrame({"bag": [bag]}))[0]
            held_loo[slot] = model.leave_one_out_scores(bag)
        score = pd.Series(held, index=va.index)
        loc = within_region_auroc(va["y_w"], score, va["region_id"])
        loc.update(within_region_topk(va["y_w"], score, va["region_id"]))
        loc["fold"] = k
        rows.append(loc)
        loo_score = pd.Series(held_loo, index=va.index)
        loo = within_region_auroc(va["y_w"], loo_score, va["region_id"])
        loo.update(within_region_topk(va["y_w"], loo_score, va["region_id"]))
        loo["fold"] = k
        loo_rows.append(loo)
    return pd.DataFrame(rows), pd.DataFrame(loo_rows)


def _summary_row(dataset: str, motif: str, group: str, fold: pd.DataFrame, method: str = "additive_mil") -> dict:
    if fold.empty:
        auroc = pwr = prev = score = np.nan
        n_regions = n_folds = 0
    else:
        auroc = float(fold["auroc_window_mean"].mean())
        pwr = float(fold["precision_at_20"].mean())
        prev = float(fold["labeled_prev"].mean())
        score = float(fold["score_top20_mean"].mean()) if "score_top20_mean" in fold.columns else np.nan
        n_regions = int(fold["n_regions_scored"].sum()) if "n_regions_scored" in fold.columns else int(fold["n_regions_topk"].sum())
        n_folds = int(len(fold))
    return {
        "dataset": dataset,
        "motif": motif,
        "feature_group": group,
        "method": method,
        "auroc_window_mean": auroc,
        "precision_at_20": pwr,
        "score_top20_mean": score,
        "labeled_prev": prev,
        "n_regions_scored": n_regions,
        "n_folds": n_folds,
        "pass_window": bool(np.isfinite(auroc) and auroc >= PASS_WINDOW),
        "pass_precision_at_20": bool(np.isfinite(pwr) and pwr >= PASS_PRECISION_AT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--motif", default="")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--window-dir", default=str(WINDOW_DIR))
    parser.add_argument("--bag-dir", default=str(BAG_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--out",
        default="results/pseudo_labels/mil_windows/additive_mil_localisation.csv",
    )
    args = parser.parse_args()

    datasets = list(dict.fromkeys(ds for ds, _ in SELECTED_MOTIF_SOURCES))
    if args.datasets:
        datasets = list(args.datasets)
    families = tuple(args.families)
    group = combo_name(families)
    rows = []
    for dataset in datasets:
        feats = pd.read_csv(Path(args.feature_dir) / f"{dataset}_panel.csv")
        wins = pd.read_csv(Path(args.window_dir) / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(Path(args.bag_dir) / f"{dataset}_v2_noisy.csv")
        motifs = [args.motif] if args.motif else list(selected_motifs_for(dataset))
        y_index = bags.set_index(bags["region_id"].astype(str))
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(y_index[f"{motif}_label_v2"], errors="coerce")
            fold, loo = evaluate_additive_localisation(
                tab, y_r, families, epochs=args.epochs, patience=args.patience,
            )
            summary = _summary_row(dataset, motif, group, fold)
            loo_summary = _summary_row(dataset, motif, group, loo, method="additive_loo")
            rows.extend([summary, loo_summary])
            print(summary, flush=True)
            print(loo_summary, flush=True)
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
