#!/usr/bin/env python
"""Train AttnMIL on Y_r; score windows with Shapley phi_w; AUROC vs y_w.

    python scripts/eval_shap_mil_window_importance.py \
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
from benchmark.motifs.shap_mil import shap_window_importance
from benchmark.motifs.source_select import selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    PASS_WINDOW,
    feature_columns,
    join_features_and_labels,
    within_region_auroc,
)
from benchmark.validation.splits import safe_patient_kfold

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")


def _bags_for_split(table: pd.DataFrame, feat_cols: list[str], region_ids) -> pd.DataFrame:
    rows = []
    for rid, g in table[table["region_id"].isin(set(map(str, region_ids)))].groupby("region_id"):
        g = g.sort_values("window_id")
        rows.append({"region_id": str(rid), "bag": g[feat_cols].to_numpy(np.float32)})
    out = pd.DataFrame(rows)
    return out.set_index("region_id") if len(out) else out


def evaluate_shap_mil(table: pd.DataFrame, y_r: pd.Series, families, n_samples=32, top_m=None, seed=0):
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
    recs = []
    scored = []
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
        phi = np.full(len(va), np.nan)
        from benchmark.motifs.window_extractor_eval import MIN_LABELED  # 文件头已有 within_region_auroc，一并导入即可
        for rid, g in va.groupby("region_id"):
            idx = g.sort_values("window_id").index
            yw = g.loc[idx, "y_w"].to_numpy(float)
            labeled = np.isfinite(yw)
            if labeled.sum() < MIN_LABELED or np.unique(yw[labeled]).size < 2:
                continue
            bag = table.loc[idx, feat_cols].to_numpy(np.float32)
            vals = shap_window_importance(
                model,
                bag,
                n_samples=n_samples,
                seed=seed+k,
                top_m=top_m,
            )
            phi[va.index.get_indexer(idx)] = vals
        va = va.copy()
        va["phi_w"] = phi
        loc = within_region_auroc(va["y_w"], va["phi_w"], va["region_id"])
        loc["fold"] = k
        recs.append(loc)
        scored.append(va)
    return pd.DataFrame(recs), pd.concat(scored, ignore_index=True) if scored else va


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--motif", default="")
    p.add_argument("--features", required=True)
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--top-m", type=int, default=None)
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/shap_mil_localisation.csv")
    args = p.parse_args()

    feats = pd.read_csv(args.features)
    wins = pd.read_csv(WINDOW_DIR / f"{args.dataset}_windows_noisy.csv")
    bags = pd.read_csv(BAG_DIR / f"{args.dataset}_v2_noisy.csv")
    motifs = [args.motif] if args.motif else list(selected_motifs_for(args.dataset))
    combos = (tuple(args.families),) if args.families else WINDOW_PANEL_COMBOS

    rows = []
    for families in combos:
        group = combo_name(families)
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(bags.set_index(bags["region_id"].astype(str))[f"{motif}_label_v2"], errors="coerce")
            fold, _ = evaluate_shap_mil(tab, y_r, families, n_samples=args.n_samples, top_m=args.top_m)
            summary = {
                "dataset": args.dataset,
                "motif": motif,
                "feature_group": group,
                "method": "shap_mil",
                "auroc_window_mean": float(fold["auroc_window_mean"].mean()),
                "n_regions_scored": int(fold["n_regions_scored"].sum()),
                "n_folds": int(len(fold)),
                "pass_window": bool(fold["auroc_window_mean"].mean() >= PASS_WINDOW),
            }
            rows.append(summary)
            print(summary, flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()