#!/usr/bin/env python
"""Train a mean-pooled window MLP on Y_r.

The full-window score is the MLP logit. Leave-one-out is the drop in bag
probability. Global importance x_f is still the mean absolute IG inside each
validation fold when feature ranking is requested.

    python scripts/eval_mlp_ig_window_importance.py \
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

from benchmark.features.local_window import combo_name
from benchmark.motifs.mlp_ig import (
    WindowMLP,
    global_feature_importance,
    integrated_gradients,
    mean_feature_rank,
    recovery_at_k,
)
from benchmark.motifs.source_select import CANDIDATE_DATASETS, selected_motifs_for
from benchmark.motifs.window_extractor_eval import (
    PASS_PRECISION_AT,
    PASS_WINDOW,
    bags_for_regions,
    feature_columns,
    join_features_and_labels,
    patient_region_folds,
    region_label,
    within_region_auroc,
    within_region_topk,
)

WINDOW_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
K = 5


def evaluate_mlp_ig(
    table: pd.DataFrame,
    y_r: pd.Series,
    families,
    steps: int = 50,
    seed: int = 0,
    epochs: int = 80,
    patience: int = 15,
    checkpoint_dir: str | Path | None = None,
    rank_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-level CV on Y_r.

    Returns full-window metrics, leave-one-out metrics, and the mean feature
    ranking. The ranking is empty unless ``rank_features`` is set.
    """
    feat_cols = feature_columns(table, families=families)
    folds = patient_region_folds(table, y_r, seed=seed)
    rows = []
    loo_rows = []
    orders: list[list[str]] = []
    for k, (tr_regs, va_regs) in enumerate(folds):
        tr = bags_for_regions(table, feat_cols, tr_regs)
        ytr = y_r.reindex(tr.index)
        keep = ytr.isin([0.0, 1.0])
        tr, ytr = tr.loc[keep], ytr.loc[keep]
        if ytr.nunique() < 2:
            continue
        model = WindowMLP(seed=seed + k, epochs=epochs, patience=patience).fit(
            tr[["bag"]], ytr.astype(float), columns=feat_cols,
        )
        from benchmark.motifs.checkpoints import mlp_payload, save_fold
        save_fold(checkpoint_dir, k, mlp_payload(model))
        va = table[table["region_id"].astype(str).isin(set(map(str, va_regs)))].copy()
        scores = pd.Series(np.nan, index=va.index, dtype=float)
        loo_scores = pd.Series(np.nan, index=va.index, dtype=float)
        ig_parts: list[np.ndarray] = []
        for _, group in va.groupby("region_id", sort=False):
            group = group.sort_values("window_id")
            bag = group[feat_cols].to_numpy(np.float32)
            scores.loc[group.index] = model.window_logits(bag)
            loo_scores.loc[group.index] = model.leave_one_out_scores(bag)
            if rank_features:
                ig, _window_scores = integrated_gradients(model, bag, steps=steps)
                if len(ig):
                    ig_parts.append(ig)
        loc = within_region_auroc(va["y_w"], scores, va["region_id"])
        loc.update(within_region_topk(va["y_w"], scores, va["region_id"]))
        loc["fold"] = k
        rows.append(loc)
        loo = within_region_auroc(va["y_w"], loo_scores, va["region_id"])
        loo.update(within_region_topk(va["y_w"], loo_scores, va["region_id"]))
        loo["fold"] = k
        loo_rows.append(loo)
        if ig_parts:
            importance = global_feature_importance(np.concatenate(ig_parts, axis=0))
            order = np.argsort(-importance)
            orders.append([model._columns[int(i)] for i in order])
    return pd.DataFrame(rows), pd.DataFrame(loo_rows), mean_feature_rank(orders)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(CANDIDATE_DATASETS))
    p.add_argument("--motif", default="")
    p.add_argument("--feature-dir", default="results/pseudo_labels/mil_windows/features")
    p.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--window-dir", default=str(WINDOW_DIR))
    p.add_argument("--bag-dir", default=str(BAG_DIR))
    p.add_argument("--out", default="results/pseudo_labels/mil_windows/mlp_ig_localisation.csv")
    p.add_argument("--features-out", default="results/pseudo_labels/mil_windows/mlp_ig_features.csv")
    args = p.parse_args()

    families = tuple(args.families)
    group = combo_name(families)
    rows = []
    feature_rows = []
    for dataset in args.datasets:
        feats = pd.read_csv(Path(args.feature_dir) / f"{dataset}_panel.csv")
        wins = pd.read_csv(Path(args.window_dir) / f"{dataset}_windows_noisy.csv")
        bags = pd.read_csv(Path(args.bag_dir) / f"{dataset}_v2_noisy.csv")
        motifs = [args.motif] if args.motif else list(selected_motifs_for(dataset))
        for motif in motifs:
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = region_label(bags, motif)
            fold, loo, ranking = evaluate_mlp_ig(
                tab, y_r, families, steps=args.steps, epochs=args.epochs,
                rank_features=True,
            )
            def _mean(frame: pd.DataFrame) -> tuple[float, float, float, int, int]:
                if frame.empty:
                    return float("nan"), float("nan"), float("nan"), 0, 0
                return (
                    float(frame["auroc_window_mean"].mean()),
                    float(frame["precision_at_20"].mean()),
                    float(frame["labeled_prev"].mean()),
                    int(frame["n_regions_scored"].sum()),
                    int(len(frame)),
                )

            auroc, pwr, prev, n_regions, n_folds = _mean(fold)
            loo_auroc, loo_pwr, loo_prev, loo_regions, loo_folds = _mean(loo)
            ranked = ranking["feature"].tolist() if len(ranking) else []
            rec = recovery_at_k(ranked, motif, k=K) if ranked else {
                "hit_at_k": 0, "precision_at_k": float("nan"), "best_hit_rank": None,
                "top1": "", "miss_as_top1": False, "passed": False, "k": K,
            }
            summary = {
                "dataset": dataset,
                "motif": motif,
                "feature_group": group,
                "method": "mlp",
                "n_windows": int(len(tab)),
                "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
                "auroc_window_mean": auroc,
                "precision_at_20": pwr,
                "labeled_prev": prev,
                "n_regions_scored": n_regions,
                "n_folds": n_folds,
                "pass_window": bool(auroc == auroc and auroc >= PASS_WINDOW),
                "pass_precision_at_20": bool(pwr == pwr and pwr >= PASS_PRECISION_AT),
                "hit_at_5": rec["hit_at_k"],
                "precision_at_5": rec["precision_at_k"],
                "best_hit_rank": rec["best_hit_rank"],
                "top1": rec["top1"],
                "miss_as_top1": rec["miss_as_top1"],
                "feature_passed": rec["passed"],
            }
            loo_summary = {
                **summary,
                "method": "mlp_loo",
                "auroc_window_mean": loo_auroc,
                "precision_at_20": loo_pwr,
                "labeled_prev": loo_prev,
                "n_regions_scored": loo_regions,
                "n_folds": loo_folds,
                "pass_window": bool(loo_auroc == loo_auroc and loo_auroc >= PASS_WINDOW),
                "pass_precision_at_20": bool(loo_pwr == loo_pwr and loo_pwr >= PASS_PRECISION_AT),
                "hit_at_5": 0,
                "precision_at_5": float("nan"),
                "best_hit_rank": None,
                "top1": "",
                "miss_as_top1": False,
                "feature_passed": False,
            }
            rows.extend([summary, loo_summary])
            print(summary, flush=True)
            print(loo_summary, flush=True)
            if len(ranking):
                top = ranking.head(15).copy()
                top.insert(0, "motif", motif)
                top.insert(0, "dataset", dataset)
                feature_rows.append(top)
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
            if feature_rows:
                pd.concat(feature_rows, ignore_index=True).to_csv(args.features_out, index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
