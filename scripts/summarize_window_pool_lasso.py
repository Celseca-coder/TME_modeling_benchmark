#!/usr/bin/env python
"""Summarize window-pooled Lasso / SHAP ranks in the global family table layout.

Precision@k is the fraction of the top k ranks that match the task's hit regex.
Recall@k divides by min(k, number of hit columns), so a family with fewer aimed
columns than k is not penalised for the empty slots. ``frac_H_over_F`` is the
share of family columns that match the hit regex.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.panel import jobs_by_dataset, load_selected_panel  # noqa: E402
from benchmark.motifs.recovery import RULES, feature_hits  # noqa: E402

FAMILY_NAME = {
    "composition": "composition",
    "density": "density",
    "mixing": "mixing",
    "point-pattern": "point_pattern",
    "joint": "composition_density_mixing_point_pattern",
}
FAMILY_ORDER = list(FAMILY_NAME)
DATASET_ORDER = (
    "bc_jackson2020",
    "bc_metabric_ali2020",
    "hnc_wu2022",
    "tnbc_wang2023",
)
KS = (1, 3, 5)
LIST_LIMIT = 10


def _column_names(path: Path) -> list[str]:
    """Read the cached column list without materialising the window bags."""

    class _Skip:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            pass

    class _Loader(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy"):
                return _Skip
            return super().find_class(module, name)

    with path.open("rb") as handle:
        payload = _Loader(handle).load()
    return list(payload["columns"])


def _family_columns(columns: list[str], group: str) -> list[str]:
    if group == "joint":
        return list(columns)
    prefix = f"{group}::"
    return [col for col in columns if col.startswith(prefix)]


def _format_hits(counts: Counter, n_folds: int) -> str:
    if not counts or n_folds == 0:
        return ""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    shown = ranked[:LIST_LIMIT]
    text = "; ".join(f"{name} ({count}/{n_folds})" for name, count in shown)
    extra = len(ranked) - len(shown)
    if extra > 0:
        text = f"{text}; +{extra} more"
    return text


def _summarize_method(folds: pd.DataFrame, ranks: pd.DataFrame, hit_columns: list[str], method: str) -> dict:
    n_folds = int(len(folds))
    auc_col = "auc_lasso" if method == "lasso" else "auc_ridge"
    auc = folds[auc_col].to_numpy(float)
    hit_set = set(hit_columns)
    n_hit = len(hit_columns)
    per_k = {k: [] for k in KS}
    lists = {k: Counter() for k in KS}
    for row in folds.itertuples(index=False):
        part = ranks[
            (ranks["seed"] == row.seed)
            & (ranks["fold"] == row.fold)
            & (ranks["method"] == method)
        ].sort_values("rank")
        ordered = part["feature"].astype(str).tolist()
        for k in KS:
            chosen = [name for name in ordered[:k] if name in hit_set]
            per_k[k].append(chosen)
            lists[k].update(set(chosen))
    out = {
        "auc_roc": float(np.nanmean(auc)) if n_folds else np.nan,
        "auc_roc_sd": float(np.nanstd(auc, ddof=1)) if n_folds > 1 else np.nan,
        "n_hit_candidates": n_hit,
        "mean_n_hit_at_5": float(np.mean([len(names) for names in per_k[5]])) if n_folds else np.nan,
        "frac_n_hit_at_5_ge1": float(np.mean([len(names) >= 1 for names in per_k[5]])) if n_folds else np.nan,
    }
    for k in KS:
        out[f"top{k}_hit_features"] = _format_hits(lists[k], n_folds)
        precision = [len(names) / k for names in per_k[k]]
        out[f"precision_at_{k}"] = float(np.mean(precision)) if n_folds else np.nan
        if n_hit == 0:
            out[f"recall_at_{k}"] = np.nan
        else:
            denom = min(k, n_hit)
            out[f"recall_at_{k}"] = float(np.mean([len(names) / denom for names in per_k[k]])) if n_folds else np.nan
    return out


def build_table(folds: pd.DataFrame, ranks: pd.DataFrame, columns_by_dataset: dict[str, list[str]], panel: pd.DataFrame) -> pd.DataFrame:
    jobs = jobs_by_dataset(panel)
    rows = []
    for group, family in FAMILY_NAME.items():
        for dataset in DATASET_ORDER:
            tasks = jobs.get(dataset, [])
            columns = _family_columns(columns_by_dataset[dataset], group)
            n_features = len(columns)
            for task in tasks:
                rule = RULES[task]
                hit_columns = [col for col in columns if feature_hits(col, rule.hit)]
                sub_folds = folds[
                    (folds["dataset"] == dataset)
                    & (folds["feature_group"] == group)
                    & (folds["task"] == task)
                ]
                sub_ranks = ranks[
                    (ranks["dataset"] == dataset)
                    & (ranks["feature_group"] == group)
                    & (ranks["task"] == task)
                ]
                if sub_folds.empty:
                    continue
                for method in ("lasso", "shap"):
                    stats = _summarize_method(sub_folds, sub_ranks, hit_columns, method)
                    stats.update({
                        "family": family,
                        "dataset": dataset,
                        "task": task,
                        "kind": rule.kind,
                        "method": method,
                        "frac_H_over_F": (len(hit_columns) / n_features) if n_features else np.nan,
                    })
                    rows.append(stats)
    table = pd.DataFrame(rows)
    columns = [
        "family", "dataset", "task", "kind", "method",
        "auc_roc", "auc_roc_sd", "n_hit_candidates",
        "top1_hit_features", "top3_hit_features", "top5_hit_features",
        "mean_n_hit_at_5", "frac_n_hit_at_5_ge1",
        "precision_at_1", "precision_at_3", "precision_at_5",
        "recall_at_1", "recall_at_3", "recall_at_5",
        "frac_H_over_F",
    ]
    return table[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=str(_CODE / "results" / "window_pool_lasso" / "mean" / "fold_recovery.csv"))
    parser.add_argument("--ranks", default=str(_CODE / "results" / "window_pool_lasso" / "mean" / "feature_ranks.csv"))
    parser.add_argument("--bag-dir", default=str(_CODE / "results" / "window_pool" / "bags"))
    parser.add_argument("--panel", default=str(_CODE / "results" / "pseudo_labels_all" / "selected_catalog.csv"))
    parser.add_argument("--output", default=str(_CODE / "results" / "window_pool_lasso" / "mean" / "family_recovery.csv"))
    args = parser.parse_args()

    folds = pd.read_csv(args.folds)
    ranks = pd.read_csv(args.ranks)
    panel = load_selected_panel(args.panel)
    columns_by_dataset = {}
    for dataset in sorted(folds["dataset"].unique()):
        path = Path(args.bag_dir) / f"{dataset}.pkl"
        columns_by_dataset[dataset] = _column_names(path)
        print(f"{dataset}: {len(columns_by_dataset[dataset])} columns", flush=True)
    table = build_table(folds, ranks, columns_by_dataset, panel)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False, na_rep="")
    print(f"wrote {dest}  ({len(table)} rows)", flush=True)


if __name__ == "__main__":
    main()
