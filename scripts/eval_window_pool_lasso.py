#!/usr/bin/env python
"""Pool square-window features, then rank them with the global Lasso.

Mean pooling averages each column across the windows of a region. Attention
pooling learns one weight per window inside the training fold and applies it
to the raw window vector. Both tables use the same column names as the global
tabular explanation, then the same L1 logistic regression (C=0.5) and top-5
recovery rules.

    python scripts/eval_window_pool_lasso.py --pooling mean --datasets hnc_wu2022
    python scripts/eval_window_pool_lasso.py --pooling attention --datasets hnc_wu2022
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.window_pool import (  # noqa: E402
    WINDOW_SOURCES,
    _region_bag,
    attention_weighted_mean,
    columns_for_source,
    feature_names,
    load_bags,
    make_featurizers,
    mean_pool_table,
    save_bags,
)
from benchmark.models.attention_mil import AttentionMILModel  # noqa: E402
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.panel import jobs_by_dataset, load_selected_panel  # noqa: E402
from benchmark.motifs.overlay import attach_pseudo_labels  # noqa: E402
from benchmark.motifs.recovery import RULES, rank_recovery  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation.splits import safe_patient_kfold, stratify_column  # noqa: E402
from scripts.verify_pseudo_label_explanations import (  # noqa: E402
    _auc,
    _faithfulness,
    _lasso_rank,
    _positive_proba,
    _run_named_features,
    _shap_rank,
    _summarize,
)

LABEL_TEMPLATE = (
    "/autofs/nas8/tywang/tjzou/PseudoLabelDatasetExtreme20/region/per_dataset/{dataset}_v2_noisy.csv"
)
DATA_ROOT = "/autofs/bal14/zqwu/CellularTables/TME_benchmark_data"


def _bag_frame(bags: dict[str, np.ndarray], columns: list[str], region_ids) -> pd.DataFrame:
    rows = []
    index = []
    wanted = set(map(str, region_ids))
    for region_id, bag in bags.items():
        if region_id not in wanted or bag is None or len(bag) == 0:
            continue
        rows.append({"bag": np.asarray(bag, dtype=np.float32)})
        index.append(region_id)
    if not rows:
        return pd.DataFrame(columns=["bag"])
    return pd.DataFrame(rows, index=index)


def _pool_with_attention(model: AttentionMILModel, bags, columns, region_ids) -> pd.DataFrame:
    rows = []
    index = []
    for region_id in map(str, region_ids):
        bag = bags.get(region_id)
        if bag is None or len(bag) == 0:
            continue
        weights = model.attention_weights(pd.DataFrame({"bag": [bag]}))[0]
        rows.append(attention_weighted_mean(bag, weights))
        index.append(region_id)
    table = pd.DataFrame(rows, index=index, columns=columns)
    table.index.name = "region_id"
    return table


def _score_attention(
    dataset,
    tasks,
    bags,
    columns,
    seeds,
    k,
    epochs,
    patience,
    device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient CV. Attention is fit on the training fold, then Lasso sees the pooled rows."""
    rows = []
    rank_rows = []
    vcfg = dataset.validation_config
    patient_col = vcfg.get("patient_col", "patient_id")
    n_folds = vcfg.get("n_folds", 5)
    rng = np.random.default_rng(0)
    for task in tasks:
        if task not in RULES:
            print(f"skip {task}: no recovery rule", flush=True)
            continue
        rule = RULES[task]
        present = [rid for rid, bag in bags.items() if bag is not None and len(bag)]
        meta = dataset.get_task_metadata(task)
        meta = meta[meta["region_id"].astype(str).isin(present)].copy()
        if meta.empty:
            print(f"skip {task}: no labelled regions", flush=True)
            continue
        y_all = dataset.build_target(meta["region_id"].astype(str).tolist(), task)
        print(f"=== {task}  n={len(meta)}  kind={rule.kind} ===", flush=True)
        for seed in seeds:
            folds = safe_patient_kfold(
                meta, n_folds, patient_col, stratify_column(dataset.get_task_config(task)), seed
            )
            if folds is None:
                continue
            for fold_i, (train_ids, val_ids) in enumerate(folds):
                train_ids = [str(i) for i in train_ids if str(i) in bags]
                val_ids = [str(i) for i in val_ids if str(i) in bags]
                y_tr = y_all.reindex(train_ids)
                y_va = y_all.reindex(val_ids)
                y_tr = y_tr[y_tr.notna()]
                y_va = y_va[y_va.notna()]
                if len(y_tr) < 8 or len(y_va) < 4 or y_tr.nunique() < 2 or y_va.nunique() < 2:
                    continue
                train_bags = _bag_frame(bags, columns, y_tr.index)
                y_fit = y_tr.reindex(train_bags.index).dropna()
                if y_fit.nunique() < 2:
                    continue
                model = AttentionMILModel(
                    seed=seed + fold_i,
                    epochs=epochs,
                    patience=patience,
                    max_instances=None,
                    device=device,
                ).fit(train_bags.loc[y_fit.index], y_fit.astype(int))
                pooled_tr = _pool_with_attention(model, bags, columns, y_tr.index)
                pooled_va = _pool_with_attention(model, bags, columns, y_va.index)
                y_tr = y_tr.reindex(pooled_tr.index).dropna()
                y_va = y_va.reindex(pooled_va.index).dropna()
                pooled_tr = pooled_tr.loc[y_tr.index]
                pooled_va = pooled_va.loc[y_va.index]
                if y_tr.nunique() < 2 or y_va.nunique() < 2:
                    continue
                lasso = LinearClassifier(seed=seed, C=0.5, l1_ratio=1.0).fit(pooled_tr, y_tr)
                ridge = LinearClassifier(seed=seed, C=1.0, l1_ratio=0.0).fit(pooled_tr, y_tr)
                lasso_rank = _lasso_rank(lasso)
                shap_rank = _shap_rank(ridge, pooled_va)
                rec_l = rank_recovery(lasso_rank, rule, k=k)
                rec_s = rank_recovery(shap_rank, rule, k=k)
                base, drop_top, drop_rand, top_names = _faithfulness(
                    ridge, pooled_va, y_va, k=k, rng=rng
                )
                faithful = bool(drop_top > drop_rand)
                passed = bool(rec_l["passed"] and rec_s["passed"] and faithful)
                rows.append(dict(
                    task=task, kind=rule.kind, seed=seed, fold=fold_i,
                    n_train=len(pooled_tr), n_val=len(pooled_va),
                    auc_lasso=_auc(y_va, _positive_proba(lasso, pooled_va)),
                    auc_ridge=base,
                    lasso_top1=rec_l["top1"],
                    shap_top1=rec_s["top1"],
                    lasso_hit=rec_l[f"hit_in_top{k}"],
                    shap_hit=rec_s[f"hit_in_top{k}"],
                    lasso_best_hit_rank=rec_l["best_hit_rank"],
                    shap_best_hit_rank=rec_s["best_hit_rank"],
                    lasso_passed=rec_l["passed"],
                    shap_passed=rec_s["passed"],
                    shap_miss_as_top1=rec_s["miss_as_top1"],
                    faithfulness_drop_top=drop_top,
                    faithfulness_drop_random=drop_rand,
                    faithfulness_passed=faithful,
                    fold_passed=passed,
                    top_shap=";".join(top_names),
                ))
                for method, ranked in (("lasso", lasso_rank[:20]), ("shap", shap_rank[:20])):
                    for rank, feat in enumerate(ranked, start=1):
                        rank_rows.append(dict(
                            task=task, seed=seed, fold=fold_i, method=method,
                            rank=rank, feature=feat,
                        ))
                print(
                    f"  seed{seed}/fold{fold_i}  lassoAUC={rows[-1]['auc_lasso']:.3f}  "
                    f"lasso={rec_l['passed']}  top1={rec_l['top1'][:60]}",
                    flush=True,
                )
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def _tag(folds: pd.DataFrame, ranks: pd.DataFrame, dataset: str, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not folds.empty:
        folds = folds.copy()
        folds["dataset"] = dataset
        folds["feature_group"] = group
    if not ranks.empty:
        ranks = ranks.copy()
        ranks["dataset"] = dataset
        ranks["feature_group"] = group
    return folds, ranks


def _usable(table: pd.DataFrame) -> pd.DataFrame:
    table = table.dropna(axis=1, how="all")
    keep = [col for col in table.columns if np.isfinite(table[col].to_numpy(float)).any()]
    return table[keep]


def _score_table(dataset, tasks, table, seeds, k, dataset_name, group):
    table = _usable(table)
    if table.shape[1] == 0:
        print(f"skip {group}: no usable columns", flush=True)
        return pd.DataFrame(), pd.DataFrame()
    print(f"--- {dataset_name} / {group}  regions={len(table)}  cols={table.shape[1]} ---", flush=True)
    folds, ranks = _run_named_features(dataset, tasks, table, seeds, k, rules=RULES)
    return _tag(folds, ranks, dataset_name, group)


def _upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new.empty:
        if not path.exists():
            new.to_csv(path, index=False)
        return
    if path.exists():
        old = pd.read_csv(path)
        if all(col in old.columns for col in keys):
            idx = new.set_index(keys).index
            old = old[~old.set_index(keys).index.isin(idx)]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _labelled_region_ids(dataset, tasks) -> list[str]:
    ids: set[str] = set()
    for task in tasks:
        meta = dataset.get_task_metadata(task)
        target = dataset.build_target(meta["region_id"].astype(str).tolist(), task)
        ids.update(target[target.notna()].index.astype(str))
    return sorted(ids)


def _bag_meta(window_size, step, min_cells) -> dict:
    return {
        "window_size_um": float(window_size),
        "step_um": float(step),
        "min_cells": int(min_cells),
        "sources": list(WINDOW_SOURCES),
        "point_pattern_radii_um": [10, 20, 50],
    }


def _ensure_bags(dataset_name, regions, featurizers, cache: Path, window_size, step, min_cells):
    meta = _bag_meta(window_size, step, min_cells)
    if cache.exists():
        columns, bags, saved = load_bags(cache)
        if saved == meta:
            print(f"reuse {cache}  regions={len(bags)}  cols={len(columns)}", flush=True)
            return columns, bags
        print(f"cache meta {saved} != {meta}; re-extracting", flush=True)
    print(f"[{dataset_name}] extract windows on {len(regions)} regions", flush=True)
    bags = {}
    columns = feature_names(featurizers)
    for i, region in enumerate(regions, start=1):
        bags[str(region.region_id)] = _region_bag(
            region, featurizers, columns, window_size, step, min_cells,
        )
        if i == 1 or i % 10 == 0 or i == len(regions):
            print(
                f"[{dataset_name}] {i}/{len(regions)} {region.region_id} "
                f"windows={len(bags[str(region.region_id)])}",
                flush=True,
            )
    save_bags(cache, columns, bags, meta)
    print(f"wrote {cache}", flush=True)
    return columns, bags


def _selected_groups(args) -> list[str]:
    known = ("joint",) + WINDOW_SOURCES
    if args.groups:
        bad = [name for name in args.groups if name not in known]
        if bad:
            raise SystemExit(f"unknown feature groups: {bad}; choose from {list(known)}")
        return list(dict.fromkeys(args.groups))
    groups = ["joint"]
    if args.per_source:
        groups.extend(WINDOW_SOURCES)
    return groups


def run_dataset(args, dataset_name: str, tasks: list[str], out_dir: Path) -> None:
    catalog = load_motif_catalog(args.catalog, dataset=dataset_name)
    labels = Path(args.labels.format(dataset=dataset_name))
    dataset = load_dataset(dataset_name, data_root=args.data_root)
    attach_pseudo_labels(
        dataset, labels, catalog, label_version=args.label_version,
        motif_ids=[task.removeprefix("motif_") for task in tasks],
    )
    region_ids = _labelled_region_ids(dataset, tasks)
    print(f"[{dataset_name}] labelled regions={len(region_ids)} tasks={tasks}", flush=True)
    regions = dataset.load_regions(region_ids, normalize=False)
    featurizers = make_featurizers(catalog.cell_type_col)
    for feat in featurizers.values():
        feat.fit(regions)
    cache = Path(args.bag_dir) / f"{dataset_name}.pkl"
    columns, bags = _ensure_bags(
        dataset_name, regions, featurizers, cache,
        args.window_size, args.step, args.min_cells,
    )
    dataset.clear_region_cache()
    groups = _selected_groups(args)

    fold_frames = []
    rank_frames = []
    if args.pooling == "mean":
        table = mean_pool_table(bags, columns)
        for group in groups:
            part = table if group == "joint" else table[columns_for_source(columns, group)]
            folds, ranks = _score_table(
                dataset, tasks, part, args.seeds, args.top_k, dataset_name, group,
            )
            fold_frames.append(folds)
            rank_frames.append(ranks)
    else:
        sliced = {}
        for group in groups:
            if group == "joint":
                sliced[group] = (bags, columns)
                continue
            cols = columns_for_source(columns, group)
            idx = [columns.index(col) for col in cols]
            sliced[group] = (
                {rid: bag[:, idx] for rid, bag in bags.items()},
                cols,
            )
        for group, (group_bags, group_cols) in sliced.items():
            print(f"--- {dataset_name} / {group} attention ---", flush=True)
            folds, ranks = _score_attention(
                dataset, tasks, group_bags, group_cols, args.seeds, args.top_k,
                args.epochs, args.patience, args.device,
            )
            folds, ranks = _tag(folds, ranks, dataset_name, group)
            fold_frames.append(folds)
            rank_frames.append(ranks)

    folds = pd.concat([frame for frame in fold_frames if not frame.empty], ignore_index=True) if any(not f.empty for f in fold_frames) else pd.DataFrame()
    ranks = pd.concat([frame for frame in rank_frames if not frame.empty], ignore_index=True) if any(not f.empty for f in rank_frames) else pd.DataFrame()
    summary = _summarize(folds) if not folds.empty else pd.DataFrame()
    if not summary.empty and "feature_group" in folds.columns:
        pieces = []
        for group, sub in folds.groupby("feature_group"):
            part = _summarize(sub)
            part["feature_group"] = group
            pieces.append(part)
        summary = pd.concat(pieces, ignore_index=True)
    _upsert(out_dir / "fold_recovery.csv", folds, ["dataset", "task", "seed", "fold", "feature_group"])
    _upsert(out_dir / "feature_ranks.csv", ranks, ["dataset", "task", "seed", "fold", "method", "rank", "feature_group"])
    _upsert(out_dir / "summary.csv", summary, ["dataset", "task", "feature_group"])
    print(f"wrote {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pooling", choices=["mean", "attention"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--panel", default=str(_CODE / "results" / "pseudo_labels_all" / "selected_catalog.csv"))
    parser.add_argument("--labels", default=LABEL_TEMPLATE)
    parser.add_argument("--label-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window-size", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=50.0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--per-source", action="store_true", help="Also score each feature source on its own.")
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Groups to score: joint and/or composition density mixing point-pattern. "
             "Default is joint, plus each source when --per-source is set.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bag-dir", default=str(_CODE / "results" / "window_pool" / "bags"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    jobs = jobs_by_dataset(load_selected_panel(args.panel))
    if args.datasets:
        missing = [name for name in args.datasets if name not in jobs]
        if missing:
            raise SystemExit(f"datasets not in the selected panel: {missing}")
        jobs = {name: jobs[name] for name in args.datasets}
    out_dir = Path(args.output_dir or (_CODE / "results" / "window_pool_lasso" / args.pooling))
    for dataset_name, tasks in jobs.items():
        run_dataset(args, dataset_name, tasks, out_dir)


if __name__ == "__main__":
    main()
