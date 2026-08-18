#!/usr/bin/env python
"""Verify interpretability methods on motif pseudo-labels with known generators.

Protocol: train on hnc_wu2022 v2 selected labels, rank explanations, recover
the known motif parts, then ablate top features (faithfulness).

Tabular Lasso + linear SHAP:

    python scripts/verify_pseudo_label_explanations.py \\
        --dataset hnc_wu2022 --data-root "$DATA_ROOT" \\
        --labels results/pseudo_labels/hnc_wu2022_v2.csv \\
        --label-version v2 \\
        --feature-sources composition density point-pattern \\
        --by-type

Embedding probe (precomputed region embeddings):

    python scripts/verify_pseudo_label_explanations.py --mode embedding \\
        --embeddings-csv path/to/region_embeddings.csv

UTAG / other named region features:

    python scripts/verify_pseudo_label_explanations.py --mode precomputed \\
        --features-csv results/utag_region_features.csv

GNN-explainer node attributions (CSV with region_id, cell_type, importance):

    python scripts/verify_pseudo_label_explanations.py --mode gnn-explainer \\
        --explainer-csv results/gnn_explainer_nodes.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.basic_feats import CompositionFeaturizer  # noqa: E402
from benchmark.features.density_feats import CellTypeDensityFeaturizer  # noqa: E402
from benchmark.features.point_pattern import PointPatternFeaturizer  # noqa: E402
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.overlay import attach_pseudo_labels  # noqa: E402
from benchmark.motifs.recovery import (  # noqa: E402
    GNN_EXPECTED_SETS,
    RULES,
    SELECTED_EXPLAIN_TASKS,
    rank_recovery,
)
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation.splits import safe_patient_kfold, stratify_column  # noqa: E402


def _featurizer(name: str, cell_type_col: str, by_type: bool):
    if name == "composition":
        return CompositionFeaturizer(cell_type_col=cell_type_col)
    if name == "density":
        return CellTypeDensityFeaturizer(cell_type_col=cell_type_col)
    if name == "point-pattern":
        return PointPatternFeaturizer(
            cell_type_col=cell_type_col,
            radii=[10, 20, 50, 100, 200],
            metrics=("K", "L"),
            by_type=by_type,
        )
    raise ValueError(f"Unsupported feature source {name!r}")


def _extract_table(dataset, region_ids, sources, cell_type_col, by_type, normalize):
    regions = dataset.load_regions(list(region_ids), normalize=normalize)
    frames = []
    for source in sources:
        feat = _featurizer(source, cell_type_col, by_type).fit(regions)
        table = feat.transform(regions)
        table = table.add_prefix(f"{source}::")
        frames.append(table)
    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra, how="outer")
    out.index = out.index.astype(str)
    return out


def _positive_proba(model: LinearClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict(X)
    classes = list(model.classes_)
    if 1 in classes:
        return np.asarray(proba)[:, classes.index(1)]
    return np.asarray(proba)[:, -1]


def _auc(y: pd.Series, score: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _lasso_rank(model: LinearClassifier) -> list[str]:
    coef = np.asarray(model._clf.coef_, dtype=float).ravel()
    if coef.size != len(model._keep):
        coef = coef.reshape(len(model.classes_), -1)
        pos = list(model.classes_).index(1) if 1 in list(model.classes_) else -1
        coef = coef[pos]
    order = np.argsort(-np.abs(coef))
    return [model._keep[i] for i in order]


def _shap_rank(model: LinearClassifier, X: pd.DataFrame) -> list[str]:
    Xs = model._prep_pred(X)
    coef = np.asarray(model._clf.coef_, dtype=float).ravel()
    if coef.size != Xs.shape[1]:
        coef = coef.reshape(len(model.classes_), -1)
        pos = list(model.classes_).index(1) if 1 in list(model.classes_) else -1
        coef = coef[pos]
    contrib = np.abs(Xs * coef)
    mean_abs = contrib.mean(axis=0)
    order = np.argsort(-mean_abs)
    return [model._keep[i] for i in order]


def _faithfulness(model: LinearClassifier, X: pd.DataFrame, y: pd.Series, k: int, rng: np.random.Generator):
    names = _shap_rank(model, X)[:k]
    base = _auc(y, _positive_proba(model, X))
    top = X.copy()
    for name in names:
        if name in top.columns:
            top[name] = rng.permutation(top[name].to_numpy())
    drop_top = base - _auc(y, _positive_proba(model, top))
    rand_names = list(rng.choice(list(X.columns), size=min(k, X.shape[1]), replace=False))
    rnd = X.copy()
    for name in rand_names:
        rnd[name] = rng.permutation(rnd[name].to_numpy())
    drop_rand = base - _auc(y, _positive_proba(model, rnd))
    return base, drop_top, drop_rand, names


def _run_named_features(dataset, tasks, features: pd.DataFrame, seeds, k):
    rows = []
    rank_rows = []
    vcfg = dataset.validation_config
    patient_col = vcfg.get("patient_col", "patient_id")
    n_folds = vcfg.get("n_folds", 5)
    rng = np.random.default_rng(0)
    features = features.copy()
    features.index = features.index.astype(str)

    for task in tasks:
        if task not in RULES:
            print(f"skip {task}: no recovery rule", flush=True)
            continue
        rule = RULES[task]
        meta = dataset.get_task_metadata(task)
        meta = meta[meta["region_id"].astype(str).isin(features.index)].copy()
        if meta.empty:
            print(f"skip {task}: no labelled regions", flush=True)
            continue
        y_all = dataset.build_target(meta["region_id"].astype(str).tolist(), task)
        print(f"=== {task}  n={len(meta)}  kind={rule.kind} ===", flush=True)
        fold_pass = []
        for seed in seeds:
            folds = safe_patient_kfold(
                meta, n_folds, patient_col, stratify_column(dataset.get_task_config(task)), seed
            )
            if folds is None:
                continue
            for fold_i, (train_ids, val_ids) in enumerate(folds):
                train_ids = [str(i) for i in train_ids if str(i) in features.index]
                val_ids = [str(i) for i in val_ids if str(i) in features.index]
                if len(train_ids) < 8 or len(val_ids) < 4:
                    continue
                X_tr, X_va = features.loc[train_ids], features.loc[val_ids]
                y_tr = y_all.reindex(train_ids)
                y_va = y_all.reindex(val_ids)
                mask_tr = y_tr.notna()
                mask_va = y_va.notna()
                X_tr, y_tr = X_tr.loc[mask_tr], y_tr.loc[mask_tr]
                X_va, y_va = X_va.loc[mask_va], y_va.loc[mask_va]
                if y_tr.nunique() < 2 or y_va.nunique() < 2:
                    continue
                lasso = LinearClassifier(seed=seed, C=0.5, l1_ratio=1.0).fit(X_tr, y_tr)
                ridge = LinearClassifier(seed=seed, C=1.0, l1_ratio=0.0).fit(X_tr, y_tr)
                lasso_rank = _lasso_rank(lasso)
                shap_rank = _shap_rank(ridge, X_va)
                rec_l = rank_recovery(lasso_rank, rule, k=k)
                rec_s = rank_recovery(shap_rank, rule, k=k)
                base, drop_top, drop_rand, top_names = _faithfulness(
                    ridge, X_va, y_va, k=k, rng=rng
                )
                faithful = bool(drop_top > drop_rand)
                passed = bool(rec_l["passed"] and rec_s["passed"] and faithful)
                fold_pass.append(passed)
                rows.append(dict(
                    task=task, kind=rule.kind, seed=seed, fold=fold_i,
                    n_train=len(X_tr), n_val=len(X_va),
                    auc_lasso=_auc(y_va, _positive_proba(lasso, X_va)),
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
                    f"  seed{seed}/fold{fold_i}  ridgeAUC={base:.3f}  "
                    f"lasso={rec_l['passed']} shap={rec_s['passed']} "
                    f"faith={faithful}  shap_top1={rec_s['top1'][:60]}",
                    flush=True,
                )
        if fold_pass:
            print(f"  TASK {task}: {sum(fold_pass)}/{len(fold_pass)} folds passed", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def _summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return fold_df
    if fold_df["kind"].eq("control").any():
        control_frac = float(fold_df.loc[fold_df["kind"].eq("control"), "fold_passed"].mean())
    else:
        control_frac = float("nan")
    rows = []
    for task, sub in fold_df.groupby("task"):
        kind = sub["kind"].iloc[0]
        frac_pass = float(sub["fold_passed"].mean())
        if kind == "control":
            verdict = "pass_control" if frac_pass >= 0.5 else "fail_control"
        elif not np.isfinite(control_frac) or control_frac < 0.5:
            verdict = "fail_control_block"
        elif frac_pass >= 0.5:
            verdict = "pass_spatial"
        else:
            verdict = "abundance_only_or_fail"
        rows.append(dict(
            task=task,
            kind=kind,
            n_folds=len(sub),
            mean_auc_ridge=float(sub["auc_ridge"].mean()),
            mean_auc_lasso=float(sub["auc_lasso"].mean()),
            frac_lasso_passed=float(sub["lasso_passed"].mean()),
            frac_shap_passed=float(sub["shap_passed"].mean()),
            frac_faithfulness_passed=float(sub["faithfulness_passed"].mean()),
            frac_fold_passed=frac_pass,
            mean_drop_top=float(sub["faithfulness_drop_top"].mean()),
            mean_drop_random=float(sub["faithfulness_drop_random"].mean()),
            control_frac_passed=control_frac,
            verdict=verdict,
        ))
    order = {t: i for i, t in enumerate(SELECTED_EXPLAIN_TASKS)}
    out = pd.DataFrame(rows)
    return out.sort_values("task", key=lambda s: s.map(lambda t: order.get(t, 99)))


def _embedding_probe(dataset, tasks, embeddings: pd.DataFrame, seeds):
    rows = []
    vcfg = dataset.validation_config
    patient_col = vcfg.get("patient_col", "patient_id")
    n_folds = vcfg.get("n_folds", 5)
    embeddings = embeddings.copy()
    embeddings.index = embeddings.index.astype(str)
    for task in tasks:
        meta = dataset.get_task_metadata(task)
        meta = meta[meta["region_id"].astype(str).isin(embeddings.index)].copy()
        y_all = dataset.build_target(meta["region_id"].astype(str).tolist(), task)
        print(f"=== embedding probe {task} n={len(meta)} ===", flush=True)
        aucs = []
        for seed in seeds:
            folds = safe_patient_kfold(
                meta, n_folds, patient_col, stratify_column(dataset.get_task_config(task)), seed
            )
            if folds is None:
                continue
            for fold_i, (train_ids, val_ids) in enumerate(folds):
                train_ids = [str(i) for i in train_ids if str(i) in embeddings.index]
                val_ids = [str(i) for i in val_ids if str(i) in embeddings.index]
                y_tr = y_all.reindex(train_ids).dropna()
                y_va = y_all.reindex(val_ids).dropna()
                if y_tr.nunique() < 2 or y_va.nunique() < 2:
                    continue
                model = LinearClassifier(seed=seed).fit(embeddings.loc[y_tr.index], y_tr)
                auc = _auc(y_va, _positive_proba(model, embeddings.loc[y_va.index]))
                aucs.append(auc)
                rows.append(dict(task=task, seed=seed, fold=fold_i, auc=auc))
        if aucs:
            kind = RULES[task].kind if task in RULES else "unknown"
            passed = (np.nanmean(aucs) >= 0.90) if kind == "control" else (np.nanmean(aucs) >= 0.60)
            print(
                f"  mean AUC={np.nanmean(aucs):.3f}  "
                f"{'HAS_SIGNAL' if passed else 'NO_SPATIAL_OR_WEAK'}",
                flush=True,
            )
    return pd.DataFrame(rows)


def _rename_explainer_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"region_id", "core_id", "sample_id"} and "region_id" not in mapping.values():
            mapping[col] = "region_id"
        elif key in {"cell_type", "cell_type_uniform", "type"} and "cell_type" not in mapping.values():
            mapping[col] = "cell_type"
        elif key in {"importance", "score", "attribution", "node_mask"} and "importance" not in mapping.values():
            mapping[col] = "importance"
    out = df.rename(columns=mapping)
    missing = [c for c in ("region_id", "cell_type", "importance") if c not in out.columns]
    if missing:
        raise ValueError(
            "GNN-explainer CSV must include region_id, cell_type, importance "
            f"(aliases accepted). Missing: {missing}. Columns: {list(df.columns)}"
        )
    return out


def _run_gnn_explainer(dataset, catalog, tasks, table: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    table = _rename_explainer_columns(table)
    table = table.copy()
    table["region_id"] = table["region_id"].astype(str)
    table["cell_type"] = table["cell_type"].astype(str).str.lower()
    table["importance"] = pd.to_numeric(table["importance"], errors="coerce")
    table = table.dropna(subset=["importance"])
    rows = []
    for task in tasks:
        sets = GNN_EXPECTED_SETS.get(task)
        if not sets:
            print(f"skip {task}: no GNN expected cell sets", flush=True)
            continue
        expected = {str(name).lower() for sid in sets for name in catalog.cell_sets[sid]}
        meta = dataset.get_task_metadata(task)
        labeled = set(meta["region_id"].astype(str))
        kind = RULES[task].kind if task in RULES else "unknown"
        print(f"=== gnn-explainer {task}  labelled={len(labeled)} ===", flush=True)
        for rid, grp in table.groupby("region_id"):
            if rid not in labeled:
                continue
            n = len(grp)
            k = max(5, int(np.ceil(top_frac * n)))
            top = grp.nlargest(min(k, n), "importance")
            top_hit = float(top["cell_type"].isin(expected).mean())
            bg = float(grp["cell_type"].isin(expected).mean())
            enrich = float(top_hit / bg) if bg > 0 else (np.inf if top_hit > 0 else 1.0)
            passed = bool(np.isfinite(enrich) and enrich >= 1.25 and top_hit > bg)
            rows.append(dict(
                task=task, kind=kind, region_id=rid, n_cells=n, top_k=k,
                top_expected_frac=top_hit, background_frac=bg,
                enrichment=enrich if np.isfinite(enrich) else np.nan,
                passed=passed,
            ))
        sub = [r for r in rows if r["task"] == task]
        if sub:
            frac = float(np.mean([r["passed"] for r in sub]))
            print(f"  {sum(r['passed'] for r in sub)}/{len(sub)} regions passed  frac={frac:.3f}", flush=True)
    return pd.DataFrame(rows)


def _summarize_gnn(region_df: pd.DataFrame) -> pd.DataFrame:
    if region_df.empty:
        return region_df
    if region_df["kind"].eq("control").any():
        control_frac = float(region_df.loc[region_df["kind"].eq("control"), "passed"].mean())
    else:
        control_frac = float("nan")
    rows = []
    for task, sub in region_df.groupby("task"):
        kind = sub["kind"].iloc[0]
        frac = float(sub["passed"].mean())
        if kind == "control":
            verdict = "pass_control" if frac >= 0.5 else "fail_control"
        elif not np.isfinite(control_frac) or control_frac < 0.5:
            verdict = "fail_control_block"
        elif frac >= 0.5:
            verdict = "pass_spatial"
        else:
            verdict = "abundance_only_or_fail"
        rows.append(dict(
            task=task, kind=kind, n_regions=len(sub),
            mean_enrichment=float(sub["enrichment"].mean()),
            frac_passed=frac, control_frac_passed=control_frac, verdict=verdict,
        ))
    return pd.DataFrame(rows)


def _cache_meta_path(cache: Path) -> Path:
    return cache.with_name(cache.name + ".meta.json")


def _load_or_extract_features(dataset, tasks, args, catalog, cache: Path) -> pd.DataFrame:
    wanted = {
        "sources": list(args.feature_sources),
        "by_type": bool(args.by_type or ("point-pattern" in args.feature_sources)),
        "cell_type_col": catalog.cell_type_col,
    }
    meta_path = _cache_meta_path(cache)
    if cache.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved == wanted:
            print(f"Loading cached features {cache}", flush=True)
            features = pd.read_csv(cache, index_col=0)
            features.index = features.index.astype(str)
            return features
        print(f"Cache meta mismatch {saved} vs {wanted}; re-extracting", flush=True)
    elif cache.exists():
        print(f"Cache {cache} has no meta; re-extracting to be safe", flush=True)

    labeled = set()
    for task in tasks:
        meta = dataset.get_task_metadata(task)
        labeled.update(meta["region_id"].astype(str))
    print(
        f"Extracting {wanted['sources']} by_type={wanted['by_type']} "
        f"on {len(labeled)} labelled regions",
        flush=True,
    )
    features = _extract_table(
        dataset,
        sorted(labeled),
        wanted["sources"],
        catalog.cell_type_col,
        wanted["by_type"],
        normalize=False,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(cache)
    meta_path.write_text(json.dumps(wanted, indent=2))
    print(f"Wrote feature cache {cache}  shape={features.shape}", flush=True)
    return features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--label-version", default="v2", choices=["v1", "v2"])
    ap.add_argument(
        "--mode",
        default="tabular",
        choices=["tabular", "embedding", "precomputed", "gnn-explainer"],
    )
    ap.add_argument(
        "--feature-sources",
        nargs="+",
        default=["composition", "density", "point-pattern"],
    )
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--features-csv", default=None)
    ap.add_argument("--embeddings-csv", default=None)
    ap.add_argument("--explainer-csv", default=None)
    ap.add_argument("--gnn-top-frac", type=float, default=0.10)
    ap.add_argument("--region-id-col", default="region_id")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cache-features", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    catalog = load_motif_catalog(args.catalog, dataset=args.dataset)
    labels_path = Path(args.labels or (_CODE / "results" / "pseudo_labels" / f"{args.dataset}_v2.csv"))
    dataset = load_dataset(args.dataset, data_root=args.data_root)
    attach_pseudo_labels(dataset, labels_path, catalog, label_version=args.label_version)
    tasks = args.tasks or list(SELECTED_EXPLAIN_TASKS)
    out_dir = Path(args.output_dir or (_CODE / "results" / "pseudo_label_explanations"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "embedding":
        if not args.embeddings_csv:
            raise ValueError("--embeddings-csv is required for embedding mode")
        emb = pd.read_csv(args.embeddings_csv)
        emb = emb.set_index(args.region_id_col)
        emb = emb.select_dtypes(include=[np.number])
        fold_df = _embedding_probe(dataset, tasks, emb, args.seeds)
        fold_df.to_csv(out_dir / "embedding_probe_folds.csv", index=False)
        summary = (
            fold_df.groupby("task")["auc"].agg(["mean", "std", "count"]).reset_index()
            if not fold_df.empty else fold_df
        )
        summary.to_csv(out_dir / "embedding_probe_summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)
        print(f"Wrote {out_dir}", flush=True)
        return

    if args.mode == "gnn-explainer":
        if not args.explainer_csv:
            raise ValueError("--explainer-csv is required for gnn-explainer mode")
        table = pd.read_csv(args.explainer_csv)
        region_df = _run_gnn_explainer(dataset, catalog, tasks, table, args.gnn_top_frac)
        summary = _summarize_gnn(region_df)
        region_df.to_csv(out_dir / "gnn_region_recovery.csv", index=False)
        summary.to_csv(out_dir / "gnn_summary.csv", index=False)
        print("\n=== gnn summary ===", flush=True)
        print(summary.to_string(index=False), flush=True)
        print(f"Wrote {out_dir}", flush=True)
        return

    if args.mode == "precomputed":
        if not args.features_csv:
            raise ValueError("--features-csv is required for precomputed mode")
        features = pd.read_csv(args.features_csv).set_index(args.region_id_col)
        features = features.select_dtypes(include=[np.number])
        features.index = features.index.astype(str)
    else:
        cache = Path(args.cache_features) if args.cache_features else out_dir / "tabular_features.csv"
        features = _load_or_extract_features(dataset, tasks, args, catalog, cache)

    fold_df, rank_df = _run_named_features(
        dataset, tasks, features, args.seeds, k=args.top_k
    )
    summary = _summarize(fold_df)
    fold_df.to_csv(out_dir / "fold_recovery.csv", index=False)
    rank_df.to_csv(out_dir / "feature_ranks.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
