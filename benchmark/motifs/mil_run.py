"""Localization + faithfulness protocol for gated AttnMIL on motif bags."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark.features.attention_mil import HandcraftedAttentionMILFeaturizer
from benchmark.models.attention_mil import AttentionMILModel
from benchmark.models.mil_explainers import (
    EXPLAINERS,
    explain_bag,
    faithfulness_metrics,
    random_scores,
)
from benchmark.motifs.mil_evidence import evidence_for_bag, spec_for_task, task_kind
from benchmark.validation.splits import safe_patient_kfold, stratify_column

CONTROL_AUC = 0.90
SPATIAL_AUC = 0.60
TOP_FRAC = 0.10
ENRICH_MIN = 1.25


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.size == 0 or y.max() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def localization_metrics(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict:
    pos = np.asarray(pos).astype(int)
    neg = np.asarray(neg).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = max(1, int(np.ceil(TOP_FRAC * n)))
    order = np.argsort(-score)
    top = pos[order[:k]]
    bg = float(pos.mean()) if n else 0.0
    top_hit = float(top.mean()) if k else 0.0
    enrich = (top_hit / bg) if bg > 0 else (np.inf if top_hit > 0 else 1.0)
    auprc_pos = _safe_auprc(pos, score)
    auprc_neg = _safe_auprc(neg, -score)
    auprc2 = np.nanmean([auprc_pos, auprc_neg])
    loc_passed = bool(
        np.isfinite(enrich) and enrich >= ENRICH_MIN and top_hit > bg
    )
    return dict(
        loc_auprc=auprc_pos,
        loc_auroc=_safe_auc(pos, score),
        loc_auprc_neg=auprc_neg,
        loc_auprc2=float(auprc2) if np.isfinite(auprc2) else float("nan"),
        loc_top_hit=top_hit,
        loc_background=bg,
        loc_enrichment=enrich if np.isfinite(enrich) else float("nan"),
        loc_passed=loc_passed,
        n_windows=n,
        top_k=k,
    )


def _align_bag(model: AttentionMILModel, bag: np.ndarray, centers: np.ndarray):
    keep = model.keep_indices(len(bag), training=False)
    return np.asarray(bag, dtype=np.float32)[keep], np.asarray(centers, dtype=float)[keep]


def extract_mil_bags(dataset, region_ids, catalog, groups, args) -> pd.DataFrame:
    regions = dataset.load_regions(list(region_ids), normalize=True)
    feat = HandcraftedAttentionMILFeaturizer(
        window_size_um=args.window_size,
        step_um=args.step,
        feature_groups=tuple(groups),
        cell_type_col=catalog.cell_type_col,
        min_cells_per_window=args.min_cells,
        use_tissue_mask=not args.no_tissue_mask,
    ).fit(regions)
    table = feat.transform(regions)
    table.index = table.index.astype(str)
    return table


def run_mil_explanations(dataset, catalog, tasks, bags: pd.DataFrame, args) -> pd.DataFrame:
    explainers = tuple(args.mil_explainers)
    unknown = [name for name in explainers if name not in EXPLAINERS]
    if unknown:
        raise ValueError(f"Unknown --mil-explainers {unknown}; expected {EXPLAINERS}")
    vcfg = dataset.validation_config
    patient_col = vcfg.get("patient_col", "patient_id")
    n_folds = vcfg.get("n_folds", 5)
    rng = np.random.default_rng(0)
    rows = []
    for task in tasks:
        spec = spec_for_task(catalog, task)
        kind = task_kind(task)
        meta = dataset.get_task_metadata(task)
        meta = meta[meta["region_id"].astype(str).isin(bags.index)].copy()
        y_all = dataset.build_target(meta["region_id"].astype(str).tolist(), task)
        print(f"=== mil {task}  kind={kind}  n={len(meta)} ===", flush=True)
        for seed in args.seeds:
            folds = safe_patient_kfold(
                meta, n_folds, patient_col, stratify_column(dataset.get_task_config(task)), seed
            )
            if folds is None:
                continue
            for fold_i, (train_ids, val_ids) in enumerate(folds):
                train_ids = [str(i) for i in train_ids if str(i) in bags.index]
                val_ids = [str(i) for i in val_ids if str(i) in bags.index]
                y_tr = y_all.reindex(train_ids).dropna()
                y_va = y_all.reindex(val_ids).dropna()
                if y_tr.nunique() < 2 or y_va.nunique() < 2:
                    continue
                model = AttentionMILModel(
                    task_type="binary",
                    seed=seed,
                    device=args.device,
                    max_instances=args.max_instances,
                ).fit(bags.loc[y_tr.index], y_tr)
                pred = model.predict(bags.loc[y_va.index])
                classes = list(model.classes_)
                pos_i = classes.index(1) if 1 in classes else -1
                auc = _safe_auc(y_va.to_numpy(), np.asarray(pred)[:, pos_i])
                auc_thr = CONTROL_AUC if kind == "control" else SPATIAL_AUC
                auc_passed = bool(np.isfinite(auc) and auc >= auc_thr)
                print(
                    f"  seed{seed}/fold{fold_i}  auc={auc:.3f}  "
                    f"{'HAS_SIGNAL' if auc_passed else 'WEAK'}",
                    flush=True,
                )
                if not auc_passed:
                    for name in explainers:
                        rows.append(_fold_row(
                            task, kind, seed, fold_i, name, y_tr, y_va, auc, auc_passed,
                            loc_df=None, faith_df=None,
                        ))
                    continue
                loc_chunks = {name: [] for name in explainers}
                faith_chunks = {name: [] for name in explainers}
                for rid, y_i in y_va.items():
                    bag, centers = _align_bag(
                        model,
                        bags.loc[str(rid), "bag"],
                        bags.loc[str(rid), "instance_centers"],
                    )
                    if len(bag) == 0:
                        continue
                    evidence = evidence_for_bag(
                        dataset.load_region(str(rid), normalize=True),
                        catalog,
                        spec,
                        centers,
                        args.window_size,
                    )
                    rand = random_scores(bag, rng)
                    for name in explainers:
                        scores = explain_bag(
                            model, bag, name, rng=rng, ig_steps=args.ig_steps
                        )
                        loc_chunks[name].append(
                            localization_metrics(scores, evidence["pos"], evidence["neg"])
                        )
                        faith_chunks[name].append(
                            faithfulness_metrics(
                                model, bag, scores, rand, step_frac=args.morf_frac
                            )
                        )
                for name in explainers:
                    loc = loc_chunks[name]
                    faith = faith_chunks[name]
                    loc_df = pd.DataFrame(loc) if loc else None
                    faith_df = pd.DataFrame(faith) if faith else None
                    rows.append(_fold_row(
                        task, kind, seed, fold_i, name, y_tr, y_va, auc, auc_passed,
                        loc_df=loc_df, faith_df=faith_df,
                    ))
    return pd.DataFrame(rows)


def _paired_morf_p(sub: pd.DataFrame) -> float:
    """One-sided paired test: explainer MORF AUPC < random MORF AUPC."""
    if "aupc_morf" not in sub.columns or "aupc_random" not in sub.columns:
        return float("nan")
    morf = pd.to_numeric(sub["aupc_morf"], errors="coerce")
    rand = pd.to_numeric(sub["aupc_random"], errors="coerce")
    mask = morf.notna() & rand.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_rel
        result = ttest_rel(morf[mask], rand[mask], alternative="less")
        return float(result.pvalue)
    except TypeError:
        from scipy.stats import ttest_rel
        stat = ttest_rel(morf[mask], rand[mask])
        p = float(stat.pvalue) / 2.0
        return p if float(stat.statistic) < 0 else 1.0 - p
    except Exception:
        return float("nan")


def _mean_or_nan(frame: pd.DataFrame | None, col: str) -> float:
    if frame is None or col not in frame.columns or frame.empty:
        return float("nan")
    return float(frame[col].mean())


def _fold_row(task, kind, seed, fold_i, name, y_tr, y_va, auc, auc_passed, loc_df, faith_df):
    loc_passed = bool(loc_df is not None and loc_df["loc_passed"].mean() >= 0.5)
    faith_passed = bool(faith_df is not None and faith_df["faith_passed"].mean() >= 0.5)
    return dict(
        task=task,
        kind=kind,
        seed=seed,
        fold=fold_i,
        explainer=name,
        n_train=len(y_tr),
        n_val=len(y_va),
        n_explained=0 if loc_df is None else len(loc_df),
        auc=auc,
        auc_passed=auc_passed,
        loc_auprc=_mean_or_nan(loc_df, "loc_auprc"),
        loc_auroc=_mean_or_nan(loc_df, "loc_auroc"),
        loc_auprc_neg=_mean_or_nan(loc_df, "loc_auprc_neg"),
        loc_auprc2=_mean_or_nan(loc_df, "loc_auprc2"),
        loc_enrichment=_mean_or_nan(loc_df, "loc_enrichment"),
        loc_passed=loc_passed,
        aupc_morf=_mean_or_nan(faith_df, "aupc_morf"),
        aupc_lerf=_mean_or_nan(faith_df, "aupc_lerf"),
        aupc_insert=_mean_or_nan(faith_df, "aupc_insert"),
        aupc_random=_mean_or_nan(faith_df, "aupc_random"),
        delta_lerf_morf=_mean_or_nan(faith_df, "delta_lerf_morf"),
        faith_passed=faith_passed,
        fold_passed=bool(auc_passed and loc_passed and faith_passed),
    )


def summarize_mil(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return fold_df
    has_dataset = "dataset" in fold_df.columns
    groups = fold_df.groupby("dataset", sort=False) if has_dataset else [(None, fold_df)]
    rows = []
    for dataset, dsub in groups:
        for explainer, esub in dsub.groupby("explainer"):
            ctrl = esub["kind"].eq("control")
            auc_ctrl = float(esub.loc[ctrl, "auc_passed"].mean()) if ctrl.any() else float("nan")
            loc_ctrl = float(esub.loc[ctrl, "loc_passed"].mean()) if ctrl.any() else float("nan")
            control_frac = float(esub.loc[ctrl, "fold_passed"].mean()) if ctrl.any() else float("nan")
            for task, sub in esub.groupby("task"):
                kind = sub["kind"].iloc[0]
                loc_frac = float(sub["loc_passed"].mean())
                faith_frac = float(sub["faith_passed"].mean())
                auc_frac = float(sub["auc_passed"].mean())
                if kind == "control":
                    loc_verdict = "pass_control" if loc_frac >= 0.5 else "fail_control"
                    faith_verdict = "pass_control" if faith_frac >= 0.5 else "fail_control"
                    verdict = "pass_control" if float(sub["fold_passed"].mean()) >= 0.5 else "fail_control"
                elif not np.isfinite(auc_ctrl) or auc_ctrl < 0.5:
                    loc_verdict = faith_verdict = verdict = "fail_control_block"
                else:
                    loc_verdict = "pass_spatial" if loc_frac >= 0.5 else "abundance_only_or_fail"
                    faith_verdict = "pass_spatial" if faith_frac >= 0.5 else "unfaithful_or_fail"
                    if loc_frac >= 0.5 and faith_frac >= 0.5:
                        verdict = "pass_spatial"
                    elif faith_frac >= 0.5:
                        verdict = "faithful_but_wrong_place"
                    elif loc_frac >= 0.5:
                        verdict = "looks_biological_unfaithful"
                    else:
                        verdict = "abundance_only_or_fail"
                rec = dict(
                    task=task,
                    kind=kind,
                    explainer=explainer,
                    n_folds=len(sub),
                    mean_auc=float(sub["auc"].mean()),
                    frac_auc_passed=auc_frac,
                    mean_loc_auprc=float(sub["loc_auprc"].mean()),
                    mean_loc_auprc2=float(sub["loc_auprc2"].mean()),
                    mean_loc_enrichment=float(sub["loc_enrichment"].mean()),
                    frac_loc_passed=loc_frac,
                    mean_aupc_morf=float(sub["aupc_morf"].mean()),
                    mean_aupc_lerf=float(sub["aupc_lerf"].mean()),
                    mean_delta_lerf_morf=float(sub["delta_lerf_morf"].mean()),
                    frac_faith_passed=faith_frac,
                    control_auc_frac=auc_ctrl,
                    control_loc_frac=loc_ctrl,
                    control_frac_passed=control_frac,
                    loc_verdict=loc_verdict,
                    faith_verdict=faith_verdict,
                    verdict=verdict,
                    p_morf_lt_random=_paired_morf_p(sub),
                )
                if dataset is not None:
                    rec["dataset"] = dataset
                if "selected" in sub.columns:
                    rec["selected"] = bool(sub["selected"].iloc[0])
                rows.append(rec)
    return pd.DataFrame(rows)
