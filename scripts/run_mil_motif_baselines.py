#!/usr/bin/env python
"""Linear-mean and attention MIL on motif v2 labels.

Default panel = 10 selected motifs + matched tumor_high/cd8_high controls,
using ``results/pseudo_labels/{dataset}_v2.csv``.  Pass ``--noisy`` to switch
the sidecar to PseudoNoisyDataset (still ``*_label_v2`` columns).

    python scripts/run_mil_motif_baselines.py --panel --method linear_mean \\
        --feature-groups composition --data-root "$DATA_ROOT"

    python scripts/run_mil_motif_baselines.py --panel --method attention \\
        --feature-groups composition mixing --data-root "$DATA_ROOT" --device cuda:0

    python scripts/run_mil_motif_baselines.py --panel --all-combos --method linear_mean \\
        --data-root "$DATA_ROOT"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.attention_mil import HandcraftedAttentionMILFeaturizer  # noqa: E402
from benchmark.features.local_window import MOTIF_MIL_COMBOS, MOTIF_MIL_GROUPS  # noqa: E402
from benchmark.features.patch_feats import PatchBasedFeaturizer  # noqa: E402
from benchmark.models.attention_mil import AttentionMILModel  # noqa: E402
from benchmark.models.linear import LinearClassifier  # noqa: E402
from benchmark.motifs.overlay import attach_pseudo_labels  # noqa: E402
from benchmark.motifs.panel import (  # noqa: E402
    jobs_by_dataset,
    labels_path_for,
    load_selected_panel,
    motif_ids_from_tasks,
    run_tasks_for,
)
from benchmark.motifs.recovery import SELECTED_EXPLAIN_TASKS  # noqa: E402
from benchmark.motifs.spec import load_motif_catalog  # noqa: E402
from benchmark.utils.registry import load_dataset  # noqa: E402
from benchmark.validation import PRIMARY_METRIC, cross_validate_precomputed, summarize_folds  # noqa: E402

NOISY_LABEL_TEMPLATE = (
    "/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset/{dataset}_v2_noisy.csv"
)


def _combo_name(groups: tuple[str, ...]) -> str:
    return "+".join(groups)


def _format_path(template: str | None, dataset: str) -> str | None:
    if template is None:
        return None
    return template.format(dataset=dataset)


def _make_featurizer(method: str, groups: tuple[str, ...], cell_type_col: str, args):
    if method == "linear_mean":
        return lambda: PatchBasedFeaturizer(
            window_size_um=args.window_size,
            step_um=args.step,
            feature_groups=groups,
            cell_type_col=cell_type_col,
            aggregations=("mean",),
            min_cells_per_window=args.min_cells,
            use_tissue_mask=not args.no_tissue_mask,
        )
    return lambda: HandcraftedAttentionMILFeaturizer(
        window_size_um=args.window_size,
        step_um=args.step,
        feature_groups=groups,
        cell_type_col=cell_type_col,
        min_cells_per_window=args.min_cells,
        use_tissue_mask=not args.no_tissue_mask,
    )


def _make_model(method: str, args):
    if method == "attention":
        def factory(task_cfg, seed):
            task_type = task_cfg["type"]
            if task_type in {"binary", "binary_classification"}:
                task_type = "binary"
            return AttentionMILModel(
                task_type=task_type,
                seed=seed,
                device=args.device,
            )
        return factory
    return lambda task_cfg, seed: LinearClassifier(seed=seed)


def _cache_path(args, dataset_name: str, method: str, groups: tuple[str, ...]) -> Path:
    root = Path(args.cache_dir or (_CODE / "results" / "mil_motif_baselines" / "cache"))
    key = (
        f"{dataset_name}_{method}_{_combo_name(groups)}"
        f"_w{args.window_size:g}_s{args.step:g}_c{args.min_cells}"
    )
    return root / f"{key}.pkl"


def _extract_once(dataset, region_ids: list[str], factory, cache: Path, *, normalize: bool):
    if cache.exists():
        print(f"Loading cached windows {cache}", flush=True)
        table = pd.read_pickle(cache)
        table.index = table.index.astype(str)
        return table
    ids = [str(i) for i in region_ids]
    print(f"Extracting {len(ids)} regions -> {cache}", flush=True)
    regions = dataset.load_regions(ids, normalize=normalize)
    feat = factory().fit(regions)
    rows = []
    for i, region in enumerate(regions):
        if i % 25 == 0:
            print(f"  {i}/{len(regions)}  {region.region_id}", flush=True)
        rows.append(feat.extract_region(region))
    table = pd.DataFrame(rows, index=[r.region_id for r in regions])
    table.index = table.index.astype(str)
    table.index.name = "region_id"
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(cache)
    dataset.clear_region_cache()
    print(f"Wrote {cache}  n={len(table)}", flush=True)
    return table


def _attach_labels(dataset, dataset_name: str, run_tasks: list[str], args) -> Path:
    catalog = load_motif_catalog(args.catalog, dataset=dataset_name)
    default = (
        NOISY_LABEL_TEMPLATE.format(dataset=dataset_name)
        if args.noisy else str(labels_path_for(dataset_name, args.labels_dir))
    )
    labels_path = Path(_format_path(args.labels, dataset_name) or default)
    attach_pseudo_labels(
        dataset,
        labels_path,
        catalog,
        label_version=args.label_version,
        motif_ids=motif_ids_from_tasks(run_tasks),
    )
    return labels_path


def _run_one(dataset_name: str, selected: list[str], method: str,
             groups: tuple[str, ...], args) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, data_root=args.data_root)
    run_tasks = args.tasks or run_tasks_for(
        selected, include_matched_controls=not args.no_matched_controls
    )
    labels_path = _attach_labels(dataset, dataset_name, run_tasks, args)
    cell_type_col = load_motif_catalog(args.catalog, dataset=dataset_name).cell_type_col
    factory = _make_featurizer(method, groups, cell_type_col, args)
    make_model = _make_model(method, args)
    normalize = "expression" in groups
    combo = _combo_name(groups)
    print(
        f"=== {dataset_name}  method={method}  groups={combo}  "
        f"labels={labels_path}  tasks={run_tasks} ===",
        flush=True,
    )
    labeled = []
    for task in run_tasks:
        if task in dataset.task_ids:
            labeled.extend(dataset.get_task_metadata(task)["region_id"].astype(str).tolist())
    bags = _extract_once(
        dataset, sorted(set(labeled)), factory,
        _cache_path(args, dataset_name, method, groups),
        normalize=normalize,
    )
    rows = []
    for task in run_tasks:
        if task not in dataset.task_ids:
            print(f"  skip {task}: not registered", flush=True)
            continue
        metric = PRIMARY_METRIC[dataset.get_task_config(task)["type"]]
        folds = cross_validate_precomputed(
            dataset, task, bags, make_model, seeds=args.seeds,
        )
        if not folds:
            print(f"  skip {task}: no labelled folds", flush=True)
            continue
        mean, sd = summarize_folds(folds, metric)
        rows.append(dict(
            dataset=dataset_name, task=task, method=method,
            feature_groups=combo, scheme="cv", metric=metric,
            mean=mean, sd=sd, n=len(folds),
            labels=str(labels_path),
        ))
        print(f"  {task:32s} {metric:10s} {mean:.4f} +/- {sd:.4f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hnc_wu2022")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--labels", default=None,
                    help="Label CSV or template with {dataset}.")
    ap.add_argument("--labels-dir", default=None,
                    help="Directory of clean {dataset}_v2.csv files.")
    ap.add_argument("--label-version", default="v2", choices=["v1", "v2"])
    ap.add_argument(
        "--noisy",
        action="store_true",
        help="Use PseudoNoisyDataset/{dataset}_v2_noisy.csv instead of clean v2.",
    )
    ap.add_argument(
        "--panel",
        nargs="?",
        const=str(_CODE / "results" / "pseudo_labels_all" / "selected_catalog.csv"),
        default=None,
    )
    ap.add_argument("--no-matched-controls", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--method", required=True, choices=["linear_mean", "attention"])
    ap.add_argument(
        "--feature-groups",
        nargs="+",
        default=list(MOTIF_MIL_GROUPS[:1]),
        choices=list(MOTIF_MIL_GROUPS),
    )
    ap.add_argument(
        "--all-combos",
        action="store_true",
        help="Run the 3 singles plus 3 pairwise combinations.",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--window-size", type=float, default=100.0)
    ap.add_argument("--step", type=float, default=50.0)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--no-tissue-mask", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Pickle cache for once-per-dataset window features "
             "(default: results/mil_motif_baselines/cache).",
    )
    args = ap.parse_args()

    combos = list(MOTIF_MIL_COMBOS) if args.all_combos else [tuple(args.feature_groups)]
    if args.panel:
        jobs = jobs_by_dataset(load_selected_panel(args.panel))
    else:
        jobs = {args.dataset: list(args.tasks or SELECTED_EXPLAIN_TASKS)}

    frames = []
    for groups in combos:
        for dataset_name, selected in jobs.items():
            frames.append(_run_one(dataset_name, selected, args.method, groups, args))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result["score"] = result.apply(
            lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1
        )
    tag = args.method + ("_all" if args.all_combos else "_" + _combo_name(combos[0]))
    if args.noisy:
        tag += "_noisy"
    out = Path(args.output or (_CODE / "results" / "mil_motif_baselines" / f"{tag}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(result)} rows)", flush=True)


if __name__ == "__main__":
    main()
