#!/usr/bin/env python
"""Run all 15 pairwise feature combinations on the 17 selected benchmark tasks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.utils.registry import load_dataset
from benchmark.validation import PRIMARY_METRIC
from run_pairwise_feature_combinations import COMBINATIONS, parse_combinations, run


# dataset, task, validation scheme, expected primary metric
SELECTED_RUNS = (
    ("bc_jackson2020", "clinical_type", "Basel_to_Zurich", "balanced_acc"),
    ("bc_jackson2020", "clinical_type", "cv", "balanced_acc"),
    ("bc_jackson2020", "response", "cv", "auc_roc"),
    ("bc_metabric_ali2020", "DSS", "cv", "c_index"),
    ("bc_metabric_ali2020", "ER_status", "cv", "auc_roc"),
    ("crc_schurch2020", "CLR_DII", "cv", "auc_roc"),
    ("crc_schurch2020", "OS", "cv", "c_index"),
    ("hnc_wu2022", "OS", "cv", "c_index"),
    ("hnc_wu2022", "hpv_status", "cv", "auc_roc"),
    ("hnc_wu2022", "primary_outcome", "UPMC_to_DFCI", "auc_roc"),
    ("hnc_wu2022", "primary_outcome", "cv", "auc_roc"),
    ("luad_sorin2023", "OS", "cv", "c_index"),
    ("nsclc_aung2025", "immunotherapy_response", "Yale_to_UQ", "auc_roc"),
    ("nsclc_aung2025", "immunotherapy_response", "Yale_to_YaleExt", "auc_roc"),
    ("nsclc_gnn_hoebel2026", "OS", "cv", "c_index"),
    ("nsclc_gnn_hoebel2026", "stage_binary", "cv", "auc_roc"),
    ("tnbc_wang2023", "pCR_all", "cv", "auc_roc"),
)


def validate_selection(data_root=None):
    """Fail early if a selected task, scheme, or primary metric no longer matches YAML."""
    datasets = {}
    for dataset_name, task, scheme, expected_metric in SELECTED_RUNS:
        if dataset_name not in datasets:
            datasets[dataset_name] = load_dataset(dataset_name, data_root=data_root)
        ds = datasets[dataset_name]
        if task not in ds.task_ids:
            raise ValueError(f"Unknown selected task: {dataset_name}/{task}")
        actual_metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
        if actual_metric != expected_metric:
            raise ValueError(
                f"Metric mismatch for {dataset_name}/{task}: "
                f"selected={expected_metric}, configured={actual_metric}"
            )
        if scheme != "cv":
            generalization = {
                test["name"]: test for test in ds.validation_config.get("generalization_tests", [])
            }
            if scheme not in generalization or task not in generalization[scheme].get("tasks", ds.task_ids):
                raise ValueError(f"Unknown selected scheme/task: {dataset_name}/{task}/{scheme}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combinations", nargs="*", default=None, choices=COMBINATIONS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--point-pattern-metrics", nargs="+", default=["K", "L"],
                        choices=["K", "L", "pcf", "variogram"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output", default=str(_CODE / "results" / "pairwise_selected_tasks_benchmark.csv")
    )
    args = parser.parse_args()

    validate_selection(args.data_root)
    selected_triples = {(dataset, task, scheme) for dataset, task, scheme, _ in SELECTED_RUNS}
    dataset_names = list(dict.fromkeys(row[0] for row in SELECTED_RUNS))
    frame = run(
        dataset_names,
        parse_combinations(args.combinations),
        args.seeds,
        data_root=args.data_root,
        point_metrics=args.point_pattern_metrics,
        selected_runs=selected_triples,
    )
    frame["score"] = frame.apply(lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {output} ({len(frame)} rows; expected up to {len(SELECTED_RUNS) * len(parse_combinations(args.combinations))})")


if __name__ == "__main__":
    main()
