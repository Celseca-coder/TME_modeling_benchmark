"""Shared allowlist for the 17 tasks used in the selected-task comparison."""
from __future__ import annotations


SELECTED_BENCHMARK_TASKS = {
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
}


def should_skip_benchmark_task(dataset: str, task: str, scheme: str, metric: str) -> bool:
    """Return True for tasks outside the selected-task comparison.

    Historically this predicate was reversed: the 17 selected tasks were named
    ``EXCLUDED_BENCHMARK_TASKS`` and therefore skipped by graph-model runners.
    Keep the public predicate, but make its behaviour match its callers: skip
    everything *except* the allowlisted tasks.
    """
    return not is_selected_benchmark_task(dataset, task, scheme, metric)


def is_selected_benchmark_task(dataset: str, task: str, scheme: str, metric: str) -> bool:
    return (dataset, task, scheme, metric) in SELECTED_BENCHMARK_TASKS
