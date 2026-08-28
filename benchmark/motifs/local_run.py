"""Shared wiring for local Eva / KRONOS / UTAG runs on motif pseudo-labels.

Imaging still comes from TME_benchmark_data. Targets come from
PseudoNoisyDataset (wide ``*_label_v2`` tables) or CombinedDataset
(long ``selected_motif_labels.csv``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.motifs.noisy_overlay import attach_noisy_labels

DEFAULT_DATA_ROOT = "/autofs/bal14/zqwu/CellularTables/TME_benchmark_data"
DEFAULT_LABEL_ROOT = "/autofs/nas8/tywang/tjzou/PseudoNoisyDataset"

PSEUDO_DATASETS = (
    "bc_jackson2020",
    "hnc_wu2022",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)

# Selected 10 motifs plus matched controls on every cohort.
TASKS_BY_DATASET = {
    "bc_jackson2020": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_t_tumor_mixing", "motif_cd8_tumor_contact",
        "motif_macrophage_tumor_niche", "motif_apc_t_contact",
    ),
    "hnc_wu2022": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_cd8_clustering", "motif_immune_exclusion",
    ),
    "bc_metabric_ali2020": (
        "motif_tumor_high", "motif_cd8_high",
        "motif_tumor_stroma_mixing", "motif_interface_immune",
    ),
    "tnbc_wang2023": (
        "motif_tumor_high", "motif_cd8_high",
    ),
}

LABEL_FILES = {
    "bc_jackson2020": "per_dataset/bc_jackson2020_v2_noisy.csv",
    "hnc_wu2022": "per_dataset/hnc_wu2022_v2_noisy.csv",
    "bc_metabric_ali2020": "per_dataset/bc_metabric_ali2020_v2_noisy.csv",
    "tnbc_wang2023": "per_dataset/tnbc_wang2023_v2_noisy.csv",
}


def data_roots(args) -> list[str]:
    if getattr(args, "data_roots", None):
        return [str(Path(root).expanduser()) for root in args.data_roots]
    if getattr(args, "data_root", None):
        return [str(Path(args.data_root).expanduser())]
    return [DEFAULT_DATA_ROOT]


def dataset_names(requested, target: str = "pseudo") -> list[str]:
    if requested:
        return list(requested)
    if target == "pseudo":
        return list(PSEUDO_DATASETS)
    from benchmark.utils.registry import list_datasets
    return list(list_datasets())


def resolve_label_path(dataset: str, label_root: str | Path) -> Path:
    root = Path(label_root)
    candidates = [
        root / LABEL_FILES.get(dataset, f"per_dataset/{dataset}_v2_noisy.csv"),
        root / "per_dataset" / f"{dataset}_v2.csv",
        root / f"{dataset}_v2_noisy.csv",
        root / f"{dataset}_v2.csv",
        root / "selected_motif_labels.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No motif label table for {dataset} under {root}. "
        "Expected per_dataset/*_v2_noisy.csv or selected_motif_labels.csv."
    )


def _wide_from_long(labels: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = labels.copy()
    if "source_dataset" in frame.columns:
        frame = frame.loc[frame["source_dataset"].astype(str) == dataset]
    elif "dataset" in frame.columns:
        frame = frame.loc[frame["dataset"].astype(str) == dataset]
    value_col = "pseudo_label" if "pseudo_label" in frame.columns else "label"
    wide = frame.pivot_table(
        index="region_id", columns="motif_id", values=value_col, aggfunc="first",
    )
    wide.columns = [f"{name}_label_v2" for name in wide.columns]
    return wide.reset_index()


def attach_pseudo_tasks(dataset, dataset_name: str, label_root: str | Path,
                        tasks: list[str] | None = None) -> list[str]:
    path = resolve_label_path(dataset_name, label_root)
    labels = pd.read_csv(path)
    if "motif_id" in labels.columns and (
        "pseudo_label" in labels.columns or "label" in labels.columns
    ):
        labels = _wide_from_long(labels, dataset_name)
    attach_noisy_labels(dataset, labels)
    wanted = list(tasks or TASKS_BY_DATASET.get(dataset_name, ()))
    return [task for task in wanted if task in dataset.task_ids]
