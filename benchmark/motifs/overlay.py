"""Attach motif pseudo-label columns and tasks to a loaded TMEDataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.data.dataset import TMEDataset
from benchmark.motifs.spec import MotifCatalog


def _merge_labels(metadata: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    extra = labels.copy()
    if extra.index.name == "region_id" or "region_id" not in extra.columns:
        extra = extra.reset_index()
        if "index" in extra.columns and "region_id" not in extra.columns:
            extra = extra.rename(columns={"index": "region_id"})
    overlap = [c for c in extra.columns if c in metadata.columns and c != "region_id"]
    extra = extra.drop(columns=overlap)
    return metadata.merge(extra, on="region_id", how="left")


def _label_column(spec, label_version: str) -> str:
    if label_version in ("v1", "label", ""):
        return spec.label_col
    if label_version in ("v2", "label_v2"):
        return f"{spec.id}_label_v2"
    raise ValueError(f"Unknown label_version {label_version!r}; expected 'v1' or 'v2'")


def _iter_specs(catalog: MotifCatalog, motif_ids: list[str] | None):
    if motif_ids is None:
        return catalog.motifs
    wanted = set(motif_ids)
    specs = tuple(spec for spec in catalog.motifs if spec.id in wanted)
    missing = wanted - {spec.id for spec in specs}
    if missing:
        raise KeyError(
            f"Motifs not in catalog for {catalog.dataset}: {sorted(missing)}"
        )
    return specs


def missing_label_columns(
    labels: pd.DataFrame,
    catalog: MotifCatalog,
    label_version: str = "v1",
    motif_ids: list[str] | None = None,
) -> list[str]:
    cols = set(labels.columns)
    return [
        _label_column(spec, label_version)
        for spec in _iter_specs(catalog, motif_ids)
        if _label_column(spec, label_version) not in cols
    ]


def attach_pseudo_labels(
    dataset: TMEDataset,
    labels: pd.DataFrame | str | Path,
    catalog: MotifCatalog,
    label_version: str = "v1",
    motif_ids: list[str] | None = None,
) -> TMEDataset:
    """Merge sidecar labels into in-memory metadata and register motif tasks.

    Does not rewrite the dataset YAML. Clinical tasks stay in place; motif tasks
    are appended as ``motif_<id>`` if they are not already present.

    ``motif_ids`` limits which catalog recipes are required and registered.
    """
    if not isinstance(labels, pd.DataFrame):
        labels = pd.read_csv(labels)
    missing = missing_label_columns(
        labels, catalog, label_version=label_version, motif_ids=motif_ids
    )
    if missing:
        raise ValueError(
            "Pseudo-label table is missing columns for the current motif catalog: "
            f"{missing}. Re-run scripts/generate_pseudo_labels.py after updating "
            "configs/motifs/, or scripts/refine_pseudo_labels.py for v2 labels."
        )
    meta = dataset.get_metadata().copy()
    dataset._metadata = _merge_labels(meta, labels)

    existing = {t["id"] for t in dataset.config.get("tasks", [])}
    extra = []
    for spec in _iter_specs(catalog, motif_ids):
        if spec.task_id in existing:
            continue
        cfg = spec.task_config()
        cfg["label_col"] = _label_column(spec, label_version)
        extra.append(cfg)
    dataset.config["tasks"] = list(dataset.config.get("tasks", [])) + extra
    return dataset


def motif_task_ids(dataset: TMEDataset) -> list[str]:
    return [t["id"] for t in dataset.config.get("tasks", []) if str(t["id"]).startswith("motif_")]
