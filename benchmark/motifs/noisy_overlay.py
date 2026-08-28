"""Attach noisy motif label columns as binary tasks without a motif catalog."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.data.dataset import TMEDataset


def attach_noisy_labels(dataset: TMEDataset, labels: pd.DataFrame | str | Path) -> list[str]:
    if not isinstance(labels, pd.DataFrame):
        labels = pd.read_csv(labels)
    extra = labels.copy()
    extra["region_id"] = extra["region_id"].astype(str)
    meta = dataset.get_metadata().copy()
    meta["region_id"] = meta["region_id"].astype(str)
    overlap = [c for c in extra.columns if c in meta.columns and c != "region_id"]
    extra = extra.drop(columns=overlap)
    dataset._metadata = meta.merge(extra, on="region_id", how="left")

    existing = {t["id"] for t in dataset.config.get("tasks", [])}
    added = []
    for col in extra.columns:
        if not col.endswith("_label_v2"):
            continue
        motif = col[: -len("_label_v2")]
        task_id = f"motif_{motif}"
        if task_id in existing:
            continue
        y = pd.to_numeric(dataset._metadata[col], errors="coerce")
        if y.notna().sum() < 16 or y.dropna().nunique() < 2:
            continue
        dataset.config.setdefault("tasks", []).append({
            "id": task_id,
            "type": "binary_classification",
            "label_col": col,
            "positive_class": 1.0,
        })
        added.append(task_id)
    return added
