"""Selected motif × source-dataset panel from ``results/pseudo_labels_all``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent.parent
DEFAULT_PANEL = _CODE / "results" / "pseudo_labels_all" / "selected_catalog.csv"
DEFAULT_LABELS_DIR = _CODE / "results" / "pseudo_labels"
MATCHED_CONTROLS = ("tumor_high", "cd8_high")


def load_selected_panel(path: str | Path | None = None) -> pd.DataFrame:
    """Return eligible motif rows from the combined selected catalog."""
    panel_path = Path(path or DEFAULT_PANEL)
    if not panel_path.is_file():
        raise FileNotFoundError(f"Selected motif catalog not found: {panel_path}")
    table = pd.read_csv(panel_path)
    if "eligible" in table.columns:
        table = table.loc[table["eligible"].astype(str).str.lower().isin(["true", "1"])].copy()
    if table.empty:
        raise ValueError(f"No eligible motifs in {panel_path}")
    return table.reset_index(drop=True)


def jobs_by_dataset(panel: pd.DataFrame) -> dict[str, list[str]]:
    """Map source dataset → selected ``motif_*`` task ids."""
    jobs: dict[str, list[str]] = {}
    for row in panel.itertuples(index=False):
        dataset = str(row.source_dataset)
        jobs.setdefault(dataset, []).append(f"motif_{row.motif_id}")
    return jobs


def run_tasks_for(
    selected: list[str],
    include_matched_controls: bool = True,
) -> list[str]:
    """Selected tasks plus same-dataset ``tumor_high`` / ``cd8_high`` controls."""
    tasks = list(selected)
    if include_matched_controls:
        for motif_id in MATCHED_CONTROLS:
            task = f"motif_{motif_id}"
            if task not in tasks:
                tasks.append(task)
    return tasks


def motif_ids_from_tasks(tasks: list[str]) -> list[str]:
    return [str(task).removeprefix("motif_") for task in tasks]


def labels_path_for(dataset: str, labels_dir: str | Path | None = None) -> Path:
    root = Path(labels_dir or DEFAULT_LABELS_DIR)
    return root / f"{dataset}_v2.csv"
