"""Motif catalog: a frozen, interpretable spatial recipe loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "motifs"

LABEL_RULES = ("median", "tertile_extremes")
MOTIF_KINDS = (
    "composition",
    "neighbor_fraction",
    "immune_exclusion",
    "tls_like",
    "interface",
)


@dataclass(frozen=True)
class MotifSpec:
    """One motif → one continuous score → one (or more) pseudo-label tasks.

    ``kind`` selects the detector in :mod:`benchmark.motifs.detect`. Cell-set
    names refer to keys in :class:`MotifCatalog.cell_sets`, not raw type strings.
    """

    id: str
    kind: str
    spatial: bool = True
    label_rule: str = "median"
    cell_set: str | None = None
    source_set: str | None = None
    neighbor_set: str | None = None
    immune_set: str | None = None
    tumor_set: str | None = None
    required_sets: tuple[str, ...] = ()
    min_frac: dict[str, float] = field(default_factory=dict)
    radius_um: float = 50.0
    min_cluster_size: int = 5
    residual_on: tuple[str, ...] = ()
    min_source_cells: int = 5

    def __post_init__(self) -> None:
        if self.kind not in MOTIF_KINDS:
            raise ValueError(f"Unknown motif kind {self.kind!r}; expected {MOTIF_KINDS}")
        if self.label_rule not in LABEL_RULES:
            raise ValueError(
                f"Unknown label_rule {self.label_rule!r}; expected {LABEL_RULES}"
            )

    @property
    def task_id(self) -> str:
        return f"motif_{self.id}"

    @property
    def score_col(self) -> str:
        return f"{self.id}_score"

    @property
    def label_col(self) -> str:
        return f"{self.id}_label"

    def task_config(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "type": "binary_classification",
            "label_col": self.label_col,
            "positive_class": 1,
        }


@dataclass(frozen=True)
class MotifCatalog:
    dataset: str
    cell_type_col: str
    cell_sets: dict[str, tuple[str, ...]]
    motifs: tuple[MotifSpec, ...]
    cv_filter: str | None = None
    atlas_n: int = 20
    radius_um: float = 50.0

    def motif(self, motif_id: str) -> MotifSpec:
        for spec in self.motifs:
            if spec.id == motif_id:
                return spec
        raise KeyError(f"Motif {motif_id!r} not in catalog for {self.dataset}")

    def resolve_set(self, name: str) -> set[str]:
        if name not in self.cell_sets:
            raise KeyError(
                f"Unknown cell set {name!r}. Available: {sorted(self.cell_sets)}"
            )
        return set(self.cell_sets[name])

    def type_to_sets(self) -> dict[str, list[str]]:
        """Map a raw cell-type name to the catalog sets that contain it."""
        out: dict[str, list[str]] = {}
        for set_name, members in self.cell_sets.items():
            for member in members:
                out.setdefault(member, []).append(set_name)
        return out


def default_catalog_path(dataset: str) -> Path:
    return _CONFIGS_DIR / f"{dataset}.yaml"


def load_motif_catalog(path: str | Path | None = None, dataset: str | None = None) -> MotifCatalog:
    if path is None:
        if dataset is None:
            raise ValueError("Provide path or dataset")
        path = default_catalog_path(dataset)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Motif catalog not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return catalog_from_dict(raw)


def catalog_from_dict(raw: dict[str, Any]) -> MotifCatalog:
    cell_sets = {
        name: tuple(members)
        for name, members in (raw.get("cell_sets") or {}).items()
    }
    default_radius = float(raw.get("radius_um", 50.0))
    motifs = []
    for item in raw.get("motifs") or []:
        item = dict(item)
        item.setdefault("radius_um", default_radius)
        if "required_sets" in item:
            item["required_sets"] = tuple(item["required_sets"])
        if "residual_on" in item:
            item["residual_on"] = tuple(item["residual_on"])
        if "min_frac" in item and item["min_frac"] is None:
            item["min_frac"] = {}
        motifs.append(MotifSpec(**item))
    return MotifCatalog(
        dataset=raw["dataset"],
        cell_type_col=raw.get("cell_type_col", "cell_type"),
        cell_sets=cell_sets,
        motifs=tuple(motifs),
        cv_filter=raw.get("cv_filter"),
        atlas_n=int(raw.get("atlas_n", 20)),
        radius_um=default_radius,
    )
