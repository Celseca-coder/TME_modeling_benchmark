"""Dataset registry: discover and load all benchmark dataset configs."""
from __future__ import annotations

from pathlib import Path

from benchmark.data.dataset import TMEDataset

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "datasets"

# Default data root: TME_benchmark/ (parent of code/)
_DEFAULT_DATA_ROOT = _CONFIGS_DIR.parent.parent.parent


def list_datasets() -> list[str]:
    """Return sorted list of available dataset config names (without .yaml)."""
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))


def load_dataset(name: str, data_root: str | Path | None = None) -> TMEDataset:
    """Load a TMEDataset by its config filename stem.

    Parameters
    ----------
    name : str
        Config stem, e.g. ``'bc_metabric_ali2020'`` or ``'nsclc_aung2025'``.
    data_root : path
        Root folder containing dataset directories (e.g. BC-METABRIC_Ali2020/).
        Defaults to the parent of the ``code/`` folder (i.e. TME_benchmark/).
    """
    yaml_path = _CONFIGS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        available = list_datasets()
        raise FileNotFoundError(
            f"Dataset config '{name}.yaml' not found.\nAvailable: {available}"
        )
    if data_root is None:
        data_root = _DEFAULT_DATA_ROOT
    return TMEDataset.from_yaml(yaml_path, data_root)


def load_all_datasets(data_root: str | Path | None = None) -> dict[str, TMEDataset]:
    """Load all available datasets, returning a dict keyed by config name."""
    return {name: load_dataset(name, data_root) for name in list_datasets()}
