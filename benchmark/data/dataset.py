"""Core dataset abstractions: RegionData and TMEDataset."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import yaml


# Default filename for the exported tissue/tumour polygons (see the
# tissue_area_estimation notebook); overridable via the ``polygons_file`` config key.
DEFAULT_POLYGONS_FILE = "tissue_polygons.geojson"


def load_region_polygons(path: str | Path) -> dict | None:
    """Read a ``tissue_polygons.geojson`` FeatureCollection into ``{kind: shapely geometry}``.

    Geometry is kept in the region's pixel coordinate space (matching ``coordinates``);
    ``kind`` is the Feature's ``properties.kind`` (``"tissue"`` / ``"tumour"``). Returns
    ``None`` if the file is missing or empty.
    """
    path = Path(path)
    if not path.exists():
        return None
    from shapely.geometry import shape  # lazy: only needed when polygons are present

    with open(path) as f:
        fc = json.load(f)
    polys = {}
    for feat in fc.get("features", []):
        kind = feat.get("properties", {}).get("kind", "tissue")
        polys[kind] = shape(feat["geometry"])
    return polys or None


@dataclass
class RegionData:
    """Single spatial region (TMA core / ROI / FOV).

    All three DataFrames share the same ``cell_id`` index column.

    ``microns_per_pixel`` is the physical size of one coordinate unit in microns
    (set from the dataset config; see :class:`TMEDataset`). For datasets whose
    coordinates are already stored in microns it is ``1.0``. It lets spatial
    operations such as :meth:`neighborhood` work in physical units (µm) regardless
    of whether the raw coordinates are pixels or microns.
    """
    region_id: str
    coordinates: pd.DataFrame   # index: cell_id, columns: x, y
    expression: pd.DataFrame    # index: cell_id, columns: <marker1>, ...
    cell_types: pd.DataFrame    # index: cell_id, columns: cell_type [, cell_class, ...]
    microns_per_pixel: float
    polygons: dict | None = field(default=None, repr=False, compare=False)

    # Accept "tumor" as an alias for the exported "tumour" key.
    _KIND_ALIASES = {"tumor": "tumour"}

    @property
    def n_cells(self) -> int:
        return len(self.coordinates)

    def polygon_area_mm2(self, kind: str = "tissue") -> float | None:
        """Physical area (mm^2) of the ``kind`` polygon, or None if absent.

        ``shapely``'s ``.area`` is in coordinate (pixel) units squared; we convert with
        ``microns_per_pixel`` the same way as :attr:`coordinates_um`.
        """
        if self.polygons is None:
            return None
        g = self.polygons.get(self._KIND_ALIASES.get(kind, kind))
        if g is None:
            return None
        return g.area * (self.microns_per_pixel ** 2) / 1e6

    @property
    def tissue_area_mm2(self) -> float | None:
        return self.polygon_area_mm2("tissue")

    @property
    def tumor_area_mm2(self) -> float | None:
        return self.polygon_area_mm2("tumour")

    def polygon_contains(self, xy, kind: str = "tissue"):
        """Boolean array: which of ``xy`` (pixel coords, Nx2) lie inside the ``kind`` polygon.

        Returns None if the polygon is absent. Defaults to all cells' coordinates when
        ``xy`` is None.
        """
        # Acquire the polygon
        if self.polygons is None:
            return None
        g = self.polygons.get(self._KIND_ALIASES.get(kind, kind))
        if g is None:
            return None

        import numpy as np
        import shapely
        if xy is None:
            xy = self.coordinates[["x", "y"]].to_numpy(float)
        xy = np.asarray(xy, float)
        return shapely.contains_xy(g, xy[:, 0], xy[:, 1])

    def subset(
        self,
        cell_ids: Iterable[int],
        region_id: str | None = None,
        polygons: dict | None | str = "inherit",
    ) -> "RegionData":
        """Return a new RegionData object with only the specified cell IDs.

        Pass ``region_id`` to label the subset explicitly; otherwise the parent
        region_id is suffixed with the (sorted) cell IDs.

        ``polygons`` controls the tissue/tumour geometry of the subset:
          * ``"inherit"`` (default) — keep the parent region's polygons (they describe the
            same physical region, so this is right for label-based subsets such as
            "all tumour cells");
          * ``None`` — drop the polygons;
          * a ``{kind: geometry}`` dict — redefine them (e.g. clipped geometry from a
            spatial subset; see :meth:`neighborhood`).
        """
        cell_ids = sorted(set(cell_ids))
        if region_id is None:
            region_id = self.region_id + f"___subset-{str(cell_ids)}"

        return RegionData(
            region_id=region_id,
            coordinates=self.coordinates.loc[cell_ids].copy(),
            expression=self.expression.loc[cell_ids].copy(),
            cell_types=self.cell_types.loc[cell_ids].copy(),
            microns_per_pixel=self.microns_per_pixel,
            polygons=self.polygons if polygons == "inherit" else polygons,
        )

    def neighborhood(
        self,
        center_cell_id: int,
        radius_um: float = 100.0,
    ):
        """Subset the local spatial neighborhood around one cell.

        Returns every cell whose centroid lies within ``radius_um`` microns of the
        centroid of ``center_cell_id`` (Euclidean distance). The radius is given in
        physical units (µm) and converted to coordinate units internally using
        :attr:`microns_per_pixel`, so the same ``radius_um`` selects a comparable
        physical area across datasets regardless of pixel size.

        Parameters
        ----------
        center_cell_id : int
            ``cell_id`` (coordinates index value) of the focal cell.
        radius_um : float
            Neighborhood radius in microns (default 100 µm).

        Returns
        -------
        RegionData | list[int]
        """
        if center_cell_id not in self.coordinates.index:
            raise KeyError(
                f"center_cell_id {center_cell_id!r} not in region {self.region_id!r}"
            )

        radius_px = radius_um / self.microns_per_pixel
        cx, cy = self.coordinates.loc[center_cell_id, ["x", "y"]]
        dx = self.coordinates["x"] - cx
        dy = self.coordinates["y"] - cy
        within = (dx * dx + dy * dy) <= radius_px * radius_px

        neighbor_ids = self.coordinates.index[within].tolist()

        pgs = "inherit"
        if self.polygons:
            from shapely.geometry import Point
            disk = Point(float(cx), float(cy)).buffer(radius_px)
            pgs = {k: g.intersection(disk) for k, g in self.polygons.items()}

        return self.subset(
            neighbor_ids,
            region_id=f"{self.region_id}___nbhd-cell{center_cell_id}-r{radius_um:g}um",
            polygons=pgs,
        )


class TMEDataset:
    """Wraps a processed benchmark dataset folder described by a YAML config.

    Config keys
    -----------
    name : str
    root : str  — path relative to ``data_root``
    metadata_files : list[str]  — filenames under ``root``; concatenated if multiple
    regions_dir : str  — subfolder of ``root`` containing per-region directories
    normalization : {method: arcsinh|none, cofactor: float | "quantile",
                     quantile: float, quantile_scale: float}
        Stateless arcsinh variance-stabilisation only. Feature z-scoring is done
        model-side (fit on the training split) — see ``benchmark/models/``.
    tasks : list of task dicts (see :meth:`get_task_metadata`)
    validation : dict with strategy parameters
    cell_type_col : str  — column in cell_types.csv to use as the cell type label
    microns_per_pixel : float  — physical size of one coordinate unit in microns
        (default 1.0). Set to the imaging pixel size for datasets whose
        coordinates are stored in pixels; leave at 1.0 for coordinates already in
        microns. Propagated to each loaded :class:`RegionData`.
    """

    def __init__(self, config: dict, data_root: str | Path):
        self.config = config
        self.name = config["name"]
        self._root = self._resolve_existing_path(data_root, config["root"])
        self._regions_dir = self._resolve_existing_path(self._root, config.get("regions_dir", "regions"))
        self._meta_files = config.get("metadata_files", ["metadata.csv"])
        self._cell_type_col = config.get("cell_type_col", "cell_type")

        # Per-region tissue/tumour polygon file (GeoJSON); loaded if present.
        self._polygons_file = config.get("polygons_file", DEFAULT_POLYGONS_FILE)

        # Physical size of one coordinate unit, in microns (1.0 if coords are in µm).
        if "microns_per_pixel" in config:
            self._microns_per_pixel = float(config["microns_per_pixel"])
        else:
            self._microns_per_pixel = 1.0
            warning(f"Dataset {self.name!r} config does not specify 'microns_per_pixel'; "
                    f"using default value of 1.0")

        self._metadata: pd.DataFrame | None = None
        self._region_cache: dict[tuple[str, bool], "RegionData"] = {}

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @staticmethod
    def _path_variants(path: str | Path) -> list[Path]:
        path = Path(path)
        if not path.parts:
            return [path]

        variants: list[tuple[str, ...]] = [()]
        for part in path.parts:
            next_variants: list[tuple[str, ...]] = []
            for prefix in variants:
                forms = {
                    part,
                    unicodedata.normalize("NFC", part),
                    unicodedata.normalize("NFD", part),
                }
                for form in forms:
                    next_variants.append(prefix + (form,))
            variants = next_variants
        return [Path(*parts) for parts in variants]

    def _resolve_existing_path(self, base: str | Path, path: str | Path) -> Path:
        base_path = Path(base)
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = base_path / candidate

        if candidate.exists():
            return candidate

        for variant in self._path_variants(candidate):
            if variant.exists():
                return variant

        return candidate

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, data_root: str | Path) -> "TMEDataset":
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(config, data_root)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def task_ids(self) -> list[str]:
        return [t["id"] for t in self.config.get("tasks", [])]

    @property
    def validation_config(self) -> dict:
        return self.config.get("validation", {})

    @property
    def microns_per_pixel(self) -> float:
        """Physical size of one coordinate unit in microns (see config)."""
        return self._microns_per_pixel

    def get_metadata(self) -> pd.DataFrame:
        """Return combined metadata DataFrame (all regions)."""
        if self._metadata is None:
            parts = []
            for fname in self._meta_files:
                df = pd.read_csv(self._root / fname, dtype={"patient_id": str})
                parts.append(df)
            self._metadata = pd.concat(parts, ignore_index=True)
        return self._metadata

    def get_task_config(self, task_id: str) -> dict:
        tasks = {t["id"]: t for t in self.config.get("tasks", [])}
        if task_id not in tasks:
            raise KeyError(f"Task '{task_id}' not found in dataset '{self.name}'. "
                           f"Available: {list(tasks)}")
        task = tasks[task_id]
        if task["type"] == "survival":
            for required_key in ("time_col", "event_col"):
                if required_key not in task:
                    raise ValueError(
                        f"Survival task '{task_id}' in dataset '{self.name}' "
                        f"must specify '{required_key}'."
                    )
        return task

    def get_task_metadata(self, task_id: str) -> pd.DataFrame:
        """Return metadata rows relevant for *task_id*, with NaN labels dropped.

        Task config keys:
          type : survival | binary | multiclass
          label_col | time_col + event_col
          row_filter : pandas query string (optional)
          drop_na_cols : list[str] — extra columns to require non-NaN
        """
        task = self.get_task_config(task_id)
        df = self.get_metadata()

        # Apply row filter
        row_filter = task.get("row_filter")
        if row_filter:
            df = df.query(row_filter).copy()

        # Drop rows with missing labels
        if task["type"] == "survival":
            required = [task["time_col"], task["event_col"]]
        else:
            # binary_classification / multiclass_classification both key off label_col
            required = [task["label_col"]]
        required += task.get("drop_na_cols", [])
        df = df.dropna(subset=required).copy()

        return df.reset_index(drop=True)

    def build_target(self, region_ids: list[str], task_id: str):
        """Region-level prediction target for *task_id*, indexed by ``region_id``
        and ordered to match ``region_ids`` (so it aligns with features/predictions):

          * binary classification     -> Series of 0/1 (1 = ``positive_class``)
          * multiclass classification -> Series of raw labels
          * survival                  -> DataFrame with columns ``time`` / ``event``
        """
        task = self.get_task_config(task_id)
        m = self.get_metadata().set_index("region_id").loc[list(region_ids)]
        if task["type"] == "survival":
            y = pd.DataFrame(
                {"time": m[task["time_col"]].astype(float),
                 "event": m[task["event_col"]].astype(float)},
                index=m.index,
            )
        elif task["type"] in ("binary", "binary_classification"):
            y = (m[task["label_col"]] == task.get("positive_class")).astype(int)
        else:  # multiclass
            y = m[task["label_col"]]
        y.index.name = "region_id"
        return y

    # ------------------------------------------------------------------
    # Region loading
    # ------------------------------------------------------------------

    def region_dir(self, region_id: str) -> Path:
        return self._regions_dir / region_id

    def load_region(
        self,
        region_id: str,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> RegionData:
        """Load coordinates, expression, and cell_types for one region.

        By default the expression is normalised with the dataset's ``normalization``
        config (the stateless arcsinh transform in :mod:`benchmark.data.transforms`),
        so the returned RegionData is ready for feature extraction. Pass
        ``normalize=False`` for the raw expression.

        Loaded regions are cached in-memory (keyed by ``region_id`` and the
        ``normalize`` flag); set ``use_cache=False`` to bypass, and
        :meth:`clear_region_cache` to free the memory.
        """
        key = (region_id, normalize)
        if use_cache and key in self._region_cache:
            return self._region_cache[key]

        d = self.region_dir(region_id)
        if not d.exists():
            raise FileNotFoundError(f"Region directory not found: {d}")

        coords = pd.read_csv(d / "coordinates.csv").set_index("cell_id")
        expr = pd.read_csv(d / "expression.csv").set_index("cell_id")
        ct = pd.read_csv(d / "cell_types.csv").set_index("cell_id")

        # Normalise cell_type column name
        if self._cell_type_col not in ct.columns and "cell_type" in ct.columns:
            ct = ct.rename(columns={"cell_type": self._cell_type_col})

        polygons = load_region_polygons(d / self._polygons_file)

        region = RegionData(
            region_id=region_id,
            coordinates=coords,
            expression=expr,
            cell_types=ct,
            microns_per_pixel=self._microns_per_pixel,
            polygons=polygons,
        )

        if normalize:
            from benchmark.data.transforms import apply_normalization
            region = apply_normalization(region, self.config.get("normalization", {}))

        if use_cache:
            self._region_cache[key] = region
        return region

    def load_regions(
        self,
        region_ids: Iterable[str],
        normalize: bool = True,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> list[RegionData]:
        """Load multiple regions (see :meth:`load_region`), optionally with a tqdm bar."""
        ids = list(region_ids)
        if show_progress:
            try:
                from tqdm import tqdm
                ids = tqdm(ids, desc=f"Loading {self.name}")
            except ImportError:
                pass
        return [self.load_region(rid, normalize=normalize, use_cache=use_cache)
                for rid in ids]

    def iter_regions(
        self,
        region_ids: Iterable[str],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> Iterator[RegionData]:
        """Lazy iterator over regions (see :meth:`load_region`)."""
        for rid in region_ids:
            yield self.load_region(rid, normalize=normalize, use_cache=use_cache)

    def clear_region_cache(self) -> None:
        """Drop all cached RegionData (frees the memory held by :meth:`load_region`)."""
        self._region_cache.clear()

    @property
    def all_region_ids(self) -> list[str]:
        """All region IDs present in the regions directory."""
        return sorted(p.name for p in self._regions_dir.iterdir() if p.is_dir())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TMEDataset(name={self.name!r}, root={self._root})"
