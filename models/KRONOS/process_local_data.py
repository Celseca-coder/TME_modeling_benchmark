from __future__ import annotations

"""Run frozen KRONOS embeddings on the local TME benchmark datasets.

Each region is represented by the mean and standard deviation of its KRONOS
CLS embeddings over 224 x 224 image patches.  The resulting region vectors are
evaluated with the benchmark's existing linear classification/Cox heads.

Examples (from the repository root)
-----------------------------------
    python models/KRONOS/process_local_data.py check \
        --data-roots /autofs/nas8/tywang/tjzou/TME_benchmark_data
    python models/KRONOS/process_local_data.py precompute --datasets bc_jackson2020 \
        --data-roots /autofs/nas8/tywang/tjzou/TME_benchmark_data --image-mode auto
    python models/KRONOS/process_local_data.py benchmark \
        --data-roots /autofs/nas8/tywang/tjzou/TME_benchmark_data --device cuda

Native images and marker sidecars follow the same layout accepted by Eva.  A
dataset YAML can override discovery with a ``kronos`` section, for example::

    kronos:
      image_file: images/{region_id}.ome.tif
      markers_file: markers.csv
      channel_axis: 0
      max_value: 65535
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
CODE_DIR = MODEL_DIR.parents[1]
DEFAULT_RESULTS = CODE_DIR / "model_results" / "KRONOS"
DEFAULT_CACHE = DEFAULT_RESULTS / "embeddings"
DEFAULT_MODEL_CACHE = DEFAULT_RESULTS / "model_assets"
DEFAULT_OUTPUT = CODE_DIR / "results" / "kronos_benchmark.csv"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(MODEL_DIR))

from benchmark.models.linear import LinearClassifier, LinearCox  # noqa: E402
from benchmark.utils.registry import list_datasets, load_dataset  # noqa: E402
from benchmark.validation import (  # noqa: E402
    PRIMARY_METRIC,
    cohort_split_test,
    cross_validate,
    summarize_folds,
)


IMAGE_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".npy", ".npz")
IMAGE_NAMES = (
    "image.ome.tif", "image.ome.tiff", "image.tif", "image.tiff",
    "mif.ome.tif", "mif.tif", "multiplex.ome.tif", "multiplex.tif",
    "image.npy", "mif.npy", "image.npz", "mif.npz",
)
MARKER_FILES = ("markers.json", "markers.txt", "markers.csv", "channels.csv")


def log(message: str) -> None:
    print(message, flush=True)


def _marker_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


MARKER_ALIASES = {
    "dna": "dapi", "dna1": "dapi", "dna2": "dapi", "ir191": "dapi",
    "ir193": "dapi", "hoechst": "dapi", "nuclei": "dapi",
    "cd3epsilon": "cd3e", "cd3": "cd3e", "cd8a": "cd8",
    "cd20a": "cd20", "ms4a1": "cd20", "mki67": "ki67",
    "pancytokeratin": "panck", "cytokeratin": "panck",
    "pdcd1": "pd1", "cd274": "pdl1", "ecadherin": "ecad",
}


def _alias_key(value: object) -> str:
    key = _marker_key(value)
    # Cyclic IF panels commonly contain one DAPI acquisition per cycle
    # (DAPI, DAPI2, ..., DAPI15). They all have the same biological identity;
    # _match_markers subsequently keeps only the first occurrence/marker ID.
    if re.fullmatch(r"dapi\d*", key):
        return "dapi"
    return MARKER_ALIASES.get(key, key)


def _read_markers(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("markers", value.get("channels", []))
        return [str(x).strip() for x in value]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        columns = {_marker_key(column): column for column in frame.columns}
        for candidate in ("marker", "biomarker", "markername", "channelname"):
            if candidate in columns:
                return [str(x).strip() for x in frame[columns[candidate]].dropna()]
        # Headerless one-column CSV: reload so its first marker is not consumed
        # as the column name (as happens for 4301_channelnames.csv).
        frame = pd.read_csv(path, header=None)
        column = frame.iloc[:, 0] if frame.shape[1] == 1 else frame.iloc[:, -1]
        return [str(x).strip() for x in column.dropna()]
    frame = pd.read_csv(path, header=None)
    column = frame.iloc[:, 0] if frame.shape[1] == 1 else frame.iloc[:, -1]
    return [str(x).strip() for x in column.dropna()]


def _format_path(value: str, ds, region_id: str) -> Path:
    metadata = ds.get_metadata()
    hit = metadata.loc[metadata["region_id"].astype(str) == str(region_id)]
    fields = {"region_id": region_id, "dataset_root": str(ds._root)}
    if len(hit):
        fields.update({k: v for k, v in hit.iloc[0].to_dict().items() if pd.notna(v)})
    rendered = str(value).format_map(fields)
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else ds._root / path


def _resolve_native_files(ds, region_id: str) -> tuple[Path, list[str], int | None, float | None]:
    region_dir = ds.region_dir(region_id)
    cfg = ds.config.get("kronos", {})
    image_setting = cfg.get("image_file") or ds.config.get("image_file")
    image_root = cfg.get("image_root") or ds.config.get("image_root")
    image_path = _format_path(image_setting, ds, region_id) if image_setting else None
    if image_path is not None and image_root and not Path(str(image_setting)).is_absolute():
        root = _format_path(str(image_root), ds, region_id)
        rendered = Path(str(image_setting).format_map({
            "region_id": region_id,
            **{
                key: value
                for key, value in ds.get_metadata().loc[
                    lambda frame: frame["region_id"].astype(str) == str(region_id)
                ].iloc[0].to_dict().items()
                if pd.notna(value)
            },
        }))
        image_path = root / rendered

    if image_path is None or not image_path.exists():
        metadata = ds.get_metadata()
        hit = metadata.loc[metadata["region_id"].astype(str) == str(region_id)]
        for column in cfg.get("image_path_columns", [
            "image_path", "image_file", "raw_image_path", "mif_path", "tiff_path"
        ]):
            if len(hit) and column in hit.columns and pd.notna(hit.iloc[0][column]):
                candidate = _format_path(str(hit.iloc[0][column]), ds, region_id)
                if candidate.exists():
                    image_path = candidate
                    break
    if image_path is None or not image_path.exists():
        shared = ds._root / "images"
        candidates = [p for suffix in IMAGE_SUFFIXES for p in shared.glob(f"{region_id}{suffix}")]
        if len(candidates) == 1:
            image_path = candidates[0]
    if image_path is None or not image_path.exists():
        image_path = next((region_dir / n for n in IMAGE_NAMES if (region_dir / n).exists()), None)
    if image_path is None:
        candidates = [p for p in region_dir.iterdir() if p.is_file()
                      and any(p.name.lower().endswith(s) for s in IMAGE_SUFFIXES)]
        if len(candidates) == 1:
            image_path = candidates[0]
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"No native multiplex image found for {region_id} in {region_dir}")

    markers = cfg.get("markers")
    if markers:
        markers = [str(x).strip() for x in markers]
    else:
        marker_setting = cfg.get("markers_file") or ds.config.get("markers_file")
        marker_path = _format_path(marker_setting, ds, region_id) if marker_setting else None
        if marker_path is None or not marker_path.exists():
            marker_path = next((region_dir / n for n in MARKER_FILES if (region_dir / n).exists()), None)
        if marker_path is not None:
            markers = _read_markers(marker_path)
        else:
            expression = region_dir / "expression.csv"
            if not expression.exists():
                raise FileNotFoundError(f"No marker list found for {image_path}")
            markers = [c for c in pd.read_csv(expression, nrows=0).columns if c != "cell_id"]
    max_value = cfg.get("max_value")
    return image_path, markers, cfg.get("channel_axis"), float(max_value) if max_value else None


def _load_image(path: Path) -> np.ndarray:
    lower = path.name.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r"))
    if lower.endswith(".npz"):
        archive = np.load(path)
        return np.asarray(archive["image" if "image" in archive.files else archive.files[0]])
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("TIFF input requires: pip install tifffile") from exc
    return np.asarray(tifffile.imread(path))


def _as_hwc(image: np.ndarray, n_channels: int, channel_axis: int | None) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D multiplex image, got shape {image.shape}")
    if channel_axis is None:
        matches = [i for i, size in enumerate(image.shape) if size == n_channels]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer channel axis from {image.shape} and {n_channels} markers; "
                "set kronos.channel_axis in the dataset YAML"
            )
        channel_axis = matches[0]
    image = np.moveaxis(image, int(channel_axis), -1)
    if image.shape[-1] != n_channels:
        raise ValueError(f"Image has {image.shape[-1]} channels but marker list has {n_channels}")
    return image


def _robust_unit_scale(image: np.ndarray) -> np.ndarray:
    """Scale rasterized expression independently per channel to [0, 1]."""
    out = np.zeros(image.shape, dtype=np.float32)
    for channel in range(image.shape[-1]):
        values = np.asarray(image[..., channel], dtype=np.float32)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        low, high = np.percentile(finite, [1.0, 99.5])
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high > low:
            out[..., channel] = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
    return out


def _rasterize(region, columns: list[str], radius: int) -> np.ndarray:
    coords = region.coordinates[["x", "y"]].to_numpy(float)
    if not len(coords) or not np.isfinite(coords).all():
        raise ValueError(f"Region {region.region_id} has invalid coordinates")
    coords -= coords.min(axis=0, keepdims=True)
    xy = np.rint(coords).astype(int)
    width, height = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
    image = np.zeros((height + 2 * radius, width + 2 * radius, len(columns)), dtype=np.float32)
    values = region.expression[columns].reindex(region.coordinates.index).to_numpy(np.float32)
    offsets = [(dx, dy) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius]
    for dx, dy in offsets:
        np.maximum.at(image, (xy[:, 1] + dy + radius, xy[:, 0] + dx + radius, slice(None)),
                      np.nan_to_num(values))
    return _robust_unit_scale(image)


def _patches(image: np.ndarray, size: int, stride: int,
             min_foreground: float) -> Iterable[np.ndarray]:
    height, width, _ = image.shape
    image = np.pad(image, ((0, max(0, size - height)), (0, max(0, size - width)), (0, 0)))
    height, width = image.shape[:2]
    ys = list(range(0, max(1, height - size + 1), stride))
    xs = list(range(0, max(1, width - size + 1), stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)
    result = [image[y:y + size, x:x + size] for y in ys for x in xs
              if np.mean(np.any(image[y:y + size, x:x + size] > 0, axis=-1)) >= min_foreground]
    return result or [image[:size, :size]]


def _load_marker_metadata(path: str | None, hf_repo: str, model_cache: Path,
                          hf_token: str | None) -> tuple[pd.DataFrame, Path]:
    if path:
        metadata_path = Path(path).expanduser()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub or pass --marker-metadata") from exc
        metadata_path = Path(hf_hub_download(
            repo_id=hf_repo, filename="marker_metadata.csv", cache_dir=str(model_cache), token=hf_token
        ))
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    frame = pd.read_csv(metadata_path)
    required = {"marker_name", "marker_id", "marker_mean", "marker_std"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metadata_path} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["_key"] = frame["marker_name"].map(_alias_key)
    frame = frame.drop_duplicates("_key").set_index("_key")
    return frame, metadata_path


class KronosFeaturizer:
    """Return one frozen-KRONOS feature vector per benchmark region."""

    def __init__(self, ds, args):
        self.ds = ds
        self.args = args
        self.device = args.device
        self.model_cache = Path(args.model_cache)
        self.marker_meta, self.marker_meta_path = _load_marker_metadata(
            args.marker_metadata, args.hf_repo, self.model_cache, args.hf_token
        )
        signature = json.dumps({
            "checkpoint": args.checkpoint or args.hf_repo,
            "model_type": args.model_type,
            "token_overlap": args.token_overlap,
            "mode": args.image_mode,
            "stride": args.stride,
            "max_patches": args.max_patches,
            "max_value": args.max_value,
            "radius": args.raster_radius,
        }, sort_keys=True)
        dataset_name = ds.config.get("name", "dataset")
        self.cache_dir = Path(args.cache_dir) / args.image_mode / dataset_name
        self.cache_dir /= hashlib.sha1(signature.encode()).hexdigest()[:12]
        self._model = None
        self._precision = None

    def fit(self, regions):
        return self

    def _get_model(self):
        if self._model is None:
            from kronos import create_model_from_pretrained
            checkpoint = self.args.checkpoint or f"hf_hub:{self.args.hf_repo}"
            self._model, self._precision, _ = create_model_from_pretrained(
                checkpoint_path=checkpoint,
                cfg_path=self.args.cfg_path,
                cache_dir=str(self.model_cache),
                hf_auth_token=self.args.hf_token,
                cfg=None if self.args.cfg_path else {
                    "model_type": self.args.model_type,
                    "token_overlap": self.args.token_overlap,
                },
            )
            self._model.to(self.device).eval()
        return self._model

    def _match_markers(self, markers: list[str]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        image_indices, ids, means, stds = [], [], [], []
        seen_ids: set[int] = set()
        for index, marker in enumerate(markers):
            key = _alias_key(marker)
            if key not in self.marker_meta.index:
                continue
            row = self.marker_meta.loc[key]
            marker_id = int(row["marker_id"])
            std = float(row["marker_std"])
            # A TIFF may contain two metal channels assigned to the same
            # biological marker (for example HistoneH3 or DAPI/DNA2). KRONOS
            # expects marker identities, so retain the first channel only.
            if marker_id in seen_ids or not np.isfinite(std) or std <= 0:
                continue
            seen_ids.add(marker_id)
            image_indices.append(index)
            ids.append(marker_id)
            means.append(float(row["marker_mean"]))
            stds.append(std)
        if not image_indices:
            raise ValueError(f"No KRONOS-supported markers among: {markers}")
        return image_indices, np.asarray(ids), np.asarray(means), np.asarray(stds)

    def _input(self, region):
        mode = self.args.image_mode
        if mode in ("native", "auto"):
            try:
                path, markers, channel_axis, configured_max = _resolve_native_files(
                    self.ds, region.region_id
                )
                raw = _as_hwc(_load_image(path), len(markers), channel_axis)
                max_value = configured_max or self.args.max_value
                if max_value <= 0:
                    raise ValueError("--max-value must be positive")
                image = np.nan_to_num(raw.astype(np.float32)) / float(max_value)
                source = str(path)
            except (FileNotFoundError, ValueError, OSError) as exc:
                if mode == "native":
                    raise
                log(
                    f"  KRONOS native input unavailable for {region.region_id}: {exc}; "
                    "falling back to expression rasterization"
                )
                mode = "rasterized"
        if mode == "rasterized":
            markers = list(region.expression.columns)
            image = _rasterize(region, markers, self.args.raster_radius)
            source = "rasterized"
        indices, marker_ids, means, stds = self._match_markers(markers)
        image = image[..., indices]
        image = (image - means[None, None, :]) / stds[None, None, :]
        return image.astype(np.float32), marker_ids, [markers[i] for i in indices], source

    def _one(self, region) -> np.ndarray:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(region.region_id))
        cache_path = self.cache_dir / f"{safe}.npz"
        if cache_path.exists():
            return np.load(cache_path)["feature"]
        image, marker_ids, markers, source = self._input(region)
        patches = list(_patches(image, 224, self.args.stride, self.args.min_foreground))
        if self.args.max_patches is not None:
            patches = patches[:self.args.max_patches]
        model = self._get_model()
        import torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(patches), self.args.batch_size):
                batch_np = np.stack(patches[start:start + self.args.batch_size])
                batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(
                    self.device, dtype=self._precision
                )
                ids = torch.as_tensor(marker_ids, dtype=torch.long, device=self.device)
                ids = ids.unsqueeze(0).expand(len(batch), -1)
                patch_embeddings, _, _ = model(batch, marker_ids=ids)
                outputs.append(patch_embeddings.detach().float().cpu().numpy())
        embeddings = np.concatenate(outputs, axis=0)
        feature = np.concatenate([embeddings.mean(0), embeddings.std(0)]).astype(np.float32)
        np.savez_compressed(cache_path, feature=feature, markers=np.asarray(markers),
                            marker_ids=marker_ids, source=np.asarray(source), n_patches=len(patches))
        return feature

    def transform(self, regions):
        rows = []
        for index, region in enumerate(regions, 1):
            log(f"  KRONOS embedding {index}/{len(regions)}: {region.region_id}")
            rows.append(self._one(region))
        columns = [f"kronos_{i:04d}" for i in range(len(rows[0]))]
        return pd.DataFrame(rows, index=[r.region_id for r in regions], columns=columns)


def _load_dataset_from_roots(name: str, roots: list[str] | None):
    errors = []
    for root in roots or [None]:
        try:
            ds = load_dataset(name, data_root=root)
            if not ds._root.exists() or not ds._regions_dir.exists():
                raise FileNotFoundError(f"missing {ds._root} or {ds._regions_dir}")
            ds.get_metadata()
            return ds, root
        except Exception as exc:
            errors.append(f"{root or '<registry default>'}: {exc}")
    raise FileNotFoundError(f"Could not load {name} from any data root:\n" + "\n".join(errors))


def model_factory(task_cfg, seed):
    return LinearCox(seed=seed) if task_cfg["type"] == "survival" else LinearClassifier(seed=seed)


def _limit_regions(ds, maximum: int | None) -> None:
    if maximum is None:
        return
    allowed = set(ds.get_metadata()["region_id"].astype(str).head(maximum))
    original = ds.get_task_metadata
    ds.get_task_metadata = lambda task, _original=original: _original(task).loc[
        lambda frame: frame["region_id"].astype(str).isin(allowed)
    ].reset_index(drop=True)


def _restrict_to_available_regions(ds, enabled: bool) -> None:
    """Restrict metadata in memory to region directories that exist on disk."""
    if not enabled:
        return
    metadata = ds.get_metadata()
    available = metadata["region_id"].astype(str).map(
        lambda region_id: ds.region_dir(region_id).is_dir()
    )
    kept = int(available.sum())
    dropped = int((~available).sum())
    if not kept:
        raise FileNotFoundError(
            f"No region directories from {ds._root} match its metadata"
        )
    ds._metadata = metadata.loc[available].reset_index(drop=True)
    log(
        f"[{ds.config.get('name', 'dataset')}] available-regions-only: "
        f"kept={kept}, dropped={dropped} (in memory only)"
    )


def run_benchmark(args) -> pd.DataFrame:
    rows = []
    for dataset_name in args.datasets or list_datasets():
        ds, matched_root = _load_dataset_from_roots(
            dataset_name, args.data_roots or ([args.data_root] if args.data_root else None)
        )
        _restrict_to_available_regions(ds, args.available_regions_only)
        _limit_regions(ds, args.max_regions)
        log(f"=== {dataset_name}; data root={matched_root or '<registry default>'} ===")

        def make_featurizer():
            return KronosFeaturizer(ds, args)

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            fold_metrics = cross_validate(
                ds, task, make_featurizer, model_factory, seeds=args.seeds, normalize=False
            )
            mean, sd = summarize_folds(fold_metrics, metric)
            rows.append(dict(dataset=dataset_name, task=task, scheme="cv", metric=metric,
                             mean=mean, sd=sd, n=len(fold_metrics)))
            log(f"  {task} cv {metric}: {mean:.4f} +/- {sd:.4f}")
        if args.max_regions is None:
            for test in ds.validation_config.get("generalization_tests", []):
                for task in test.get("tasks", ds.task_ids):
                    metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                    results = cohort_split_test(
                        ds, task, test, make_featurizer, model_factory,
                        seeds=args.seeds, normalize=False
                    )
                    if results:
                        mean, sd = summarize_folds(results, metric)
                        rows.append(dict(dataset=dataset_name, task=task, scheme=test["name"],
                                         metric=metric, mean=mean, sd=sd, n=len(results)))
        ds.clear_region_cache()
    result = pd.DataFrame(rows)
    if len(result):
        result["score"] = result.apply(lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1)
    return result


def check(args) -> int:
    log(f"KRONOS code: {MODEL_DIR}")
    try:
        marker_meta, marker_path = _load_marker_metadata(
            args.marker_metadata, args.hf_repo, Path(args.model_cache), args.hf_token
        )
        log(f"Marker metadata: OK ({len(marker_meta)} markers; {marker_path})")
    except Exception as exc:
        log(f"Marker metadata: ERROR: {exc}")
    for name in args.datasets or list_datasets():
        try:
            ds, root = _load_dataset_from_roots(
                name, args.data_roots or ([args.data_root] if args.data_root else None)
            )
            metadata = ds.get_metadata()
            native = 0
            for region_id in metadata["region_id"].astype(str):
                try:
                    _resolve_native_files(ds, region_id)
                    native += 1
                except (FileNotFoundError, ValueError, OSError):
                    pass
            log(f"{name}: regions={len(metadata)}, native_images={native}, root={ds._root}, data_root={root}")
        except Exception as exc:
            log(f"{name}: ERROR: {exc}")
    return 0


def precompute(args) -> Path:
    rows = []
    for name in args.datasets or list_datasets():
        ds, root = _load_dataset_from_roots(
            name, args.data_roots or ([args.data_root] if args.data_root else None)
        )
        _restrict_to_available_regions(ds, args.available_regions_only)
        region_ids = ds.get_metadata()["region_id"].astype(str).drop_duplicates().tolist()
        if args.max_regions is not None:
            region_ids = region_ids[:args.max_regions]
        featurizer = KronosFeaturizer(ds, args)
        log(f"[{name}] precomputing {len(region_ids)} region(s); data_root={root}")
        for index, region_id in enumerate(region_ids, 1):
            try:
                region = ds.load_region(region_id, normalize=False, use_cache=False)
                feature = featurizer._one(region)
                rows.append({"dataset": name, "region_id": region_id, "status": "ok",
                             "n_features": len(feature), "error": ""})
                log(f"[{name}] {index}/{len(region_ids)} OK {region_id}")
            except Exception as exc:
                rows.append({"dataset": name, "region_id": region_id, "status": "error",
                             "n_features": "", "error": str(exc)})
                log(f"[{name}] {index}/{len(region_ids)} ERROR {region_id}: {exc}")
                if args.debug:
                    raise
        ds.clear_region_cache()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "precompute_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "region_id", "status", "n_features", "error"])
        writer.writeheader()
        writer.writerows(rows)
    (root / "precompute_timestamp.txt").write_text(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8"
    )
    log(f"Wrote {manifest}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["check", "precompute", "benchmark"], nargs="?", default="benchmark")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--data-roots", nargs="+", default=None)
    parser.add_argument("--image-mode", choices=["native", "rasterized", "auto"], default="auto")
    parser.add_argument("--checkpoint", default=None, help="Local KRONOS checkpoint; default: Hugging Face")
    parser.add_argument("--cfg-path", default=None, help="Optional local KRONOS config.json")
    parser.add_argument("--hf-repo", default="MahmoodLab/KRONOS")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--marker-metadata", default=None,
                        help="Local marker_metadata.csv; default: download from --hf-repo")
    parser.add_argument("--model-type", choices=["vits16", "vitl16"], default="vits16")
    parser.add_argument("--token-overlap", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min-foreground", type=float, default=0.01)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--max-regions", type=int, default=None, help="Smoke test; skips cohort tests")
    parser.add_argument(
        "--available-regions-only",
        action="store_true",
        help=("Restrict metadata in memory to region directories that exist. "
              "Does not write to or modify the dataset directory."),
    )
    parser.add_argument("--max-value", type=float, default=65535.0,
                        help="Native intensity divisor; dataset kronos.max_value overrides it")
    parser.add_argument("--raster-radius", type=int, default=2)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--model-cache", default=str(DEFAULT_MODEL_CACHE))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "check":
        raise SystemExit(check(args))
    if args.action == "precompute":
        precompute(args)
        return
    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"
    result = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    log(f"Wrote {output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
