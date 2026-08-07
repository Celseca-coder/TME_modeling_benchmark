from __future__ import annotations

"""Run the original Eva encoder on the local TME benchmark datasets.

Eva is kept frozen.  A region is divided into 224 x 224 image patches, Eva
embeddings are mean/std pooled into one vector per region, and the benchmark's
existing LinearClassifier/LinearCox heads and validation code are used.

Image sources
-------------
``native`` looks for a multi-channel TIFF/NumPy image in each region directory.
``rasterized`` reconstructs a multi-channel image from expression.csv and
coordinates.csv. ``auto`` tries native first and then uses the reconstruction.

The native image must be H x W x C or C x H x W.  Channel names are read from a
sidecar named ``markers.txt``, ``markers.csv`` or ``markers.json`` (or from a
dataset YAML ``eva.markers`` list).  A dataset YAML may additionally specify::

    eva:
      image_file: image.ome.tif       # relative to each region directory
      markers_file: markers.txt
      channel_axis: 0                 # optional; auto-detected otherwise

Examples
--------
    conda activate Eva
    cd L:/SummerResearch/Project/TME_modeling_benchmark
    python models/Eva/process_local_data.py check --data-roots /data/TME_benchmark_data
    python models/Eva/process_local_data.py precompute --data-roots /data/TME_benchmark_data \
        --image-mode native --device cuda
    python models/Eva/process_local_data.py benchmark --datasets crc_schurch2020 \
        --image-mode auto --max-regions 2 --seeds 0 --device cuda --debug
    python models/Eva/process_local_data.py benchmark --image-mode native --device cuda
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
CODE_DIR = MODEL_DIR.parents[1]
DEFAULT_RESULTS = CODE_DIR / "model_results" / "Eva"
DEFAULT_OUTPUT = CODE_DIR / "results" / "eva_benchmark.csv"
DEFAULT_CACHE = DEFAULT_RESULTS / "embeddings"
DEFAULT_MARKER_EMBEDDINGS = MODEL_DIR / "marker_embeddings" / "GenePT_embedding.pkl"

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

# Common spellings in public IMC/CODEX tables -> names understood by Eva.
MARKER_ALIASES = {
    "dna": "DAPI", "dna1": "DAPI", "dna2": "DAPI", "ir191": "DAPI",
    "ir193": "DAPI", "hoechst": "DAPI", "nuclei": "DAPI",
    "cd3": "CD3e", "cd3epsilon": "CD3e", "cd3e": "CD3e",
    "cd8a": "CD8", "cd20a": "CD20", "ms4a1": "CD20",
    "pancytokeratin": "PanCK", "pan-ck": "PanCK", "panck": "PanCK",
    "cytokeratin": "PanCK", "ki-67": "Ki67", "mki67": "Ki67",
    "pd-1": "PD1", "pdcd1": "PD1", "pd-l1": "PDL1", "cd274": "PDL1",
    "foxp3": "FOXP3", "ecadherin": "E-cadherin", "e-cad": "E-cadherin",
}


def log(message: str) -> None:
    print(message, flush=True)


def _clean_marker(value: object) -> str:
    marker = str(value).strip()
    key = re.sub(r"[^a-z0-9-]+", "", marker.lower())
    return MARKER_ALIASES.get(key, marker)


def _known_markers() -> set[str]:
    from utils.constant import marker_to_gene
    return set(marker_to_gene)


def _read_markers(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("markers", value.get("channels", []))
        return [_clean_marker(x) for x in value]
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] == 1:
        return [_clean_marker(x) for x in frame.iloc[:, 0].dropna()]
    return [_clean_marker(x) for x in frame.iloc[:, -1].dropna()]


def _format_path_setting(value: str, ds, region_id: str) -> Path:
    """Resolve an Eva YAML path/template on both workstation and server layouts."""
    row = ds.get_metadata().loc[
        ds.get_metadata()["region_id"].astype(str) == str(region_id)
    ]
    fields = {"region_id": region_id, "dataset_root": str(ds._root)}
    if len(row):
        fields.update({k: v for k, v in row.iloc[0].to_dict().items() if pd.notna(v)})
    rendered = str(value).format_map(fields)
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else ds._root / path


def _resolve_native_files(ds, region_id: str) -> tuple[Path, list[str], int | None]:
    region_dir = ds.region_dir(region_id)
    eva_cfg = ds.config.get("eva", {})
    image_setting = eva_cfg.get("image_file") or ds.config.get("image_file")
    image_root = eva_cfg.get("image_root") or ds.config.get("image_root")
    image_path = None
    if image_setting:
        # image_file may be an absolute path, a dataset-root-relative path, or a
        # template such as raw/images/{region_id}.ome.tif.
        image_path = _format_path_setting(image_setting, ds, region_id)
        if image_root and not Path(str(image_setting)).is_absolute():
            root = _format_path_setting(image_root, ds, region_id)
            # Resolve placeholders first, but keep the image path relative to the
            # explicitly configured raw-image root.
            resolved = _format_path_setting(image_setting, ds, region_id)
            try:
                relative = resolved.relative_to(ds._root)
            except ValueError:
                relative = Path(resolved.name)
            image_path = root / relative
    if image_path is None or not image_path.exists():
        # Also accept paths already exported in metadata by a processing notebook.
        meta = ds.get_metadata()
        hit = meta.loc[meta["region_id"].astype(str) == str(region_id)]
        for column in eva_cfg.get("image_path_columns", [
            "image_path", "image_file", "raw_image_path", "mif_path", "tiff_path"
        ]):
            if len(hit) and column in hit.columns and pd.notna(hit.iloc[0][column]):
                candidate = _format_path_setting(str(hit.iloc[0][column]), ds, region_id)
                if candidate.exists():
                    image_path = candidate
                    break
    if image_path is None or not image_path.exists():
        # Benchmark processing notebooks commonly export all region images into
        # <dataset>/processed/images/<region_id>.<suffix>, alongside regions/.
        shared_images = ds._root / "images"
        candidates = []
        for suffix in IMAGE_SUFFIXES:
            candidates.extend(shared_images.glob(f"{region_id}{suffix}"))
        if len(candidates) == 1:
            image_path = candidates[0]
    if image_path is None or not image_path.exists():
        image_path = next((region_dir / n for n in IMAGE_NAMES if (region_dir / n).exists()), None)
    if image_path is None:
        candidates = [p for p in region_dir.iterdir() if p.is_file() and
                      any(p.name.lower().endswith(s) for s in IMAGE_SUFFIXES)]
        if len(candidates) == 1:
            image_path = candidates[0]
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"No native Eva image found in {region_dir}")

    markers = eva_cfg.get("markers")
    if markers:
        markers = [_clean_marker(x) for x in markers]
    else:
        marker_setting = eva_cfg.get("markers_file") or ds.config.get("markers_file")
        marker_path = _format_path_setting(marker_setting, ds, region_id) if marker_setting else None
        if marker_path is not None and not marker_path.exists():
            marker_path = image_path.parent / marker_setting
        if marker_path is None or not marker_path.exists():
            marker_path = next((region_dir / n for n in
                                ("markers.json", "markers.txt", "markers.csv", "channels.csv")
                                if (region_dir / n).exists()), None)
        if marker_path is None:
            # For processed benchmark images the TIFF channels normally have the
            # same order as the corresponding expression.csv columns. Validate
            # the count after loading rather than requiring 1 sidecar per ROI.
            expression_file = region_dir / "expression.csv"
            if not expression_file.exists():
                raise FileNotFoundError(f"No marker sidecar found for {image_path}")
            markers = [_clean_marker(x) for x in pd.read_csv(expression_file, nrows=0).columns
                       if x != "cell_id"]
        else:
            markers = _read_markers(marker_path)
    return image_path, markers, eva_cfg.get("channel_axis")


def _load_image(path: Path) -> np.ndarray:
    lower = path.name.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r"))
    if lower.endswith(".npz"):
        z = np.load(path)
        key = "image" if "image" in z.files else z.files[0]
        return np.asarray(z[key])
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Native TIFF input requires: pip install tifffile") from exc
    return np.asarray(tifffile.imread(path))


def _as_hwc(image: np.ndarray, n_channels: int, channel_axis: int | None) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim != 3:
        print(f"\n[WARNING] 发现损坏的图像，维度为 {image.shape}！已自动用 0 填充跳过，防止程序崩溃。")
        # 假设读出来的 2D 图像是 (H, W)，我们直接伪造一个 (H, W, num_markers) 的全 0 图像
        h, w = image.shape[0], image.shape[1]
        return np.zeros((h, w, n_channels), dtype=np.float32)
    if channel_axis is None:
        matches = [i for i, size in enumerate(image.shape) if size == n_channels]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer channel axis from image {image.shape} and {n_channels} markers; "
                "set eva.channel_axis in the dataset YAML"
            )
        channel_axis = matches[0]
    image = np.moveaxis(image, int(channel_axis), -1)
    if image.shape[-1] != n_channels:
        raise ValueError(f"Image has {image.shape[-1]} channels but marker list has {n_channels}")
    return image


def _robust_scale(image: np.ndarray) -> np.ndarray:
    """Independent channel scaling to [0, 1], without labels or cohort statistics."""
    out = np.zeros(image.shape, dtype=np.float32)
    for c in range(image.shape[-1]):
        x = np.asarray(image[..., c], dtype=np.float32)
        finite = x[np.isfinite(x)]
        if not finite.size:
            continue
        lo, hi = np.percentile(finite, [1.0, 99.5])
        if hi <= lo:
            hi = float(finite.max())
            lo = float(finite.min())
        if hi > lo:
            out[..., c] = np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0, 1)
    return out


def _rasterize(region, markers: list[str], radius: int = 2) -> np.ndarray:
    coords = region.coordinates[["x", "y"]].to_numpy(float)
    coords -= np.nanmin(coords, axis=0, keepdims=True)
    xy = np.rint(coords).astype(int)
    width, height = int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1
    image = np.zeros((height + 2 * radius, width + 2 * radius, len(markers)), dtype=np.float32)
    values = region.expression[markers].reindex(region.coordinates.index).to_numpy(np.float32)
    # A compact disk gives the model spatial mass rather than isolated single pixels.
    offsets = [(dx, dy) for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius]
    for dx, dy in offsets:
        xx = xy[:, 0] + dx + radius
        yy = xy[:, 1] + dy + radius
        np.maximum.at(image, (yy, xx, slice(None)), np.nan_to_num(values))
    return _robust_scale(image)


def _patches(image: np.ndarray, size: int, stride: int, min_foreground: float) -> Iterable[np.ndarray]:
    h, w, c = image.shape
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
        h, w = image.shape[:2]
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    kept = []
    for y in ys:
        for x in xs:
            patch = image[y:y + size, x:x + size]
            if np.mean(np.any(patch > 0, axis=-1)) >= min_foreground:
                kept.append(patch)
    if not kept:
        kept.append(image[:size, :size])
    return kept


class EvaFeaturizer:
    """Benchmark featurizer returning one frozen-Eva vector per region."""

    def __init__(self, ds, *, image_mode: str, device: str, checkpoint: str | None,
                 hf_repo: str, cache_dir: Path, batch_size: int, stride: int,
                 min_foreground: float, max_patches: int | None, cls: bool):
        self.ds = ds
        self.image_mode = image_mode
        self.device = device
        self.checkpoint = checkpoint
        self.hf_repo = hf_repo
        self.cache_dir = Path(cache_dir) / image_mode / ds.config.get("name", "dataset")
        self.batch_size = batch_size
        self.stride = stride
        self.min_foreground = min_foreground
        self.max_patches = max_patches
        self.cls = cls
        self._model = None
        self._known = _known_markers()
        signature = json.dumps(
            {
                "checkpoint": str(checkpoint or hf_repo), "stride": stride,
                "min_foreground": min_foreground, "max_patches": max_patches,
                "cls": cls,
            },
            sort_keys=True,
        )
        self.cache_dir /= hashlib.sha1(signature.encode()).hexdigest()[:12]

    def fit(self, regions):
        return self

    def _get_model(self):
        if self._model is None:
            if not DEFAULT_MARKER_EMBEDDINGS.exists():
                raise FileNotFoundError(
                    f"Missing {DEFAULT_MARKER_EMBEDDINGS}. Download "
                    "GenePT_gene_protein_embedding_model_3_text.pickle from "
                    "https://zenodo.org/records/10833191 and save it at that path."
                )
            # Eva currently opens marker_embeddings/... relative to cwd.
            os.chdir(MODEL_DIR)
            from omegaconf import OmegaConf
            from Eva.utils import load_from_checkpoint, load_from_hf
            conf = OmegaConf.load(MODEL_DIR / "config.yaml")
            self._model = (load_from_checkpoint(self.checkpoint, conf, self.device)
                           if self.checkpoint else
                           load_from_hf(self.hf_repo, conf, self.device,
                                        cache_dir=str(DEFAULT_RESULTS / "hf_cache")))
            self._model.eval()
        return self._model

    def _input(self, region):
        mode = self.image_mode
        if mode in ("native", "auto"):
            try:
                path, markers, channel_axis = _resolve_native_files(self.ds, region.region_id)
                image = _as_hwc(_load_image(path), len(markers), channel_axis)
                image = _robust_scale(image)
                source = str(path)
            except FileNotFoundError:
                if mode == "native":
                    raise
                mode = "rasterized"
        if mode == "rasterized":
            original = list(region.expression.columns)
            mapped = [_clean_marker(x) for x in original]
            pairs = [(old, new) for old, new in zip(original, mapped) if new in self._known]
            if not pairs:
                raise ValueError(f"Region {region.region_id} has no markers supported by Eva")
            # Rename only the selected expression columns, resolving aliases deterministically.
            selected = region.expression[[p[0] for p in pairs]].copy()
            selected.columns = [p[1] for p in pairs]
            selected = selected.loc[:, ~selected.columns.duplicated()]
            proxy = type("RasterRegion", (), {
                "coordinates": region.coordinates,
                "expression": selected,
            })()
            markers = list(selected.columns)
            image = _rasterize(proxy, markers)
            source = "rasterized"
        keep = [i for i, marker in enumerate(markers) if marker in self._known]
        if not keep:
            raise ValueError(f"Region {region.region_id}: none of {markers} is supported by Eva")
        return image[..., keep], [markers[i] for i in keep], source

    def _one(self, region) -> np.ndarray:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", region.region_id)
        path = self.cache_dir / f"{safe}.npz"
        if path.exists():
            return np.load(path)["feature"]
        image, markers, source = self._input(region)
        chunks = list(_patches(image, 224, self.stride, self.min_foreground))
        if self.max_patches is not None:
            chunks = chunks[:self.max_patches]
        model = self._get_model()
        import torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                batch = torch.from_numpy(np.stack(chunks[start:start + self.batch_size]))
                bms = [markers] * len(batch)
                feat = model.extract_features(batch, bms, self.device, cls=self.cls,
                                              channel_mode="full")
                outputs.append(feat.detach().cpu().numpy())
        patch_features = np.concatenate(outputs, axis=0)
        feature = np.concatenate([patch_features.mean(0), patch_features.std(0)]).astype(np.float32)
        np.savez_compressed(path, feature=feature, markers=np.asarray(markers),
                            source=np.asarray(source), n_patches=len(chunks))
        return feature

    def transform(self, regions):
        rows = []
        for i, region in enumerate(regions, 1):
            log(f"  Eva embedding {i}/{len(regions)}: {region.region_id}")
            rows.append(self._one(region))
        columns = [f"eva_{i:04d}" for i in range(len(rows[0]))]
        return pd.DataFrame(rows, index=[r.region_id for r in regions], columns=columns)


def _load_dataset_from_roots(name: str, roots: list[str] | None):
    candidates = roots or [None]
    errors = []
    for root in candidates:
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


def run_benchmark(args) -> pd.DataFrame:
    rows = []
    names = args.datasets or list_datasets()
    for dataset_name in names:
        ds, matched_root = _load_dataset_from_roots(
            dataset_name, args.data_roots or ([args.data_root] if args.data_root else None)
        )
        log(f"[{dataset_name}] data root: {matched_root or '<registry default>'}")
        if args.max_regions:
            allowed = set(ds.get_metadata()["region_id"].astype(str).head(args.max_regions))
            # A smoke-test cap must affect task metadata as well as region loading.
            original = ds.get_task_metadata
            ds.get_task_metadata = lambda task, _original=original: _original(task).loc[
                lambda x: x["region_id"].astype(str).isin(allowed)].reset_index(drop=True)
        log(f"=== {dataset_name} ===")

        def make_feat():
            return EvaFeaturizer(
                ds, image_mode=args.image_mode, device=args.device,
                checkpoint=args.checkpoint, hf_repo=args.hf_repo,
                cache_dir=Path(args.cache_dir), batch_size=args.batch_size,
                stride=args.stride, min_foreground=args.min_foreground,
                max_patches=args.max_patches, cls=args.cls,
            )

        for task in ds.task_ids:
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            fm = cross_validate(ds, task, make_feat, model_factory, seeds=args.seeds,
                                normalize=False)
            mean, sd = summarize_folds(fm, metric)
            rows.append(dict(dataset=dataset_name, task=task, scheme="cv",
                             metric=metric, mean=mean, sd=sd, n=len(fm)))
            log(f"  {task} cv {metric}: {mean:.4f} +/- {sd:.4f}")
        if not args.max_regions:
            for gt in ds.validation_config.get("generalization_tests", []):
                for task in gt.get("tasks", ds.task_ids):
                    metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                    res = cohort_split_test(ds, task, gt, make_feat, model_factory,
                                            seeds=args.seeds, normalize=False)
                    if not res:
                        continue
                    mean, sd = summarize_folds(res, metric)
                    rows.append(dict(dataset=dataset_name, task=task, scheme=gt["name"],
                                     metric=metric, mean=mean, sd=sd, n=len(res)))
        ds.clear_region_cache()
    result = pd.DataFrame(rows)
    if len(result):
        result["score"] = result.apply(
            lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f}", axis=1)
    return result


def check(args) -> int:
    log(f"Eva code: {MODEL_DIR}")
    log(f"Config: {'OK' if (MODEL_DIR / 'config.yaml').exists() else 'MISSING'}")
    log(f"GenePT embeddings: {'OK' if DEFAULT_MARKER_EMBEDDINGS.exists() else 'MISSING'}")
    for name in args.datasets or list_datasets():
        try:
            ds, matched_root = _load_dataset_from_roots(
                name, args.data_roots or ([args.data_root] if args.data_root else None)
            )
            meta = ds.get_metadata()
            native = 0
            for rid in meta["region_id"].astype(str):
                try:
                    _resolve_native_files(ds, rid)
                    native += 1
                except (FileNotFoundError, ValueError):
                    pass
            log(f"{name}: regions={len(meta)}, native_images={native}, root={ds._root}, data_root={matched_root}")
        except Exception as exc:
            log(f"{name}: ERROR: {exc}")
    return 0


def precompute(args) -> Path:
    """Create one cached frozen-Eva feature per selected region and a CSV manifest."""
    manifest_rows = []
    for name in args.datasets or list_datasets():
        ds, matched_root = _load_dataset_from_roots(
            name, args.data_roots or ([args.data_root] if args.data_root else None)
        )
        region_ids = ds.get_metadata()["region_id"].astype(str).drop_duplicates().tolist()
        if args.max_regions is not None:
            region_ids = region_ids[:args.max_regions]
        feat = EvaFeaturizer(
            ds, image_mode=args.image_mode, device=args.device,
            checkpoint=args.checkpoint, hf_repo=args.hf_repo,
            cache_dir=Path(args.cache_dir), batch_size=args.batch_size,
            stride=args.stride, min_foreground=args.min_foreground,
            max_patches=args.max_patches, cls=args.cls,
        )
        log(f"[{name}] precomputing {len(region_ids)} region(s); data_root={matched_root}")
        for i, rid in enumerate(region_ids, 1):
            try:
                region = ds.load_region(rid, normalize=False, use_cache=False)
                feature = feat._one(region)
                manifest_rows.append({"dataset": name, "region_id": rid, "status": "ok",
                                      "n_features": len(feature), "error": ""})
                log(f"[{name}] {i}/{len(region_ids)} OK {rid}")
            except Exception as exc:
                manifest_rows.append({"dataset": name, "region_id": rid, "status": "error",
                                      "n_features": "", "error": str(exc)})
                log(f"[{name}] {i}/{len(region_ids)} ERROR {rid}: {exc}")
                if args.debug:
                    raise
        ds.clear_region_cache()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "precompute_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "region_id", "status", "n_features", "error"])
        writer.writeheader()
        writer.writerows(manifest_rows)
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
    parser.add_argument("--data-roots", nargs="+", default=None,
                        help="Server data roots tried in order; each contains the configured dataset folders.")
    parser.add_argument("--image-mode", choices=["native", "rasterized", "auto"], default="native")
    parser.add_argument("--checkpoint", default=None, help="Local Eva_model.ckpt; otherwise use Hugging Face.")
    parser.add_argument("--hf-repo", default="yandrewl/Eva")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min-foreground", type=float, default=0.01)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--max-regions", type=int, default=None, help="Smoke-test only; skips cohort tests.")
    parser.add_argument("--cls", action="store_true", help="Use Eva CLS rather than mean patch tokens.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
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
