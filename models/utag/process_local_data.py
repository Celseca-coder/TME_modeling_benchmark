from __future__ import annotations

"""Run UTAG on the local TME benchmark datasets.

The adapter treats UTAG's spatial message-passed cell matrix as the native
representation.  It returns one fixed-length vector per region using either:

* ``message-passing``: per-marker mean and standard deviation;
* ``domains``: proportions of train-fitted tissue domains;
* ``combined``: both (default).

For ``domains``/``combined``, the domain model is fitted only on training
regions inside each cross-validation fold.  Validation regions are assigned to
the frozen training centroids, preventing clustering leakage.

Motif-explanation verification (native domain portraits, v2 pseudo-labels)::

    python models/utag/verify_native_domain_portraits.py
    bash models/utag/run_verify_motif_explanations.sh
"""

import argparse
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
DEFAULT_RESULTS = CODE_DIR / "model_results" / "UTAG"
DEFAULT_CACHE = DEFAULT_RESULTS / "message_passing_cache"
DEFAULT_OUTPUT = CODE_DIR / "results" / "utag_benchmark.csv"

# Reused by the fresh featurizer created for every CV fold.  The cache contains
# input column names only (never labels or expression values).
_DATASET_COMMON_MARKERS: dict[str, list[str]] = {}

for path in (str(CODE_DIR), str(MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark.models.linear import LinearClassifier, LinearCox  # noqa: E402
from benchmark.utils.registry import list_datasets, load_dataset  # noqa: E402
from benchmark.validation import (  # noqa: E402
    PRIMARY_METRIC,
    cohort_split_test,
    cross_validate,
    summarize_folds,
)


def default_featurizer_args(**overrides) -> argparse.Namespace:
    """Defaults shared by the clinical adapter and motif verification scripts."""
    args = argparse.Namespace(
        feature_mode="domains",
        max_dist=20.0,
        normalization_mode="l1_norm",
        coordinate_mode="auto",
        expression_transform="arcsinh",
        arcsinh_cofactor=5.0,
        n_domains=10,
        max_fit_cells=100000,
        seed=0,
        cache_dir=str(DEFAULT_CACHE),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value[:200] or "region"


def _load_dataset_from_roots(name: str, roots: list[str] | None):
    errors: list[str] = []
    for root in roots or [None]:
        try:
            ds = load_dataset(name, data_root=root)
            if not ds._root.exists() or not ds._regions_dir.exists():
                raise FileNotFoundError(f"missing {ds._root} or {ds._regions_dir}")
            ds.get_metadata()
            return ds, root
        except Exception as exc:
            errors.append(f"{root or '<registry default>'}: {exc}")
    raise FileNotFoundError(
        f"Could not load {name} from any data root:\n" + "\n".join(errors)
    )


def _coordinate_frame(region, mode: str) -> tuple[pd.DataFrame, str]:
    if mode in ("auto", "um") and hasattr(region, "coordinates_um"):
        frame = region.coordinates_um
        if frame is not None and {"x", "y"}.issubset(frame.columns):
            return frame[["x", "y"]], "coordinates_um"
    if mode == "um":
        raise ValueError(f"Region {region.region_id} has no coordinates_um")
    if not {"x", "y"}.issubset(region.coordinates.columns):
        raise ValueError(f"Region {region.region_id} coordinates lack x/y columns")
    return region.coordinates[["x", "y"]], "coordinates"


def _transform_expression(values: np.ndarray, mode: str, cofactor: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), copy=False)
    if mode == "none":
        return values
    if mode == "arcsinh":
        if cofactor <= 0:
            raise ValueError("--arcsinh-cofactor must be positive")
        return np.arcsinh(values / cofactor).astype(np.float32, copy=False)
    if mode == "log1p":
        return np.log1p(np.clip(values, 0, None)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown expression transform: {mode}")


class UTAGFeaturizer:
    """Frozen UTAG message passing plus optional train-fitted tissue domains."""

    def __init__(self, ds, args):
        self.ds = ds
        self.args = args
        self.feature_mode = args.feature_mode
        self.n_domains = args.n_domains
        self._domain_model = None
        self._markers: list[str] | None = None

        signature = json.dumps(
            {
                "max_dist": args.max_dist,
                "normalization": args.normalization_mode,
                "coordinate_mode": args.coordinate_mode,
                "expression_transform": args.expression_transform,
                "arcsinh_cofactor": args.arcsinh_cofactor,
            },
            sort_keys=True,
        )
        dataset_name = ds.config.get("name", "dataset")
        self.cache_dir = Path(args.cache_dir) / dataset_name
        self.cache_dir /= hashlib.sha1(signature.encode()).hexdigest()[:12]

    def _select_markers(self, regions: Iterable) -> list[str]:
        columns: list[str] | None = None
        for region in regions:
            current = [str(x) for x in region.expression.columns]
            if columns is None:
                columns = current
            else:
                available = set(current)
                columns = [x for x in columns if x in available]
        if not columns:
            raise ValueError("No common expression markers among selected regions")
        return columns

    def _dataset_common_markers(self) -> list[str]:
        """Return markers present in every available region of this dataset.

        Generalization cohorts can use different panels.  Determining the input
        schema globally is safe (no outcomes or expression values are read) and
        gives train/test matrices identical columns without encoding an absent
        assay channel as biological zero expression.
        """
        key = str(self.ds._root.resolve())
        if key in _DATASET_COMMON_MARKERS:
            return _DATASET_COMMON_MARKERS[key]

        common: list[str] | None = None
        inspected = 0
        skipped = 0
        for region_id in self.ds.get_metadata()["region_id"].astype(str):
            path = self.ds.region_dir(region_id) / "expression.csv"
            if not path.exists():
                skipped += 1
                continue
            try:
                columns = [
                    str(column)
                    for column in pd.read_csv(path, nrows=0).columns
                    if str(column) != "cell_id"
                ]
            except Exception:
                skipped += 1
                continue
            inspected += 1
            if common is None:
                common = columns
            else:
                available = set(columns)
                common = [column for column in common if column in available]

        if not inspected:
            raise ValueError(f"No readable expression.csv headers under {self.ds._root}")
        if not common:
            raise ValueError(
                f"No expression markers are shared by all {inspected} available "
                f"regions in {self.ds.config.get('name', 'dataset')}"
            )
        _DATASET_COMMON_MARKERS[key] = common
        log(
            f"[{self.ds.config.get('name', 'dataset')}] common marker panel: "
            f"markers={len(common)}, inspected_regions={inspected}, skipped={skipped}"
        )
        return common

    def _cache_path(self, region, markers: list[str]) -> Path:
        marker_hash = hashlib.sha1("\n".join(markers).encode()).hexdigest()[:10]
        return self.cache_dir / marker_hash / f"{_safe_name(region.region_id)}.npz"

    def _message_pass(self, region, markers: list[str]) -> np.ndarray:
        path = self._cache_path(region, markers)
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            return cached["X"].astype(np.float32, copy=False)

        from anndata import AnnData
        from utag import utag

        coords, coord_source = _coordinate_frame(region, self.args.coordinate_mode)
        expression = region.expression.reindex(region.coordinates.index)
        missing = [marker for marker in markers if marker not in expression.columns]
        if missing:
            raise ValueError(
                f"Region {region.region_id} lacks {len(missing)} selected marker(s): {missing[:5]}"
            )

        values = _transform_expression(
            expression[markers].to_numpy(),
            self.args.expression_transform,
            self.args.arcsinh_cofactor,
        )
        xy = coords.reindex(region.coordinates.index).to_numpy(dtype=np.float32)
        valid = np.isfinite(xy).all(axis=1)
        if not valid.any():
            raise ValueError(f"Region {region.region_id} has no finite coordinates")
        if not valid.all():
            values, xy = values[valid], xy[valid]

        adata = AnnData(X=values)
        adata.var_names = markers
        adata.obsm["spatial"] = xy
        result = utag(
            adata,
            slide_key=None,
            max_dist=self.args.max_dist,
            normalization_mode=self.args.normalization_mode,
            apply_clustering=False,
            parallel=False,
            return_copy=False,
        )
        passed = result.X
        if hasattr(passed, "toarray"):
            passed = passed.toarray()
        passed = np.asarray(passed, dtype=np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=passed,
            markers=np.asarray(markers),
            coordinate_source=np.asarray(coord_source),
        )
        return passed

    def fit(self, regions):
        regions = list(regions)
        dataset_markers = self._dataset_common_markers()
        train_markers = set(self._select_markers(regions))
        self._markers = [marker for marker in dataset_markers if marker in train_markers]
        if not self._markers:
            raise ValueError("No common UTAG markers remain in this training fold")
        if self.feature_mode == "message-passing":
            return self

        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(self.args.seed)
        per_region = max(1, self.args.max_fit_cells // max(1, len(regions)))
        sampled: list[np.ndarray] = []
        for region in regions:
            values = self._message_pass(region, self._markers)
            if len(values) > per_region:
                indices = rng.choice(len(values), size=per_region, replace=False)
                values = values[indices]
            sampled.append(values)
        training_cells = np.concatenate(sampled, axis=0)
        if len(training_cells) < self.n_domains:
            raise ValueError(
                f"Only {len(training_cells)} training cells for {self.n_domains} UTAG domains"
            )
        self._domain_scaler = StandardScaler().fit(training_cells)
        training_cells = self._domain_scaler.transform(training_cells)
        self._domain_model = MiniBatchKMeans(
            n_clusters=self.n_domains,
            random_state=self.args.seed,
            batch_size=min(4096, max(256, len(training_cells))),
            n_init=10,
        ).fit(training_cells)
        return self

    def _one(self, region) -> np.ndarray:
        if self._markers is None:
            raise RuntimeError("UTAGFeaturizer.fit must be called before transform")
        values = self._message_pass(region, self._markers)
        parts: list[np.ndarray] = []
        if self.feature_mode in ("message-passing", "combined"):
            parts.extend([values.mean(axis=0), values.std(axis=0)])
        if self.feature_mode in ("domains", "combined"):
            if self._domain_model is None:
                raise RuntimeError("UTAG domain model was not fitted")
            labels = self._domain_model.predict(self._domain_scaler.transform(values))
            composition = np.bincount(labels, minlength=self.n_domains).astype(np.float32)
            composition /= max(1, len(labels))
            parts.append(composition)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def transform(self, regions):
        rows: list[np.ndarray] = []
        for index, region in enumerate(regions, start=1):
            log(f"  UTAG feature {index}/{len(regions)}: {region.region_id}")
            rows.append(self._one(region))
        if not rows:
            return pd.DataFrame()
        columns = [f"utag_{i:04d}" for i in range(len(rows[0]))]
        return pd.DataFrame(
            rows,
            index=[region.region_id for region in regions],
            columns=columns,
        )


def model_factory(task_cfg, seed):
    if task_cfg["type"] == "survival":
        return LinearCox(seed=seed)
    return LinearClassifier(seed=seed)


def _limit_regions(ds, maximum: int | None) -> None:
    if maximum is None:
        return
    allowed = set(ds.get_metadata()["region_id"].astype(str).head(maximum))
    original = ds.get_task_metadata
    ds.get_task_metadata = lambda task, _original=original: _original(task).loc[
        lambda frame: frame["region_id"].astype(str).isin(allowed)
    ].reset_index(drop=True)


def _restrict_to_available_regions(ds) -> tuple[int, int]:
    """Drop metadata rows whose region directory is absent (in memory only)."""
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
    if dropped:
        log(
            f"[{ds.config.get('name', 'dataset')}] skipping {dropped} metadata "
            f"region(s) with no directory; kept={kept} (in memory only)"
        )
    return kept, dropped


def run_benchmark(args) -> pd.DataFrame:
    rows: list[dict] = []
    names = args.datasets or list_datasets()
    for dataset_name in names:
        ds, matched_root = _load_dataset_from_roots(dataset_name, args.data_roots)
        _restrict_to_available_regions(ds)
        _limit_regions(ds, args.max_regions)
        log(f"[{dataset_name}] data root: {matched_root or '<registry default>'}")
        log(f"=== {dataset_name} ===")

        def make_featurizer():
            return UTAGFeaturizer(ds, args)

        for task in ds.task_ids:
            task_cfg = ds.get_task_config(task)
            if args.only_c_index and task_cfg["type"] != "survival":
                continue
            metric = PRIMARY_METRIC[task_cfg["type"]]
            fold_metrics = cross_validate(
                ds,
                task,
                make_featurizer,
                model_factory,
                seeds=args.seeds,
                normalize=False,
            )
            mean, sd = summarize_folds(fold_metrics, metric)
            rows.append(
                dict(
                    dataset=dataset_name,
                    task=task,
                    scheme="cv",
                    metric=metric,
                    mean=mean,
                    sd=sd,
                    n=len(fold_metrics),
                )
            )
            log(f"  {task} cv {metric}: {mean:.4f} +/- {sd:.4f}")

        if args.max_regions is None:
            for test in ds.validation_config.get("generalization_tests", []):
                for task in test.get("tasks", ds.task_ids):
                    task_cfg = ds.get_task_config(task)
                    if args.only_c_index and task_cfg["type"] != "survival":
                        continue
                    metric = PRIMARY_METRIC[task_cfg["type"]]
                    results = cohort_split_test(
                        ds,
                        task,
                        test,
                        make_featurizer,
                        model_factory,
                        seeds=args.seeds,
                        normalize=False,
                    )
                    if not results:
                        continue
                    mean, sd = summarize_folds(results, metric)
                    rows.append(
                        dict(
                            dataset=dataset_name,
                            task=task,
                            scheme=test["name"],
                            metric=metric,
                            mean=mean,
                            sd=sd,
                            n=len(results),
                        )
                    )
                    log(f"  {task} {test['name']} {metric}: {mean:.4f} +/- {sd:.4f}")
        ds.clear_region_cache()

    result = pd.DataFrame(rows)
    if len(result):
        result["score"] = result.apply(
            lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1
        )
    return result


def precompute(args) -> Path:
    """Precompute UTAG message-passed matrices without fitting tissue domains."""
    manifest: list[dict] = []
    names = args.datasets or list_datasets()
    for dataset_name in names:
        ds, matched_root = _load_dataset_from_roots(dataset_name, args.data_roots)
        metadata = ds.get_metadata()
        region_ids = metadata["region_id"].astype(str).tolist()
        if args.max_regions is not None:
            region_ids = region_ids[: args.max_regions]
        regions = []
        for region_id in region_ids:
            try:
                regions.append(
                    ds.load_region(region_id, normalize=False, use_cache=True)
                )
            except Exception as exc:
                status = "missing" if isinstance(exc, FileNotFoundError) else "load_error"
                manifest.append(
                    dict(
                        dataset=dataset_name,
                        region_id=region_id,
                        status=status,
                        n_cells="",
                        n_markers="",
                        error=str(exc),
                    )
                )
                log(f"[{dataset_name}] SKIP {region_id}: {exc}")
                if args.debug:
                    raise
        if not regions:
            log(f"[{dataset_name}] no loadable regions; skipping dataset")
            continue
        featurizer = UTAGFeaturizer(ds, args)
        featurizer._markers = featurizer._dataset_common_markers()
        log(
            f"[{dataset_name}] precomputing {len(regions)} region(s); "
            f"data_root={matched_root or '<registry default>'}"
        )
        for index, region in enumerate(regions, start=1):
            try:
                values = featurizer._message_pass(region, featurizer._markers)
                manifest.append(
                    dict(
                        dataset=dataset_name,
                        region_id=region.region_id,
                        status="ok",
                        n_cells=len(values),
                        n_markers=values.shape[1],
                        error="",
                    )
                )
                log(f"[{dataset_name}] {index}/{len(regions)} OK {region.region_id}")
            except Exception as exc:
                manifest.append(
                    dict(
                        dataset=dataset_name,
                        region_id=region.region_id,
                        status="error",
                        n_cells="",
                        n_markers="",
                        error=str(exc),
                    )
                )
                log(f"[{dataset_name}] {index}/{len(regions)} ERROR {region.region_id}: {exc}")
                if args.debug:
                    raise
        ds.clear_region_cache()

    output = Path(args.results_root) / "precompute_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(output, index=False)
    log(f"Wrote {output}")
    return output


def check(args) -> int:
    try:
        import anndata
        import scanpy
        import squidpy
        import utag

        log(
            "UTAG dependencies: OK "
            f"(anndata={anndata.__version__}, scanpy={scanpy.__version__}, "
            f"squidpy={squidpy.__version__})"
        )
        log(f"UTAG package: {Path(utag.__file__).resolve()}")
    except Exception as exc:
        log(f"UTAG dependencies: ERROR: {exc}")
        return 1

    status = 0
    for name in args.datasets or list_datasets():
        try:
            ds, root = _load_dataset_from_roots(name, args.data_roots)
            _restrict_to_available_regions(ds)
            metadata = ds.get_metadata()
            first_id = str(metadata.iloc[0]["region_id"])
            region = ds.load_region(first_id, normalize=False, use_cache=False)
            coords, source = _coordinate_frame(region, args.coordinate_mode)
            log(
                f"{name}: regions={len(metadata)}, first={first_id}, "
                f"cells={len(region.coordinates)}, markers={region.expression.shape[1]}, "
                f"coordinates={source}, root={root or '<registry default>'}"
            )
            if len(coords) != len(region.expression):
                log(f"{name}: WARNING coordinate/expression row counts differ")
        except Exception as exc:
            status = 1
            log(f"{name}: ERROR: {exc}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["check", "precompute", "benchmark"], nargs="?", default="benchmark"
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=None,
        help="Root(s) containing the configured benchmark dataset folders.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["message-passing", "domains", "combined"],
        default="combined",
    )
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument(
        "--normalization-mode", choices=["l1_norm", "sum"], default="l1_norm"
    )
    parser.add_argument(
        "--coordinate-mode", choices=["auto", "um", "native"], default="auto"
    )
    parser.add_argument(
        "--expression-transform", choices=["none", "arcsinh", "log1p"], default="arcsinh"
    )
    parser.add_argument("--arcsinh-cofactor", type=float, default=5.0)
    parser.add_argument("--n-domains", type=int, default=10)
    parser.add_argument("--max-fit-cells", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0, help="UTAG domain clustering seed.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--only-c-index",
        action="store_true",
        help="Run only survival tasks, whose primary benchmark metric is c_index.",
    )
    parser.add_argument("--max-regions", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_roots:
        args.data_roots = [str(Path(root).expanduser()) for root in args.data_roots]
    if args.debug:
        os.environ["BENCHMARK_RAISE_ERRORS"] = "1"
    if args.action == "check":
        raise SystemExit(check(args))
    if args.action == "precompute":
        precompute(args)
        return

    result = run_benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    log(f"Wrote {output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
