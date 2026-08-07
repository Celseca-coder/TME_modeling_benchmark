from __future__ import annotations

"""Batch-process local benchmark data with Cell-Graph Signature.

The interface mirrors ``models/Eva/process_local_data.py``:

* ``check`` verifies configured datasets and region files;
* ``precompute`` exports one graph bundle per region plus a CSV manifest;
* ``benchmark`` runs the repository's patient-level CV and cohort tests.

The precomputed bundles are useful for inspection and external/native runs.
The benchmark deliberately rebuilds graphs inside each training fold so marker
selection and scaling cannot leak from validation/test data.

Examples
--------
    cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
    python models/Cell-Graph_Signature/process_local_data.py check \
        --data-roots /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
    python models/Cell-Graph_Signature/process_local_data.py precompute \
        --datasets bc_jackson2020 \
        --data-roots /autofs/nas8/tywang/tjzou/TME_modeling_benchmark \
        --max-regions 10
    python models/Cell-Graph_Signature/process_local_data.py benchmark \
        --datasets bc_jackson2020 \
        --data-roots /autofs/nas8/tywang/tjzou/TME_modeling_benchmark \
        --seeds 0 --device cuda
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
CODE_DIR = MODEL_DIR.parents[1]
DEFAULT_RESULTS = CODE_DIR / "model_results" / "Cell-Graph_Signature"
DEFAULT_CACHE = DEFAULT_RESULTS / "graphs"
DEFAULT_OUTPUT = CODE_DIR / "results" / "cell_graph_signature_benchmark.csv"

sys.path.insert(0, str(CODE_DIR))

from benchmark.utils.registry import list_datasets, load_dataset  # noqa: E402
from benchmark.utils.task_filter import should_skip_benchmark_task  # noqa: E402
from benchmark.validation import (  # noqa: E402
    PRIMARY_METRIC,
    cohort_split_test,
    cross_validate,
    summarize_folds,
)


def log(message: str) -> None:
    print(message, flush=True)


def _graph_components():
    """Import optional GNN dependencies only for actions that need them."""
    try:
        from benchmark.features.cell_graph_signature import CellGraphSignatureBuilder
        from benchmark.models.cell_graph_signature import (
            CellGraphSignatureClassifier,
            CellGraphSignatureCox,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "torch_geometric"}:
            raise RuntimeError(
                "Cell-Graph Signature requires PyTorch and torch-geometric; "
                "install the packages in requirements.txt first"
            ) from exc
        raise
    return CellGraphSignatureBuilder, CellGraphSignatureClassifier, CellGraphSignatureCox


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
    raise FileNotFoundError(
        f"Could not load {name} from any data root:\n" + "\n".join(errors)
    )


def _roots(args) -> list[str] | None:
    return args.data_roots or ([args.data_root] if args.data_root else None)


def _region_is_complete(ds, region_id: str) -> bool:
    directory = ds.region_dir(region_id)
    return directory.is_dir() and all(
        (directory / filename).is_file()
        for filename in ("coordinates.csv", "expression.csv", "cell_types.csv")
    )


def _region_inventory(ds) -> tuple[list[str], list[str], list[str]]:
    """Return metadata IDs split into usable/missing, plus unreferenced folders."""
    metadata_ids = ds.get_metadata()["region_id"].astype(str).drop_duplicates().tolist()
    usable = [region_id for region_id in metadata_ids if _region_is_complete(ds, region_id)]
    missing = [region_id for region_id in metadata_ids if region_id not in set(usable)]
    metadata_set = set(metadata_ids)
    unreferenced = (
        sorted(path.name for path in ds._regions_dir.iterdir()
               if path.is_dir() and path.name not in metadata_set)
        if ds._regions_dir.is_dir() else []
    )
    return usable, missing, unreferenced


def _selected_region_ids(
    ds, max_regions: int | None, *, require_existing: bool = False
) -> list[str]:
    if require_existing:
        ids, _, _ = _region_inventory(ds)
    else:
        ids = ds.get_metadata()["region_id"].astype(str).drop_duplicates().tolist()
    return ids[:max_regions] if max_regions is not None else ids


def _limit_dataset(ds, max_regions: int | None) -> None:
    # Validation must never attempt metadata rows whose region files are absent.
    allowed = set(_selected_region_ids(ds, max_regions, require_existing=True))
    if not allowed:
        raise FileNotFoundError(
            f"No usable region directories match metadata under {ds._regions_dir}. "
            "Expected one directory per metadata region_id, each containing "
            "coordinates.csv, expression.csv, and cell_types.csv."
        )
    original = ds.get_task_metadata
    ds.get_task_metadata = lambda task, _original=original: _original(task).loc[
        lambda frame: frame["region_id"].astype(str).isin(allowed)
    ].reset_index(drop=True)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def check(args) -> int:
    torch_ok = importlib.util.find_spec("torch") is not None
    pyg_ok = importlib.util.find_spec("torch_geometric") is not None
    log(f"Cell-Graph Signature code: {MODEL_DIR}")
    log(f"PyTorch: {'OK' if torch_ok else 'MISSING'}")
    log(f"PyTorch Geometric: {'OK' if pyg_ok else 'MISSING'}")
    had_error = False
    for name in args.datasets or list_datasets():
        try:
            ds, matched_root = _load_dataset_from_roots(name, _roots(args))
            ids = _selected_region_ids(ds, args.max_regions)
            complete = 0
            missing = 0
            cells = 0
            for region_id in ids:
                directory = ds.region_dir(region_id)
                required = [
                    directory / "coordinates.csv",
                    directory / "expression.csv",
                    directory / "cell_types.csv",
                ]
                if all(path.exists() for path in required):
                    complete += 1
                    try:
                        with required[0].open(encoding="utf-8") as handle:
                            cells += max(0, sum(1 for _ in handle) - 1)
                    except (OSError, UnicodeError):
                        pass
                else:
                    missing += 1
            log(
                f"{name}: regions={len(ids)}, complete={complete}, missing={missing}, "
                f"cells={cells}, root={ds._root}, data_root={matched_root}"
            )
            ds.clear_region_cache()
        except Exception as exc:
            had_error = True
            log(f"{name}: ERROR: {exc}")
            if args.debug:
                raise
    return int(had_error or not (torch_ok and pyg_ok))


def precompute(args) -> Path:
    """Write graph bundles and a manifest for all selected local regions."""
    import torch

    Builder, _, _ = _graph_components()
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for name in args.datasets or list_datasets():
        ds, matched_root = _load_dataset_from_roots(name, _roots(args))
        usable, missing, unreferenced = _region_inventory(ds)
        ids = usable[:args.max_regions] if args.max_regions is not None else usable
        log(
            f"[{name}] metadata={len(usable) + len(missing)}, usable={len(usable)}, "
            f"missing={len(missing)}, unreferenced_dirs={len(unreferenced)}"
        )
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            log(f"[{name}] skipping missing/incomplete metadata regions: {preview}{suffix}")
        if unreferenced:
            preview = ", ".join(unreferenced[:5])
            suffix = " ..." if len(unreferenced) > 5 else ""
            log(f"[{name}] region folders absent from metadata: {preview}{suffix}")
        if not ids:
            raise FileNotFoundError(
                f"[{name}] no usable metadata/region-directory matches under "
                f"{ds._regions_dir}. Check --data-roots and whether directory names "
                "match metadata.region_id."
            )
        log(
            f"[{name}] loading {len(ids)} existing region(s); "
            f"data_root={matched_root}"
        )

        loaded = []
        load_errors: dict[str, str] = {}
        for index, region_id in enumerate(ids, 1):
            try:
                loaded.append(ds.load_region(
                    region_id, normalize=not args.raw_expression, use_cache=False
                ))
            except Exception as exc:
                load_errors[region_id] = str(exc)
                log(f"[{name}] load {index}/{len(ids)} ERROR {region_id}: {exc}")
                if args.debug:
                    raise

        dataset_dir = cache_root / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if loaded:
            builder = Builder(graph_size=args.graph_size, radius_um=args.radius_um).fit(loaded)
            (dataset_dir / "markers.json").write_text(
                json.dumps(list(builder.markers), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for index, region in enumerate(loaded, 1):
                output = dataset_dir / f"{_safe_name(region.region_id)}.pt"
                try:
                    if output.exists() and not args.overwrite:
                        status = "cached"
                        payload = torch.load(output, map_location="cpu", weights_only=False)
                        graphs = payload["graphs"]
                    else:
                        graphs = builder.extract_region(region)["graphs"]
                        torch.save(
                            {
                                "dataset": name,
                                "region_id": region.region_id,
                                "markers": list(builder.markers),
                                "graph_size": args.graph_size,
                                "radius_um": args.radius_um,
                                "normalized": not args.raw_expression,
                                "graphs": graphs,
                            },
                            output,
                        )
                        status = "ok"
                    manifest_rows.append({
                        "dataset": name,
                        "region_id": region.region_id,
                        "status": status,
                        "n_cells": region.n_cells,
                        "n_graphs": len(graphs),
                        "n_markers": builder.n_markers,
                        "path": str(output),
                        "error": "",
                    })
                    log(f"[{name}] graph {index}/{len(loaded)} {status.upper()} {region.region_id}")
                except Exception as exc:
                    manifest_rows.append({
                        "dataset": name, "region_id": region.region_id,
                        "status": "error", "n_cells": region.n_cells,
                        "n_graphs": "", "n_markers": builder.n_markers,
                        "path": str(output), "error": str(exc),
                    })
                    log(f"[{name}] graph {index}/{len(loaded)} ERROR {region.region_id}: {exc}")
                    if args.debug:
                        raise

        for region_id, error in load_errors.items():
            manifest_rows.append({
                "dataset": name, "region_id": region_id, "status": "error",
                "n_cells": "", "n_graphs": "", "n_markers": "",
                "path": "", "error": error,
            })
        ds.clear_region_cache()

    manifest = Path(args.results_root) / "precompute_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "region_id", "status", "n_cells", "n_graphs",
        "n_markers", "path", "error",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (manifest.parent / "precompute_timestamp.txt").write_text(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8"
    )
    log(f"Wrote {manifest} ({len(manifest_rows)} rows)")
    return manifest


def run_benchmark(args) -> pd.DataFrame:
    Builder, Classifier, Cox = _graph_components()

    def model_factory(task_cfg, seed):
        common = dict(
            seed=seed,
            hidden_dim=args.hidden_dim,
            pooling_ratio=args.pooling_ratio,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
        )
        return Cox(**common) if task_cfg["type"] == "survival" else Classifier(**common)

    rows = []
    for name in args.datasets or list_datasets():
        ds, matched_root = _load_dataset_from_roots(name, _roots(args))
        _limit_dataset(ds, args.max_regions)
        log(f"=== {name} (data_root={matched_root or '<registry default>'}) ===")

        def make_builder():
            return Builder(graph_size=args.graph_size, radius_um=args.radius_um)

        for task in ds.task_ids:
            if args.tasks and task not in args.tasks:
                continue
            if args.schemes and "cv" not in args.schemes:
                continue
            metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
            if (not args.include_skipped
                    and should_skip_benchmark_task(name, task, "cv", metric)):
                log(f"  {task} cv {metric}: skipped")
                continue
            result = cross_validate(
                ds, task, make_builder, model_factory,
                seeds=args.seeds, normalize=not args.raw_expression,
            )
            mean, sd = summarize_folds(result, metric)
            rows.append(dict(
                dataset=name, task=task, scheme="cv", metric=metric,
                mean=mean, sd=sd, n=len(result),
            ))
            log(f"  {task} cv {metric}: {mean:.4f} +/- {sd:.4f}")

        if args.max_regions is None:
            for test in ds.validation_config.get("generalization_tests", []):
                if args.schemes and test["name"] not in args.schemes:
                    continue
                for task in test.get("tasks", ds.task_ids):
                    if args.tasks and task not in args.tasks:
                        continue
                    metric = PRIMARY_METRIC[ds.get_task_config(task)["type"]]
                    if (not args.include_skipped
                            and should_skip_benchmark_task(
                                name, task, test["name"], metric
                            )):
                        continue
                    result = cohort_split_test(
                        ds, task, test, make_builder, model_factory,
                        seeds=args.seeds, normalize=not args.raw_expression,
                    )
                    if not result:
                        continue
                    mean, sd = summarize_folds(result, metric)
                    rows.append(dict(
                        dataset=name, task=task, scheme=test["name"], metric=metric,
                        mean=mean, sd=sd, n=len(result),
                    ))
                    log(f"  {task} {test['name']} {metric}: {mean:.4f} +/- {sd:.4f}")
        ds.clear_region_cache()

    result = pd.DataFrame(rows)
    if len(result):
        result["score"] = result.apply(
            lambda row: f"{row['mean']:.3f} +/- {row['sd']:.3f}", axis=1
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "action", choices=["check", "precompute", "benchmark"],
        nargs="?", default="benchmark",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Only run these task IDs, for example immunotherapy_response.",
    )
    parser.add_argument(
        "--schemes", nargs="*", default=None,
        help="Only run these schemes, for example cv or Yale_to_YaleExt.",
    )
    parser.add_argument(
        "--include-skipped", action="store_true",
        help="Run entries listed in benchmark.utils.task_filter instead of skipping them.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--data-roots", nargs="+", default=None,
        help="Data roots tried in order; each contains configured dataset folders.",
    )
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--graph-size", type=int, default=100)
    parser.add_argument("--radius-um", type=float, default=20.0)
    parser.add_argument("--raw-expression", action="store_true")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--pooling-ratio", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
