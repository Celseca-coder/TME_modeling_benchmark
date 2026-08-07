from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
CODE_DIR = MODEL_DIR.parents[1]
TUTORIAL_DIR = MODEL_DIR / "Tutorial"
RESULTS_DIR = MODEL_DIR.parents[1] / "model_results" / "CytoCommunity"
DEFAULT_LOCAL_INPUT_DIR = RESULTS_DIR / "local_inputs"
DEFAULT_NATIVE_RESULTS_DIR = RESULTS_DIR / "native_local_runs"


PIPELINES = {
    "unsupervised": {
        "workdir": TUTORIAL_DIR / "Unsupervised",
        "commands": [
            [sys.executable, "Step1_ConstructCellularSpatialGraphs.py"],
            [sys.executable, "Step2_TCNLearning_Unsupervised.py"],
            ["Rscript", "Step3_TCNEnsemble.R"],
            [sys.executable, "Step4_ResultVisualization.py"],
        ],
        "artifact_patterns": ["Step1_Output", "Step2_Output_*", "Step3_Output_*", "Step4_Output_*"],
    },
    "supervised": {
        "workdir": TUTORIAL_DIR / "Supervised",
        "commands": [
            [sys.executable, "Step1_ConstructCellularSpatialGraphs.py"],
            [sys.executable, "Step2_TCNLearning_Supervised.py"],
            ["Rscript", "Step3_TCNEnsemble.R"],
            [sys.executable, "Step4_ResultVisualization.py"],
        ],
        "artifact_patterns": ["Step1_Output", "Step2_Output", "Step3_Output", "Step4_Output"],
    },
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_command(command: list[str], workdir: Path) -> None:
    if command[0] == "Rscript" and shutil.which("Rscript") is None:
        raise SystemExit("Rscript is required but was not found on PATH.")
    log(f"Running command in {workdir}: {' '.join(command)}")
    subprocess.run(command, cwd=workdir, check=True)


def copy_artifacts(source_root: Path, target_root: Path, patterns: list[str]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for pattern in patterns:
        for source in sorted(source_root.glob(pattern)):
            if not source.exists():
                continue
            destination = target_root / source.name
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied.append(
                {
                    "artifact": source.name,
                    "source_path": str(source),
                    "target_path": str(destination),
                }
            )
    return copied


def write_manifest(target_root: Path, mode: str, copied: list[dict[str, str]]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest = target_root / f"{mode}_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "artifact", "source_path", "target_path"])
        writer.writeheader()
        for row in copied:
            writer.writerow({"mode": mode, **row})


def _import_benchmark_registry():
    sys.path.insert(0, str(CODE_DIR))
    from benchmark.utils.registry import list_datasets, load_dataset

    return list_datasets, load_dataset


def _safe_image_name(dataset_name: str, region_id: str, used: set[str]) -> str:
    name = f"{dataset_name}__{region_id}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not name:
        name = "region"
    base = name[:180]
    name = base
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = base[: 180 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def _load_dataset_from_roots(dataset_name: str, data_roots: list[Path]):
    _, load_dataset = _import_benchmark_registry()
    errors = []
    for root in data_roots:
        try:
            ds = load_dataset(dataset_name, data_root=root)
            if not ds._root.exists():
                raise FileNotFoundError(f"dataset root not found: {ds._root}")
            if not ds._regions_dir.exists():
                raise FileNotFoundError(f"regions dir not found: {ds._regions_dir}")
            ds.get_metadata()
            return ds, root
        except Exception as exc:
            errors.append(f"{root}: {exc}")
    raise FileNotFoundError(
        f"Could not load dataset '{dataset_name}' from any --data-roots.\n"
        + "\n".join(errors)
    )


def _target_labels_for_task(ds, task_id: str, region_ids: list[str]) -> dict[str, int]:
    task_cfg = ds.get_task_config(task_id)
    if task_cfg["type"] == "survival":
        raise ValueError(
            f"CytoCommunity tutorial GraphLabel is a class id, but task '{task_id}' is survival. "
            "Use --cyto-mode unsupervised for survival datasets or choose a classification task."
        )
    y = ds.build_target(region_ids, task_id)
    if y.dtype.kind in "biu":
        values = y.astype(int)
    else:
        cats = sorted(y.astype(str).unique().tolist())
        mapping = {label: i for i, label in enumerate(cats)}
        values = y.astype(str).map(mapping).astype(int)
    return values.to_dict()


def _replace_assignment(text: str, name: str, value: str | int | float, *, r_style: bool = False) -> str:
    if isinstance(value, str):
        rendered = f'"{value}"'
    else:
        rendered = str(value)
    op = "<-" if r_style else "="
    pattern = rf"^{re.escape(name)}\s*{re.escape(op)}\s*.*$"
    replacement = f"{name} {op} {rendered}"
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def _patch_text_file(path: Path, replacements: dict[str, str | int | float], *, r_style: bool = False) -> None:
    text = path.read_text()
    for key, value in replacements.items():
        text = _replace_assignment(text, key, value, r_style=r_style)
    path.write_text(text)


def _copy_native_unsupervised_template(run_dir: Path) -> None:
    src = TUTORIAL_DIR / "Unsupervised"
    ignore = shutil.ignore_patterns(
        "MERFISH-Brain_Input",
        "Step1_Output",
        "Step2_Output_*",
        "Step3_Output_*",
        "Step4_Output_*",
        "__pycache__",
        "*.pyc",
    )
    shutil.copytree(src, run_dir, ignore=ignore)


def _copy_selected_native_input(input_dir: Path, output_dir: Path, selected_images: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ImageNameList.txt").write_text("\n".join(selected_images) + "\n")
    optional_files = {"region_name_map.csv", "missing_regions.csv"}
    for name in optional_files:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    for image_name in selected_images:
        for suffix in ("Coordinates", "CellTypeLabel", "GraphLabel"):
            source = input_dir / f"{image_name}_{suffix}.txt"
            if source.exists():
                shutil.copy2(source, output_dir / source.name)
            elif suffix != "GraphLabel":
                raise FileNotFoundError(f"Missing required CytoCommunity input file: {source}")


def _patch_native_step1_safe_knn(path: Path) -> None:
    text = path.read_text()
    old = """    K = KNN_K
    KNNgraph_sparse = kneighbors_graph(x_y_coordinates, K, mode='connectivity', include_self=False, n_jobs=-1)  #should NOT include itself as a nearest neighbor. Checked. "-1" means using all available cores.
"""
    new = """    K = min(KNN_K, max(1, x_y_coordinates.shape[0] - 1))
    if K < KNN_K:
        print(f"Adjusted KNN_K from {KNN_K} to {K} for {region_name} with {x_y_coordinates.shape[0]} cells")
    KNNgraph_sparse = kneighbors_graph(x_y_coordinates, K, mode='connectivity', include_self=False, n_jobs=-1)  #should NOT include itself as a nearest neighbor. Checked. "-1" means using all available cores.
"""
    if old not in text:
        log(f"Warning: could not patch safe KNN block in {path}; original text pattern not found")
        return
    path.write_text(text.replace(old, new))


def _patch_native_step2_robust_rmtree(path: Path) -> None:
    text = path.read_text()
    if "def _safe_rmtree" not in text:
        text = text.replace(
            "import shutil\n",
            """import shutil
import time


def _safe_rmtree(path):
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 4:
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(0.5)

""",
            1,
        )
    text = text.replace("shutil.rmtree(RunFolderName)", "_safe_rmtree(RunFolderName)")
    path.write_text(text)


def _read_image_names(input_dir: Path) -> list[str]:
    image_list = input_dir / "ImageNameList.txt"
    if not image_list.exists():
        raise FileNotFoundError(f"Missing ImageNameList.txt: {image_list}")
    return [line.strip() for line in image_list.read_text().splitlines() if line.strip()]


def run_native_local_inputs(
    input_root: str | Path,
    results_root: str | Path,
    dataset_names: list[str] | None,
    image_names: list[str] | None,
    max_images: int | None,
    num_run: int,
    num_epoch: int,
    num_tcn: int,
    embedding_dim: int,
    loss_cutoff: float,
    overwrite: bool,
) -> Path:
    input_root = Path(input_root)
    results_root = Path(results_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    if dataset_names:
        dataset_dirs = [input_root / name for name in dataset_names]
    else:
        dataset_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and (p / "ImageNameList.txt").exists())
    if not dataset_dirs:
        raise ValueError(f"No dataset input folders found under {input_root}")

    summary_rows: list[dict[str, str | int | float]] = []
    log(f"Running native CytoCommunity on {len(dataset_dirs)} local input dataset(s)")
    for dataset_i, input_dir in enumerate(dataset_dirs, start=1):
        if not input_dir.exists():
            raise FileNotFoundError(f"Dataset input folder not found: {input_dir}")
        dataset_name = input_dir.name
        all_images = _read_image_names(input_dir)
        selected_images = image_names or all_images
        selected_images = [img for img in selected_images if img in set(all_images)]
        if max_images is not None:
            selected_images = selected_images[:max_images]
        if not selected_images:
            raise ValueError(f"No selected images found in {input_dir / 'ImageNameList.txt'}")

        run_dir = results_root / dataset_name
        if run_dir.exists():
            if not overwrite:
                log(f"[{dataset_name}] output exists, skip. Use --overwrite to rerun: {run_dir}")
                continue
            log(f"[{dataset_name}] removing existing native run directory: {run_dir}")
            shutil.rmtree(run_dir)
        log(f"[{dataset_i}/{len(dataset_dirs)}] setting up native run: {dataset_name}")
        _copy_native_unsupervised_template(run_dir)
        _copy_selected_native_input(input_dir, run_dir / "MERFISH-Brain_Input", selected_images)

        step1 = run_dir / "Step1_ConstructCellularSpatialGraphs.py"
        step2 = run_dir / "Step2_TCNLearning_Unsupervised.py"
        step3 = run_dir / "Step3_TCNEnsemble.R"
        step4 = run_dir / "Step4_ResultVisualization.py"
        _patch_text_file(step1, {"InputFolderName": "./MERFISH-Brain_Input/"})
        _patch_native_step1_safe_knn(step1)
        _patch_native_step2_robust_rmtree(step2)
        log(f"[{dataset_name}] Step1 building graphs for {len(selected_images)} image(s)")
        run_command([sys.executable, step1.name], run_dir)

        for image_i, image_name in enumerate(selected_images, start=1):
            log(f"[{dataset_name}] image {image_i}/{len(selected_images)}: {image_name}")
            _patch_text_file(
                step2,
                {
                    "InputFolderName": "./MERFISH-Brain_Input/",
                    "Image_Name": image_name,
                    "Num_TCN": num_tcn,
                    "Num_Run": num_run,
                    "Num_Epoch": num_epoch,
                    "Embedding_Dimension": embedding_dim,
                    "Loss_Cutoff": loss_cutoff,
                },
            )
            _patch_text_file(step3, {"Image_Name": image_name}, r_style=True)
            _patch_text_file(step4, {"InputFolderName": "./MERFISH-Brain_Input/", "Image_Name": image_name})

            run_command([sys.executable, step2.name], run_dir)
            run_command(["Rscript", step3.name], run_dir)
            run_command([sys.executable, step4.name], run_dir)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "image_name": image_name,
                    "run_dir": str(run_dir),
                    "num_run": num_run,
                    "num_epoch": num_epoch,
                    "num_tcn": num_tcn,
                    "embedding_dim": embedding_dim,
                    "loss_cutoff": loss_cutoff,
                }
            )
        log(f"[{dataset_name}] native run done")

    with (results_root / "native_local_runs_manifest.csv").open("w", newline="") as handle:
        fieldnames = ["dataset", "image_name", "run_dir", "num_run", "num_epoch", "num_tcn", "embedding_dim", "loss_cutoff"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    log(f"Wrote native run manifest: {results_root / 'native_local_runs_manifest.csv'}")
    return results_root


def prepare_cyto_inputs(
    dataset_names: list[str],
    data_roots: list[str],
    output_root: str | Path,
    cyto_mode: str,
    task_id: str | None,
    max_regions: int | None,
    normalize: bool,
) -> Path:
    list_datasets, _ = _import_benchmark_registry()
    if not dataset_names:
        dataset_names = list_datasets()

    roots = [Path(p).expanduser() for p in data_roots]
    if not roots:
        raise ValueError("--data-roots is required for prepare-inputs")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str | int]] = []
    log(f"Preparing CytoCommunity inputs for {len(dataset_names)} dataset(s)")
    log("Data roots: " + ", ".join(str(p) for p in roots))
    log(f"Output root: {output_root}")

    for dataset_i, dataset_name in enumerate(dataset_names, start=1):
        log(f"[{dataset_i}/{len(dataset_names)}] Loading dataset config '{dataset_name}'")
        ds, matched_root = _load_dataset_from_roots(dataset_name, roots)
        log(f"[{dataset_name}] matched data root: {matched_root}")
        meta = ds.get_metadata()
        if task_id:
            task_meta = ds.get_task_metadata(task_id)
            region_ids = task_meta["region_id"].astype(str).tolist()
        else:
            region_ids = meta["region_id"].astype(str).tolist()
        if max_regions is not None:
            region_ids = region_ids[:max_regions]
        if not region_ids:
            raise ValueError(f"No regions selected for dataset '{dataset_name}'")
        log(f"[{dataset_name}] selected {len(region_ids)} region(s)")

        labels = None
        if cyto_mode == "supervised":
            if not task_id:
                raise ValueError("--task is required when --cyto-mode supervised")
            log(f"[{dataset_name}] building GraphLabel files from task '{task_id}'")
            labels = _target_labels_for_task(ds, task_id, region_ids)

        dataset_out = output_root / dataset_name
        if dataset_out.exists():
            log(f"[{dataset_name}] removing existing output directory: {dataset_out}")
            shutil.rmtree(dataset_out)
        dataset_out.mkdir(parents=True, exist_ok=True)
        log(f"[{dataset_name}] writing CytoCommunity input files to: {dataset_out}")

        used_names: set[str] = set()
        image_names: list[str] = []
        map_rows: list[dict[str, str | int]] = []
        missing_rows: list[dict[str, str]] = []
        for region_i, region_id in enumerate(region_ids, start=1):
            if region_i == 1 or region_i % 25 == 0 or region_i == len(region_ids):
                log(f"[{dataset_name}] exporting region {region_i}/{len(region_ids)}: {region_id}")
            try:
                region = ds.load_region(region_id, normalize=normalize, use_cache=False)
            except FileNotFoundError as exc:
                log(f"[{dataset_name}] missing region, skip: {region_id} ({exc})")
                missing_rows.append(
                    {
                        "dataset_config": dataset_name,
                        "region_id": region_id,
                        "error": str(exc),
                    }
                )
                continue
            image_name = _safe_image_name(dataset_name, region_id, used_names)
            image_names.append(image_name)

            coord_source = "coordinates_um" if hasattr(region, "coordinates_um") else "coordinates"
            coord_table = getattr(region, "coordinates_um", region.coordinates)
            coords = coord_table[["x", "y"]]
            cell_type_col = ds.config.get("cell_type_col", "cell_type")
            if cell_type_col not in region.cell_types.columns and "cell_type" in region.cell_types.columns:
                cell_type_col = "cell_type"
            cell_types = region.cell_types[cell_type_col].reindex(region.coordinates.index)

            coords.to_csv(dataset_out / f"{image_name}_Coordinates.txt", sep="\t", header=False, index=False)
            cell_types.to_csv(dataset_out / f"{image_name}_CellTypeLabel.txt", sep="\t", header=False, index=False)
            graph_label = ""
            if labels is not None:
                graph_label = int(labels[region_id])
                (dataset_out / f"{image_name}_GraphLabel.txt").write_text(f"{graph_label}\n")

            map_rows.append(
                {
                    "dataset_config": dataset_name,
                    "dataset_name": ds.name,
                    "matched_data_root": str(matched_root),
                    "region_id": region_id,
                    "image_name": image_name,
                    "n_cells": len(region.coordinates),
                    "cell_type_col": cell_type_col,
                    "coord_source": coord_source,
                    "graph_label": graph_label,
                }
            )

        (dataset_out / "ImageNameList.txt").write_text("\n".join(image_names) + "\n")
        if missing_rows:
            with (dataset_out / "missing_regions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_config", "region_id", "error"])
                writer.writeheader()
                writer.writerows(missing_rows)
            log(f"[{dataset_name}] skipped {len(missing_rows)} missing region(s); see {dataset_out / 'missing_regions.csv'}")
        if not image_names:
            raise ValueError(
                f"No existing region directories could be exported for dataset '{dataset_name}'. "
                f"See {dataset_out / 'missing_regions.csv'}"
            )
        with (dataset_out / "region_name_map.csv").open("w", newline="") as handle:
            fieldnames = [
                "dataset_config",
                "dataset_name",
                "matched_data_root",
                "region_id",
                "image_name",
                "n_cells",
                "cell_type_col",
                "coord_source",
                "graph_label",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(map_rows)

        ds.clear_region_cache()
        log(f"[{dataset_name}] done")
        summary_rows.append(
            {
                "dataset_config": dataset_name,
                "dataset_name": ds.name,
                "matched_data_root": str(matched_root),
                "output_dir": str(dataset_out),
                "n_regions": len(image_names),
                "cyto_mode": cyto_mode,
                "task": task_id or "",
            }
        )

    with (output_root / "prepare_inputs_manifest.csv").open("w", newline="") as handle:
        fieldnames = ["dataset_config", "dataset_name", "matched_data_root", "output_dir", "n_regions", "cyto_mode", "task"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    log("Finished preparing CytoCommunity inputs")
    return output_root


def run_pipeline(mode: str, results_root: Path) -> Path:
    config = PIPELINES[mode]
    workdir = config["workdir"]
    if not workdir.exists():
        raise FileNotFoundError(f"Missing tutorial directory: {workdir}")

    for command in config["commands"]:
        run_command(command, workdir)

    mode_results_dir = results_root / mode
    copied = copy_artifacts(workdir, mode_results_dir, config["artifact_patterns"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (mode_results_dir / "run_timestamp.txt").write_text(f"mode={mode}\nrun_at={timestamp}\n")
    write_manifest(mode_results_dir, mode, copied)
    return mode_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local CytoCommunity tutorial pipelines and sync artifacts to model_results."
    )
    parser.add_argument(
        "--action",
        choices=["run-tutorial", "prepare-inputs", "run-native-local"],
        default="run-tutorial",
        help="Run bundled tutorial, export benchmark data, or run native CytoCommunity on exported local inputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["unsupervised", "supervised", "all"],
        default="all",
        help="Which tutorial pipeline to run when --action run-tutorial.",
    )
    parser.add_argument(
        "--results-root",
        default=str(RESULTS_DIR),
        help="Destination root under model_results.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset config stems to export, e.g. bc_jackson2020 hnc_wu2022. Defaults to all configs.",
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=None,
        help="One or more total roots that contain dataset folders, e.g. /data/TME_benchmark_data.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_LOCAL_INPUT_DIR),
        help="Where to write exported CytoCommunity input folders when --action prepare-inputs.",
    )
    parser.add_argument(
        "--cyto-mode",
        choices=["unsupervised", "supervised"],
        default="unsupervised",
        help="Whether to write only coordinates/cell types, or also per-region GraphLabel files.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Classification task id used to build GraphLabel files for --cyto-mode supervised.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help="Optional cap for quick local export tests.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Load normalized expression through TMEDataset. Coordinates/cell types are unaffected; default is raw.",
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_LOCAL_INPUT_DIR),
        help="Root of exported local input folders for --action run-native-local.",
    )
    parser.add_argument(
        "--native-results-root",
        default=str(DEFAULT_NATIVE_RESULTS_DIR),
        help="Where native local CytoCommunity runs are written.",
    )
    parser.add_argument(
        "--image-names",
        nargs="*",
        default=None,
        help="Optional image names to run within each dataset input folder. Defaults to ImageNameList order.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on images per dataset for --action run-native-local.",
    )
    parser.add_argument("--native-num-run", type=int, default=20)
    parser.add_argument("--native-num-epoch", type=int, default=3000)
    parser.add_argument("--native-num-tcn", type=int, default=9)
    parser.add_argument("--native-embedding-dim", type=int, default=128)
    parser.add_argument("--native-loss-cutoff", type=float, default=-0.6)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing native local run directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare-inputs":
        out = prepare_cyto_inputs(
            dataset_names=args.datasets or [],
            data_roots=args.data_roots or [],
            output_root=args.output_root,
            cyto_mode=args.cyto_mode,
            task_id=args.task,
            max_regions=args.max_regions,
            normalize=args.normalize,
        )
        print(f"Wrote CytoCommunity local input folders under {out}")
        return
    if args.action == "run-native-local":
        out = run_native_local_inputs(
            input_root=args.input_root,
            results_root=args.native_results_root,
            dataset_names=args.datasets,
            image_names=args.image_names,
            max_images=args.max_images,
            num_run=args.native_num_run,
            num_epoch=args.native_num_epoch,
            num_tcn=args.native_num_tcn,
            embedding_dim=args.native_embedding_dim,
            loss_cutoff=args.native_loss_cutoff,
            overwrite=args.overwrite,
        )
        print(f"Wrote native CytoCommunity local runs under {out}")
        return

    results_root = Path(args.results_root)

    modes = [args.mode] if args.mode != "all" else list(PIPELINES)
    for mode in modes:
        out_dir = run_pipeline(mode, results_root)
        print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
