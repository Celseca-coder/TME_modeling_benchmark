#!/usr/bin/env python
"""Attention and RF MDI on a no-direction pack.

``--pack 1in3`` uses PseudoLabelDataset1in3NoDirect: both outer thirds are 1
and the middle third is 0. ``--pack 1in5`` uses PseudoLabelDataset1in5NoDirect:
both outer fifths are 1 and the middle three fifths are 0. Within-region
AUROC compares extremes against the intermediate band. Precision@20 uses that
same label.

Features stay the joint composition+density+mixing+point_pattern panel.
Attention trains on the region label Y_r. RF trains on the window label y_w
and scores P(y=1|x). Patient 5-fold and the pass thresholds match the other
localisation runs.

    python scripts/eval_nodirect_1in3_window_importance.py --pack 1in3 --method attention rf --label noisy
    python scripts/eval_nodirect_1in3_window_importance.py --pack 1in5 --method attention rf --label noisy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.features.local_window import combo_name  # noqa: E402
from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES  # noqa: E402
from benchmark.motifs.window_extractor_eval import join_features_and_labels  # noqa: E402
from scripts.eval_extreme20_window_importance import (  # noqa: E402
    _checkpoints_ready,
    _drop_task,
    _resume,
    _row,
    _save,
    _scores_ready,
    _store_pair,
    checkpoint_dir,
    write_wide,
)

PACKS = {
    "1in3": Path("/autofs/nas8/tywang/tjzou/PseudoLabelDataset1in3NoDirect"),
    "1in5": Path("/autofs/nas8/tywang/tjzou/PseudoLabelDataset1in5NoDirect"),
}
FEATURE_DIR = Path(
    "/autofs/nas8/tywang/tjzou/TME_modeling_benchmark/results/pseudo_labels/mil_windows/features"
)
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
METHODS = ("attention", "rf")


def _out_dir(pack: str, label: str) -> Path:
    name = f"nodirect_{pack}" if label == "clean" else f"nodirect_{pack}_noisy"
    return _CODE / "results" / "pseudo_labels" / "mil_windows" / name


def _label_paths(pack: str, dataset: str, label: str) -> tuple[Path, Path]:
    root = PACKS[pack]
    window_dir = root / "window" / "per_dataset"
    bag_dir = root / "region" / "per_dataset"
    if label == "clean":
        return window_dir / f"{dataset}_windows.csv", bag_dir / f"{dataset}_v2.csv"
    if label == "noisy":
        return window_dir / f"{dataset}_windows_noisy.csv", bag_dir / f"{dataset}_v2_noisy.csv"
    raise ValueError(f"unknown label cut: {label}")


def _load(pack: str, dataset: str, label: str, feature_dir: Path):
    win_path, bag_path = _label_paths(pack, dataset, label)
    feats = pd.read_csv(feature_dir / f"{dataset}_panel.csv")
    wins = pd.read_csv(win_path)
    bags = pd.read_csv(bag_path).drop_duplicates("region_id")
    bags["region_id"] = bags["region_id"].astype(str)
    return feats, wins, bags


def _tasks(datasets: list[str] | None, motif: str) -> list[tuple[str, str]]:
    pairs = list(SELECTED_MOTIF_SOURCES)
    if datasets:
        wanted = set(datasets)
        pairs = [(ds, m) for ds, m in pairs if ds in wanted]
    if motif:
        pairs = [(ds, m) for ds, m in pairs if m == motif]
    return pairs


def _prepare(pack: str, dataset: str, motif: str, label: str, feature_dir: Path):
    feats, wins, bags = _load(pack, dataset, label, feature_dir)
    tab = join_features_and_labels(feats, wins, motif, bags)
    y_r = pd.to_numeric(bags.set_index("region_id")[f"{motif}_label_v2"], errors="coerce")
    return tab, y_r


def run_attention(jobs, group, pack, label, feature_dir, dest: Path) -> None:
    from scripts.eval_ig_mil_window_importance import evaluate_gated_localisation

    rows, done = _resume(dest)
    by_ds: dict[str, list[str]] = {}
    for dataset, motif in jobs:
        by_ds.setdefault(dataset, []).append(motif)
    for dataset, motifs in by_ds.items():
        feats, wins, bags = _load(pack, dataset, label, feature_dir)
        y_index = bags.set_index("region_id")
        for motif in motifs:
            if _scores_ready(rows, dest, dataset, motif, "attention"):
                print(f"skip attention {dataset} {motif}", flush=True)
                continue
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(y_index[f"{motif}_label_v2"], errors="coerce")
            folds = evaluate_gated_localisation(
                tab, y_r, DEFAULT_FAMILIES, methods=("attention",),
                checkpoint_dir=checkpoint_dir(dest, dataset, motif, "attention"),
            )
            rows, rec, loo_rec = _store_pair(
                rows, dataset, motif, group, tab, "attention", folds["attention"], folds["attention_loo"],
            )
            done.add((dataset, motif, "attention"))
            done.add((dataset, motif, "attention_loo"))
            _save(dest, rows)
            print(rec, flush=True)
            print(loo_rec, flush=True)


def run_rf(jobs, group, pack, label, feature_dir, dest: Path) -> None:
    from scripts.eval_rf_mdi_window_importance import evaluate_rf_mdi

    rows, done = _resume(dest)
    for dataset, motif in jobs:
        if _checkpoints_ready(rows, dest, dataset, motif, "rf_mdi"):
            print(f"skip rf {dataset} {motif}", flush=True)
            continue
        tab, _y_r = _prepare(pack, dataset, motif, label, feature_dir)
        fold = evaluate_rf_mdi(
            tab, DEFAULT_FAMILIES,
            checkpoint_dir=checkpoint_dir(dest, dataset, motif, "rf_mdi"),
        )
        rec = _row(dataset, motif, group, "rf_mdi", tab, fold)
        rows = _drop_task(rows, dataset, motif, "rf_mdi")
        rows.append(rec)
        done.add((dataset, motif, "rf_mdi"))
        _save(dest, rows)
        print(rec, flush=True)


def assemble(out_dir: Path) -> None:
    frames = []
    for name in ("attention", "rf"):
        path = out_dir / f"{name}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no method tables in {out_dir}")
    long_df = pd.concat(frames, ignore_index=True)
    long_path = out_dir / "localisation.csv"
    long_df.to_csv(long_path, index=False)
    wide_path = out_dir / "four_family_localisation_wide.csv"
    write_wide(long_df, wide_path)
    print(f"wrote {long_path}", flush=True)
    print(f"wrote {wide_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=sorted(PACKS), required=True)
    parser.add_argument("--method", nargs="+", choices=list(METHODS), required=True)
    parser.add_argument("--label", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--motif", default="")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    args = parser.parse_args()

    families = tuple(args.families)
    if families != DEFAULT_FAMILIES:
        raise SystemExit("this driver keeps the joint four-family panel")
    group = combo_name(families)
    out_dir = _out_dir(args.pack, args.label)
    jobs = _tasks(args.datasets, args.motif)
    if not jobs:
        raise SystemExit("no tasks selected")
    feature_dir = Path(args.feature_dir)
    runners = {"attention": run_attention, "rf": run_rf}
    chosen = [method for method in METHODS if method in args.method]
    for method in chosen:
        print(
            f"=== {method} pack={args.pack} label={args.label} families={group} ===",
            flush=True,
        )
        runners[method](jobs, group, args.pack, args.label, feature_dir, out_dir / f"{method}.csv")
    if set(chosen) == set(METHODS):
        assemble(out_dir)


if __name__ == "__main__":
    main()
