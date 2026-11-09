#!/usr/bin/env python
"""Window-importance extraction on PseudoLabelDatasetExtreme20.

Patient 5-fold, joint composition+density+mixing+point_pattern features,
AUROC and precision among the top 20% scored windows.

L1 MIL, gated attention, window MLP, and Additive MIL train on the region
label Y_r. RF trains on the previously calculated window label y_w and
scores each window with P(y=1|x). Window metrics always use y_w.

    python scripts/eval_extreme20_window_importance.py --method all
    python scripts/eval_extreme20_window_importance.py --method lasso
    python scripts/eval_extreme20_window_importance.py --method assemble
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
from benchmark.motifs.checkpoints import score_tag_matches  # noqa: E402
from benchmark.motifs.window_extractor_eval import (  # noqa: E402
    PASS_PRECISION_AT,
    PASS_WINDOW,
    join_features_and_labels,
)

EXTREME_ROOT = Path("/autofs/nas8/tywang/tjzou/PseudoLabelDatasetExtreme20")
WINDOW_DIR = EXTREME_ROOT / "window" / "per_dataset"
BAG_DIR = EXTREME_ROOT / "region" / "per_dataset"
FEATURE_DIR = Path(
    "/autofs/nas8/tywang/tjzou/TME_modeling_benchmark/results/pseudo_labels/mil_windows/features"
)
DEFAULT_FAMILIES = ("composition", "density", "mixing", "point_pattern")
METHODS = ("lasso", "gated", "mlp", "rf", "additive")

# Wide-table column order. Each gradient method reports the full-window score and LOO.
METHOD_SPECS = (
    ("lasso", "window L1 MIL", "Lasso"),
    ("lasso_loo", "window L1 MIL", "Lasso LOO"),
    ("attention", "Gated Attention MIL", "attention"),
    ("attention_loo", "Gated Attention MIL", "attention LOO"),
    ("mlp", "window MLP MIL", "MLP"),
    ("mlp_loo", "window MLP MIL", "MLP LOO"),
    ("rf_mdi", "window RF MIL", "RF"),
    ("additive_mil", "Additive MIL", "Additive MIL"),
    ("additive_loo", "Additive MIL", "Additive MIL LOO"),
)
LOO_METHOD = {
    "lasso": "lasso_loo",
    "attention": "attention_loo",
    "mlp": "mlp_loo",
    "additive_mil": "additive_loo",
}


def _out_dir(label: str) -> Path:
    name = "extreme20" if label == "clean" else "extreme20_noisy"
    return _CODE / "results" / "pseudo_labels" / "mil_windows" / name


def _label_paths(dataset: str, label: str) -> tuple[Path, Path]:
    if label == "clean":
        return WINDOW_DIR / f"{dataset}_windows.csv", BAG_DIR / f"{dataset}_v2.csv"
    if label == "noisy":
        return WINDOW_DIR / f"{dataset}_windows_noisy.csv", BAG_DIR / f"{dataset}_v2_noisy.csv"
    raise ValueError(f"unknown label cut: {label}")


def _load(dataset: str, label: str, feature_dir: Path):
    win_path, bag_path = _label_paths(dataset, label)
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


def _fold_metrics(fold: pd.DataFrame) -> dict:
    if fold is None or fold.empty or "auroc_window_mean" not in fold.columns:
        return {
            "auroc_window_mean": float("nan"),
            "precision_at_20": float("nan"),
            "n_regions_scored": 0,
            "n_folds": 0,
        }
    if "n_regions_scored" in fold.columns:
        n_regions = int(fold["n_regions_scored"].sum())
    else:
        n_regions = int(fold["n_regions_topk"].sum())
    return {
        "auroc_window_mean": float(fold["auroc_window_mean"].mean()),
        "precision_at_20": float(fold["precision_at_20"].mean()),
        "n_regions_scored": n_regions,
        "n_folds": int(len(fold)),
    }


def _row(dataset: str, motif: str, group: str, method: str, tab: pd.DataFrame, fold: pd.DataFrame) -> dict:
    metrics = _fold_metrics(fold)
    auroc = metrics["auroc_window_mean"]
    prec = metrics["precision_at_20"]
    return {
        "dataset": dataset,
        "motif": motif,
        "feature_group": group,
        "method": method,
        "n_windows": int(len(tab)),
        "n_labeled": int(tab["y_w"].isin([0.0, 1.0]).sum()),
        **metrics,
        "pass_window": bool(auroc == auroc and auroc >= PASS_WINDOW),
        "pass_precision_at_20": bool(prec == prec and prec >= PASS_PRECISION_AT),
    }


def _resume(path: Path) -> tuple[list[dict], set[tuple[str, str, str]]]:
    if not path.exists():
        return [], set()
    old = pd.read_csv(path)
    rows = old.to_dict("records")
    done = {(str(r["dataset"]), str(r["motif"]), str(r["method"])) for r in rows}
    return rows, done


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def checkpoint_dir(dest: Path, dataset: str, motif: str, method: str) -> Path:
    """``<pack>/checkpoints/<dataset>/<motif>/<method>/fold{k}.pt``."""
    return dest.parent / "checkpoints" / str(dataset) / str(motif) / method


def _drop_task(rows: list[dict], dataset: str, motif: str, method: str) -> list[dict]:
    return [
        row for row in rows
        if not (str(row["dataset"]) == str(dataset) and str(row["motif"]) == str(motif) and str(row["method"]) == method)
    ]


def _row_recorded(rows: list[dict], dataset: str, motif: str, method: str) -> bool:
    return any(
        str(row["dataset"]) == str(dataset) and str(row["motif"]) == str(motif) and str(row["method"]) == method
        for row in rows
    )


def _scores_ready(rows: list[dict], dest: Path, dataset: str, motif: str, method: str) -> bool:
    """A model is finished only when its current weights and the LOO row both exist."""
    if not _checkpoints_ready(rows, dest, dataset, motif, method):
        return False
    if not score_tag_matches(checkpoint_dir(dest, dataset, motif, method)):
        return False
    loo_method = LOO_METHOD.get(method)
    if loo_method is None:
        return True
    return _row_recorded(rows, dataset, motif, loo_method)


def _store_pair(
    rows: list[dict],
    dataset: str,
    motif: str,
    group: str,
    tab: pd.DataFrame,
    method: str,
    fold: pd.DataFrame,
    loo_fold: pd.DataFrame,
    legacy: tuple[str, ...] = (),
) -> tuple[list[dict], dict, dict]:
    loo_method = LOO_METHOD[method]
    for name in (method, loo_method, *legacy):
        rows = _drop_task(rows, dataset, motif, name)
    full = _row(dataset, motif, group, method, tab, fold)
    loo = _row(dataset, motif, group, loo_method, tab, loo_fold)
    rows.extend([full, loo])
    return rows, full, loo


def _checkpoints_ready(rows: list[dict], dest: Path, dataset: str, motif: str, method: str) -> bool:
    """A finished CSV row still needs a retrain when its fold files are missing."""
    matches = [
        row for row in rows
        if str(row["dataset"]) == str(dataset) and str(row["motif"]) == str(motif) and str(row["method"]) == method
    ]
    if not matches:
        return False
    n_folds = int(matches[-1].get("n_folds") or 0)
    folder = checkpoint_dir(dest, dataset, motif, method)
    if n_folds <= 0 or not folder.is_dir():
        return False
    return len(list(folder.glob("fold*.pt"))) >= n_folds


def _prepare(dataset: str, motif: str, label: str, feature_dir: Path):
    feats, wins, bags = _load(dataset, label, feature_dir)
    tab = join_features_and_labels(feats, wins, motif, bags)
    y_r = pd.to_numeric(bags.set_index("region_id")[f"{motif}_label_v2"], errors="coerce")
    return tab, y_r


def run_lasso(jobs, group, label, feature_dir, dest: Path) -> None:
    from benchmark.motifs.window_extractor_eval import evaluate_table

    rows, done = _resume(dest)
    for dataset, motif in jobs:
        if _scores_ready(rows, dest, dataset, motif, "lasso"):
            print(f"skip lasso {dataset} {motif}", flush=True)
            continue
        tab, y_r = _prepare(dataset, motif, label, feature_dir)
        fold, loo = evaluate_table(
            tab, y_r, families=DEFAULT_FAMILIES,
            checkpoint_dir=checkpoint_dir(dest, dataset, motif, "lasso"),
            with_loo=True,
        )
        rows, rec, loo_rec = _store_pair(rows, dataset, motif, group, tab, "lasso", fold, loo)
        done.add((dataset, motif, "lasso"))
        done.add((dataset, motif, "lasso_loo"))
        _save(dest, rows)
        print(rec, flush=True)
        print(loo_rec, flush=True)


def run_gated(jobs, group, label, feature_dir, dest: Path) -> None:
    from scripts.eval_ig_mil_window_importance import evaluate_gated_localisation

    rows, done = _resume(dest)
    by_ds: dict[str, list[str]] = {}
    for dataset, motif in jobs:
        by_ds.setdefault(dataset, []).append(motif)
    for dataset, motifs in by_ds.items():
        feats, wins, bags = _load(dataset, label, feature_dir)
        y_index = bags.set_index("region_id")
        for motif in motifs:
            if _scores_ready(rows, dest, dataset, motif, "attention"):
                print(f"skip gated {dataset} {motif}", flush=True)
                continue
            tab = join_features_and_labels(feats, wins, motif, bags)
            y_r = pd.to_numeric(y_index[f"{motif}_label_v2"], errors="coerce")
            folds = evaluate_gated_localisation(
                tab, y_r, DEFAULT_FAMILIES,
                checkpoint_dir=checkpoint_dir(dest, dataset, motif, "attention"),
            )
            rows, rec, loo_rec = _store_pair(
                rows, dataset, motif, group, tab, "attention", folds["attention"], folds["attention_loo"],
            )
            done.add((dataset, motif, "attention"))
            done.add((dataset, motif, "attention_loo"))
            print(rec, flush=True)
            print(loo_rec, flush=True)
            _save(dest, rows)


def run_mlp(jobs, group, label, feature_dir, dest: Path) -> None:
    from scripts.eval_mlp_ig_window_importance import evaluate_mlp_ig

    rows, done = _resume(dest)
    for dataset, motif in jobs:
        if _scores_ready(rows, dest, dataset, motif, "mlp"):
            print(f"skip mlp {dataset} {motif}", flush=True)
            continue
        tab, y_r = _prepare(dataset, motif, label, feature_dir)
        fold, loo, _ranking = evaluate_mlp_ig(
            tab, y_r, DEFAULT_FAMILIES,
            checkpoint_dir=checkpoint_dir(dest, dataset, motif, "mlp"),
        )
        rows, rec, loo_rec = _store_pair(
            rows, dataset, motif, group, tab, "mlp", fold, loo, legacy=("mlp_ig",),
        )
        done.add((dataset, motif, "mlp"))
        done.add((dataset, motif, "mlp_loo"))
        _save(dest, rows)
        print(rec, flush=True)
        print(loo_rec, flush=True)


def run_rf(jobs, group, label, feature_dir, dest: Path) -> None:
    from scripts.eval_rf_mdi_window_importance import evaluate_rf_mdi

    rows, done = _resume(dest)
    for dataset, motif in jobs:
        if _checkpoints_ready(rows, dest, dataset, motif, "rf_mdi"):
            print(f"skip rf {dataset} {motif}", flush=True)
            continue
        tab, _y_r = _prepare(dataset, motif, label, feature_dir)
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


def run_additive(jobs, group, label, feature_dir, dest: Path) -> None:
    from scripts.eval_additive_mil_window_importance import evaluate_additive_localisation

    rows, done = _resume(dest)
    for dataset, motif in jobs:
        if _scores_ready(rows, dest, dataset, motif, "additive_mil"):
            print(f"skip additive {dataset} {motif}", flush=True)
            continue
        tab, y_r = _prepare(dataset, motif, label, feature_dir)
        fold, loo = evaluate_additive_localisation(
            tab, y_r, DEFAULT_FAMILIES,
            checkpoint_dir=checkpoint_dir(dest, dataset, motif, "additive_mil"),
        )
        rows, rec, loo_rec = _store_pair(
            rows, dataset, motif, group, tab, "additive_mil", fold, loo,
        )
        done.add((dataset, motif, "additive_mil"))
        done.add((dataset, motif, "additive_loo"))
        _save(dest, rows)
        print(rec, flush=True)
        print(loo_rec, flush=True)


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    return f"{number:.3f}"


def write_wide(long_df: pd.DataFrame, dest: Path) -> None:
    """Same two-row header as four_family_localisation_wide.csv."""
    present = [spec for spec in METHOD_SPECS if spec[0] in set(long_df["method"])]
    group_row = ["", "", ""]
    metric_row = ["task", "n_windows", "n_labeled"]
    for _key, group, name in present:
        group_row.extend([group, group])
        metric_row.extend([f"{name} (AUROC)", f"{name} (precision_at_20)"])
    group_row.extend(["", ""])
    metric_row.extend(["Highest Method (AUROC)", "Highest Method (precision_at_20)"])

    lines = [",".join(group_row), ",".join(metric_row)]
    order = {(ds, motif): i for i, (ds, motif) in enumerate(SELECTED_MOTIF_SOURCES)}
    tasks = long_df[["dataset", "motif", "n_windows", "n_labeled"]].drop_duplicates()
    tasks = sorted(tasks.itertuples(index=False), key=lambda r: order.get((r.dataset, r.motif), 10**9))
    for dataset, motif, n_windows, n_labeled in tasks:
        block = long_df[(long_df["dataset"] == dataset) & (long_df["motif"] == motif)]
        by_method = {row.method: row for row in block.itertuples(index=False)}
        cells = [f"{motif} ({dataset})", str(int(n_windows)), str(int(n_labeled))]
        auroc_best = ("", float("-inf"))
        prec_best = ("", float("-inf"))
        for key, _group, name in present:
            row = by_method.get(key)
            if row is None:
                cells.extend(["", ""])
                continue
            cells.extend([_fmt(row.auroc_window_mean), _fmt(row.precision_at_20)])
            if row.auroc_window_mean == row.auroc_window_mean and row.auroc_window_mean > auroc_best[1]:
                auroc_best = (name, float(row.auroc_window_mean))
            if row.precision_at_20 == row.precision_at_20 and row.precision_at_20 > prec_best[1]:
                prec_best = (name, float(row.precision_at_20))
        cells.extend([auroc_best[0], prec_best[0]])
        lines.append(",".join(cells))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")


def assemble(out_dir: Path) -> None:
    frames = []
    for method in ("lasso", "gated", "mlp", "rf", "additive"):
        path = out_dir / f"{method}.csv"
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
    parser.add_argument("--method", choices=[*METHODS, "all", "assemble"], required=True)
    parser.add_argument("--label", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--motif", default="")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--window-dir", default="", help="override the window-label directory")
    parser.add_argument("--bag-dir", default="", help="override the region-label directory")
    parser.add_argument("--out-dir", default="", help="override the result directory")
    args = parser.parse_args()

    global WINDOW_DIR, BAG_DIR
    if args.window_dir:
        WINDOW_DIR = Path(args.window_dir)
    if args.bag_dir:
        BAG_DIR = Path(args.bag_dir)

    families = tuple(args.families)
    if families != DEFAULT_FAMILIES:
        raise SystemExit("this driver keeps the joint four-family panel used in the wide table")
    group = combo_name(families)
    out_dir = Path(args.out_dir) if args.out_dir else _out_dir(args.label)
    if args.method == "assemble":
        assemble(out_dir)
        return
    jobs = _tasks(args.datasets, args.motif)
    if not jobs:
        raise SystemExit("no tasks selected")
    feature_dir = Path(args.feature_dir)
    runners = {
        "lasso": run_lasso,
        "gated": run_gated,
        "mlp": run_mlp,
        "rf": run_rf,
        "additive": run_additive,
    }
    chosen = METHODS if args.method == "all" else (args.method,)
    for method in chosen:
        print(f"=== {method} label={args.label} ===", flush=True)
        runners[method](jobs, group, args.label, feature_dir, out_dir / f"{method}.csv")
    if args.method == "all":
        assemble(out_dir)


if __name__ == "__main__":
    main()
