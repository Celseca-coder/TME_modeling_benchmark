#!/usr/bin/env python
"""Score Lasso, SHAP, UTAG portraits, embeddings, and GNN-explainer separately.

Does not treat ``UTAG domains + Lasso/SHAP`` as one interpreter. Tabular Lasso
and SHAP share a feature table but are gated independently.

    python scripts/summarize_interpreter_panel.py \\
        --panel results/pseudo_labels_all/selected_catalog.csv \\
        --input-dir results/pseudo_label_explanations_panel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.panel import load_selected_panel  # noqa: E402

CONTROLS = ("motif_tumor_high", "motif_cd8_high")


def _is_control(task: str) -> bool:
    return str(task) in CONTROLS


def _verdict(kind: str, passed: bool, controls_ok: bool) -> str:
    if kind == "control":
        return "pass_control" if passed else "fail_control"
    if not controls_ok:
        return "fail_control_block"
    return "pass_spatial" if passed else "fail_spatial"


def _dataset_controls_ok(rows: pd.DataFrame, pass_col: str) -> dict[str, bool]:
    ok: dict[str, bool] = {}
    for dataset, sub in rows.groupby("dataset"):
        ctrl = sub[sub["task"].isin(CONTROLS)]
        if ctrl.empty:
            ok[str(dataset)] = False
            continue
        ok[str(dataset)] = bool(ctrl[pass_col].all())
    return ok


def _score_selected(
    method: str,
    rows: pd.DataFrame,
    pass_col: str,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    selected = {
        (str(r.source_dataset), f"motif_{r.motif_id}")
        for r in panel.itertuples(index=False)
    }
    controls_ok = _dataset_controls_ok(rows, pass_col)
    out_rows = []
    n_ctrl = n_ctrl_pass = n_spatial = n_spatial_pass = n_blocked = 0
    for dataset, task in sorted(selected):
        sub = rows[(rows["dataset"].astype(str) == dataset) & (rows["task"] == task)]
        kind = "control" if _is_control(task) else "spatial"
        if sub.empty:
            out_rows.append(dict(
                method=method, dataset=dataset, task=task, kind=kind,
                passed=False, verdict="missing", controls_ok=controls_ok.get(dataset, False),
            ))
            if kind == "control":
                n_ctrl += 1
            else:
                n_spatial += 1
            continue
        passed = bool(sub[pass_col].iloc[0])
        ds_ok = controls_ok.get(dataset, False)
        verdict = _verdict(kind, passed, ds_ok)
        counted = passed if kind == "control" else (passed and ds_ok)
        out_rows.append(dict(
            method=method, dataset=dataset, task=task, kind=kind,
            passed=counted, verdict=verdict, controls_ok=ds_ok,
            raw_passed=passed,
        ))
        if kind == "control":
            n_ctrl += 1
            n_ctrl_pass += int(passed)
        else:
            n_spatial += 1
            if not ds_ok:
                n_blocked += 1
            else:
                n_spatial_pass += int(passed)
    summary = dict(
        method=method,
        n_control=n_ctrl,
        n_control_passed=n_ctrl_pass,
        n_spatial=n_spatial,
        n_spatial_blocked=n_blocked,
        n_spatial_scored=n_spatial - n_blocked,
        n_spatial_passed=n_spatial_pass,
        control_ok=n_ctrl_pass == n_ctrl and n_ctrl > 0,
        spatial_score=f"{n_spatial_pass}/{n_spatial - n_blocked}" if n_spatial > n_blocked else f"0/{n_spatial}",
    )
    return pd.DataFrame(out_rows), summary


def _load_tabular(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["lasso_ok"] = table["frac_lasso_passed"] >= 0.5
    table["shap_ok"] = table["frac_shap_passed"] >= 0.5
    return table


def _load_embedding(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.rename(columns={"mean": "auc"}) if "auc" not in table.columns else table
    kinds = []
    passed = []
    for row in table.itertuples(index=False):
        control = _is_control(row.task)
        kinds.append("control" if control else "spatial")
        auc = float(row.auc) if hasattr(row, "auc") else float(row.mean)
        passed.append(auc >= (0.90 if control else 0.60))
    table["kind"] = kinds
    table["embed_ok"] = passed
    return table


def _load_utag(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/summary.csv")):
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            frame["dataset"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table["utag_ok"] = table["passed"].astype(bool)
    return table


def _load_gnn(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "frac_passed" in table.columns:
        table["gnn_ok"] = table["frac_passed"] >= 0.5
    elif "passed" in table.columns:
        table["gnn_ok"] = table["passed"].astype(bool)
    else:
        raise ValueError(f"{path} needs frac_passed or passed")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--panel",
        default=str(_CODE / "results" / "pseudo_labels_all" / "selected_catalog.csv"),
    )
    ap.add_argument(
        "--input-dir",
        default=str(_CODE / "results" / "pseudo_label_explanations_panel"),
    )
    args = ap.parse_args()
    panel = load_selected_panel(args.panel)
    root = Path(args.input_dir)
    task_frames = []
    method_rows = []

    tabular_path = root / "tabular" / "summary.csv"
    if tabular_path.is_file():
        tabular = _load_tabular(tabular_path)
        for method, col in (("lasso", "lasso_ok"), ("shap", "shap_ok")):
            tasks, summary = _score_selected(method, tabular, col, panel)
            task_frames.append(tasks)
            method_rows.append(summary)
    else:
        print(f"skip lasso/shap: missing {tabular_path}", flush=True)

    for name, rel in (("kronos", "kronos_embedding/embedding_probe_summary.csv"),
                      ("eva", "eva_embedding/embedding_probe_summary.csv")):
        path = root / rel
        if not path.is_file():
            print(f"skip {name}: missing {path}", flush=True)
            continue
        tasks, summary = _score_selected(name, _load_embedding(path), "embed_ok", panel)
        task_frames.append(tasks)
        method_rows.append(summary)

    utag = _load_utag(root / "utag_native_portraits")
    if utag.empty:
        print("skip utag: no utag_native_portraits/*/summary.csv", flush=True)
    else:
        tasks, summary = _score_selected("utag_cellular", utag, "utag_ok", panel)
        task_frames.append(tasks)
        method_rows.append(summary)

    gnn_path = root / "gnn_explainer" / "gnn_summary.csv"
    gnn = _load_gnn(gnn_path)
    if gnn.empty:
        print(f"skip gnn-explainer: missing {gnn_path}", flush=True)
    else:
        tasks, summary = _score_selected("gnn_explainer", gnn, "gnn_ok", panel)
        task_frames.append(tasks)
        method_rows.append(summary)

    out_dir = root / "method_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_out = pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame()
    methods_out = pd.DataFrame(method_rows)
    tasks_out.to_csv(out_dir / "task_verdicts.csv", index=False)
    methods_out.to_csv(out_dir / "method_summary.csv", index=False)
    print(methods_out.to_string(index=False) if not methods_out.empty else "(empty)", flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
