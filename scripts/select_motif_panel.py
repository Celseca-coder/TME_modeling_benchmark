#!/usr/bin/env python
"""Apply the HNC composition → expression / composition → density gates.

Same thresholds as the HNC methods baseline:

* controls: composition AUC-ROC ≈ 1 (implemented as ≥ 0.90) and
  expression AUC-ROC ≥ 0.90 (density panel: density ≥ 0.90).
* spatial gate 1: composition near chance. Fail if AUC-ROC ≥ 0.64
  (abundance shortcut). 0.60–0.64 is borderline but still proceeds.
* expression panel (leakage filter): keep spatial only if expression is
  also near chance (AUC-ROC < 0.65). 0.65–0.75 is hold; ≥ 0.75 is leak.
* density panel (recoverability): keep spatial if density AUC-ROC ≥ 0.80.

    python scripts/select_motif_panel.py \\
        --benchmark results/motif_screening/benchmark.csv \\
        --labels-dir results/pseudo_labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.spec import load_motif_catalog  # noqa: E402

COMPOSITION_LEAK = 0.64
COMPOSITION_BORDERLINE = 0.60
EXPRESSION_CHANCE = 0.65
EXPRESSION_HOLD_HI = 0.75
DENSITY_RECOVER = 0.80
CONTROL_MIN = 0.90

CONTROLS = ("tumor_high", "cd8_high")


def _motif_id(task: str) -> str:
    return str(task).removeprefix("motif_")


def _auc_map(bench: pd.DataFrame, source: str) -> dict[tuple[str, str], float]:
    sub = bench[bench["feature_source"] == source]
    out: dict[tuple[str, str], float] = {}
    for row in sub.itertuples(index=False):
        out[(str(row.dataset), _motif_id(str(row.task)))] = float(row.mean)
    return out


def _n_labeled(labels: pd.DataFrame, motif_id: str) -> tuple[int, int, int]:
    col = f"{motif_id}_label_v2"
    if col not in labels.columns:
        col = f"{motif_id}_label"
    if col not in labels.columns:
        return 0, 0, 0
    lab = labels[col]
    n0 = int((lab == 0).sum())
    n1 = int((lab == 1).sum())
    return n0 + n1, n0, n1


def _gate1(auc: float | None, spatial: bool) -> str:
    if auc is None or not pd.notna(auc):
        return "missing"
    if not spatial:
        return "pass_control" if auc >= CONTROL_MIN else "fail_control"
    if auc >= COMPOSITION_LEAK:
        return "fail"
    if auc >= COMPOSITION_BORDERLINE:
        return "borderline"
    return "pass"


def _gate_expression(auc: float | None, spatial: bool) -> str:
    if auc is None or not pd.notna(auc):
        return "missing"
    if not spatial:
        return "pass_control" if auc >= CONTROL_MIN else "fail_control"
    if auc < EXPRESSION_CHANCE:
        return "pass"
    if auc < EXPRESSION_HOLD_HI:
        return "hold"
    return "fail"


def _gate_density(auc: float | None, spatial: bool) -> str:
    if auc is None or not pd.notna(auc):
        return "missing"
    if not spatial:
        return "pass_control" if auc >= CONTROL_MIN else "fail_control"
    if auc >= DENSITY_RECOVER:
        return "pass"
    return "fail"


def _expr_decision(gate1: str, gate2: str, spatial: bool) -> str:
    if not spatial:
        if gate1 == "pass_control" and gate2 == "pass_control":
            return "keep_control"
        return "fail_control"
    if gate1 == "fail":
        return "drop_composition_leak"
    if gate2 == "fail":
        return "drop_expression_leak"
    if gate2 == "hold" or gate1 == "borderline":
        return "hold"
    if gate2 == "pass":
        return "keep_spatial"
    return "hold"


def _dens_decision(gate1: str, gate2: str, spatial: bool, motif_id: str) -> str:
    if not spatial:
        if gate1 == "pass_control" and gate2 == "pass_control":
            return "keep_control"
        return "fail_control"
    if gate1 == "fail":
        return "drop_composition_leak"
    if gate2 != "pass":
        return "hold"
    if motif_id in ("interface_immune", "immune_exclusion"):
        return "keep_spatial_caveat"
    if gate1 == "borderline":
        return "keep_spatial_caveat"
    return "keep_spatial"


def select_tables(
    bench: pd.DataFrame,
    labels_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp = _auc_map(bench, "composition")
    expr = _auc_map(bench, "expression")
    dens = _auc_map(bench, "density")

    expr_rows = []
    dens_rows = []
    datasets = sorted(bench["dataset"].astype(str).unique())
    for dataset in datasets:
        catalog = load_motif_catalog(dataset=dataset)
        labels_path = labels_dir / f"{dataset}_v2.csv"
        if not labels_path.exists():
            labels_path = labels_dir / f"{dataset}.csv"
        labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
        for spec in catalog.motifs:
            n, n0, n1 = _n_labeled(labels, spec.id) if len(labels) else (0, 0, 0)
            key = (dataset, spec.id)
            auc_c = comp.get(key)
            auc_e = expr.get(key)
            auc_d = dens.get(key)
            g1 = _gate1(auc_c, spec.spatial)
            g2e = _gate_expression(auc_e, spec.spatial)
            g2d = _gate_density(auc_d, spec.spatial)
            d_e = _expr_decision(g1, g2e, spec.spatial)
            d_d = _dens_decision(g1, g2d, spec.spatial, spec.id)
            common = dict(
                dataset=dataset,
                motif_id=spec.id,
                role="control" if spec.id in CONTROLS else "spatial",
                spatial=spec.spatial,
                residual_on=";".join(spec.residual_on),
                n_labeled_v2=n,
                n_0=n0,
                n_1=n1,
                auc_composition=auc_c,
                gate1_composition=g1,
            )
            expr_rows.append(dict(
                **common,
                panel="composition_then_expression",
                auc_expression=auc_e,
                gate2_expression=g2e,
                decision=d_e,
                in_formal_set=d_e in ("keep_control", "keep_spatial"),
            ))
            dens_rows.append(dict(
                **common,
                panel="composition_then_density",
                auc_density=auc_d,
                gate2_density=g2d,
                decision=d_d,
                in_formal_set=d_d in ("keep_control", "keep_spatial", "keep_spatial_caveat"),
            ))

    expr_df = pd.DataFrame(expr_rows)
    dens_df = pd.DataFrame(dens_rows)

    summary_rows = []
    for dataset in datasets:
        e = expr_df[expr_df["dataset"] == dataset]
        d = dens_df[dens_df["dataset"] == dataset]
        e_keep = e[e["in_formal_set"]]
        d_keep = d[d["in_formal_set"]]
        summary_rows.append(dict(
            dataset=dataset,
            n_motifs_scored=int(len(e)),
            expression_kept=";".join(e_keep["motif_id"]),
            expression_n=int(e_keep.shape[0]),
            density_kept=";".join(d_keep["motif_id"]),
            density_n=int(d_keep.shape[0]),
            union=";".join(sorted(set(e_keep["motif_id"]) | set(d_keep["motif_id"]))),
        ))
    return expr_df, dens_df, pd.DataFrame(summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--labels-dir", default=str(_CODE / "results" / "pseudo_labels"))
    ap.add_argument("--output-dir", default=str(_CODE / "results" / "motif_screening"))
    args = ap.parse_args()

    bench = pd.read_csv(args.benchmark)
    expr_df, dens_df, summary = select_tables(bench, Path(args.labels_dir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    expr_path = out / "motif_panel_composition_expression.csv"
    dens_path = out / "motif_panel_composition_density.csv"
    sum_path = out / "motif_panel_summary.csv"
    expr_df.to_csv(expr_path, index=False)
    dens_df.to_csv(dens_path, index=False)
    summary.to_csv(sum_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {expr_path}", flush=True)
    print(f"Wrote {dens_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)


if __name__ == "__main__":
    main()
