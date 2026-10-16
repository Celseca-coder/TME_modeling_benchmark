#!/usr/bin/env python
"""Recover bag labels Y_r from window-positive density or mean score_used.

For each dataset × motif, a region score is either:

* ``pi_hat``: (# windows with label==1) / (# windows with label in {0,1})
* ``mean_score``: nanmean of window ``score_used``

These scores predict the bag label ``{motif}_label_v2``. Reports AUROC, AUPRC,
point-biserial (Pearson) and Spearman. An optional within-bag tertile of
window scores is a negative control (AUROC should sit near 0.5).

    python scripts/window_bag_recovery.py \
        --window-dir /autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset \
        --bag-dir /autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset \
        --window-suffix _windows_noisy.csv \
        --bag-suffix _v2_noisy.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))

from benchmark.motifs.source_select import selected_motifs_for  # noqa: E402

CONTROL_MOTIFS = ("tumor_high", "cd8_high")
DATASETS = (
    "hnc_wu2022",
    "bc_jackson2020",
    "bc_metabric_ali2020",
    "tnbc_wang2023",
)


def _within_bag_tertile_pi(part: pd.DataFrame) -> pd.Series:
    """Positive-window density after tertile_extremes inside each region."""
    rows = []
    for rid, g in part.groupby("region_id", sort=False):
        vals = pd.to_numeric(g["score_used"], errors="coerce")
        finite = vals.dropna()
        if len(finite) < 3:
            rows.append((rid, np.nan))
            continue
        lo, hi = np.nanquantile(finite.to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lab = (vals >= float(finite.median())).astype(float)
            lab[vals.isna()] = np.nan
        else:
            lab = pd.Series(np.nan, index=g.index, dtype="float")
            lab.loc[vals <= lo] = 0.0
            lab.loc[vals >= hi] = 1.0
        labeled = lab.dropna()
        rows.append((rid, float((labeled == 1).mean()) if len(labeled) else np.nan))
    return pd.Series({r: p for r, p in rows}, dtype="float")


def _region_scores(part: pd.DataFrame) -> pd.DataFrame:
    score = pd.to_numeric(part.get("score_used", part.get("score")), errors="coerce")
    lab = pd.to_numeric(part["label"], errors="coerce")
    tmp = part.assign(_score=score, _lab=lab)
    rows = []
    for rid, g in tmp.groupby("region_id", sort=False):
        labeled = g["_lab"].dropna()
        n_lab = int(len(labeled))
        pi = float((labeled == 1).mean()) if n_lab else np.nan
        rho = float((labeled == 0).mean()) if n_lab else np.nan
        rows.append({
            "region_id": rid,
            "pi_hat": pi,
            "rho_hat": rho,
            "mean_score": float(g["_score"].mean(skipna=True)),
            "n_windows": int(len(g)),
            "n_labeled": n_lab,
            "n_pos": int((g["_lab"] == 1).sum()),
            "n_neg": int((g["_lab"] == 0).sum()),
        })
    return pd.DataFrame(rows)


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out = dict(n_regions=n, n_bag_pos=n_pos, n_bag_neg=n_neg,
               auroc=np.nan, auprc=np.nan, pearson=np.nan, spearman=np.nan,
               prev=n_pos / n if n else np.nan)
    if n_pos == 0 or n_neg == 0 or n < 6:
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    if np.nanstd(s) > 0:
        out["pearson"] = float(pearsonr(s, y).statistic)
        out["spearman"] = float(spearmanr(s, y).statistic)
    return out


def evaluate_pair(windows: pd.DataFrame, bags: pd.DataFrame, motif: str) -> list[dict]:
    ycol = f"{motif}_label_v2"
    if ycol not in bags.columns or motif not in set(windows["motif_id"].astype(str)):
        return []
    part = windows[windows["motif_id"].astype(str) == motif].copy()
    if "score_used" not in part.columns:
        part["score_used"] = part.get("score")
    y = pd.to_numeric(bags.set_index(bags["region_id"].astype(str))[ycol], errors="coerce")
    scores = _region_scores(part)
    scores["region_id"] = scores["region_id"].astype(str)
    scores = scores.set_index("region_id")
    tab = scores.join(y.rename("y"), how="inner")
    tab = tab[tab["y"].notna()]
    kind = "control" if motif in CONTROL_MOTIFS else "spatial"
    rows = []
    for score_name, col in (("pi_hat", "pi_hat"), ("mean_score", "mean_score")):
        m = _metrics(tab["y"].to_numpy(float), tab[col].to_numpy(float))
        rows.append({"motif": motif, "kind": kind, "score": score_name, "control": "global_window_gt", **m})
    pi_bag = _within_bag_tertile_pi(part)
    aligned = tab.join(pi_bag.rename("pi_within"), how="left")
    m = _metrics(aligned["y"].to_numpy(float), aligned["pi_within"].to_numpy(float))
    rows.append({"motif": motif, "kind": kind, "score": "pi_hat", "control": "within_bag_tertile", **m})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-dir",
                    default="/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset")
    ap.add_argument("--bag-dir",
                    default="/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
    ap.add_argument("--window-suffix", default="_windows_noisy.csv")
    ap.add_argument("--bag-suffix", default="_v2_noisy.csv")
    ap.add_argument("--output",
                    default="results/pseudo_labels/mil_windows/window_bag_recovery.csv")
    args = ap.parse_args()

    frames = []
    for ds in DATASETS:
        wpath = Path(args.window_dir) / f"{ds}{args.window_suffix}"
        bpath = Path(args.bag_dir) / f"{ds}{args.bag_suffix}"
        if not wpath.exists() or not bpath.exists():
            print(f"skip {ds}: missing {wpath.name if not wpath.exists() else bpath.name}", flush=True)
            continue
        print(f"reading {ds}", flush=True)
        windows = pd.read_csv(wpath, usecols=lambda c: c in {
            "region_id", "motif_id", "label", "score_used", "score",
        })
        bags = pd.read_csv(bpath)
        motifs = list(selected_motifs_for(ds)) or sorted(windows["motif_id"].astype(str).unique())
        for motif in motifs:
            for rec in evaluate_pair(windows, bags, motif):
                rec["dataset"] = ds
                frames.append(rec)

    out = pd.DataFrame(frames)
    cols = ["dataset", "motif", "kind", "score", "control", "n_regions",
            "n_bag_pos", "n_bag_neg", "prev", "auroc", "auprc", "pearson", "spearman"]
    out = out[[c for c in cols if c in out.columns]]
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Wrote {dest}  ({len(out)} rows)", flush=True)
    show = out[out.control == "global_window_gt"]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)


if __name__ == "__main__":
    main()
