#!/usr/bin/env python
"""Schematic: pi_hat vs bag label Y_r for the 10 selected motif×dataset pairs.

    python scripts/plot_window_bag_association.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from benchmark.motifs.source_select import SELECTED_MOTIF_SOURCES  # noqa: E402
from window_bag_recovery import _region_scores  # noqa: E402

WINDOW_DIR = Path(
    "/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset"
)
BAG_DIR = Path("/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset")
OUT_DIR = _CODE / "results/pseudo_labels/mil_windows/maps/selected"
OUT = OUT_DIR / "pi_hat_vs_Yr.png"
OUT_ALL_ROC = OUT_DIR / "pi_hat_all_roc.png"
OUT_ALL_PR = OUT_DIR / "pi_hat_all_pr.png"
OUT_RHO = OUT_DIR / "rho_hat_vs_Yr.png"
OUT_RHO_ROC = OUT_DIR / "rho_hat_all_roc.png"
CONTROLS = {"tumor_high", "cd8_high"}
TAGS = {
    "bc_jackson2020": "Jackson",
    "hnc_wu2022": "HNC",
    "bc_metabric_ali2020": "METABRIC",
    "tnbc_wang2023": "TNBC",
}

# Highest-AUROC control + highest-AUROC spatial among the 10 pairs.
STRIP_PAIRS = (
    ("tnbc_wang2023", "cd8_high"),
    ("bc_metabric_ali2020", "interface_immune"),
)
TITLES = {
    ("tnbc_wang2023", "cd8_high"): "TNBC  cd8_high  (control)",
    ("bc_metabric_ali2020", "interface_immune"): "METABRIC  interface_immune  (spatial)",
}


def _load_pair(dataset: str, motif: str) -> pd.DataFrame:
    windows = pd.read_csv(
        WINDOW_DIR / f"{dataset}_windows_noisy.csv",
        usecols=lambda c: c in {"region_id", "motif_id", "label", "score_used", "score"},
    )
    bags = pd.read_csv(BAG_DIR / f"{dataset}_v2_noisy.csv")
    part = windows[windows["motif_id"].astype(str) == motif]
    scores = _region_scores(part)
    scores["region_id"] = scores["region_id"].astype(str)
    y = pd.to_numeric(bags.set_index(bags["region_id"].astype(str))[f"{motif}_label_v2"], errors="coerce")
    tab = scores.set_index("region_id").join(y.rename("y"), how="inner")
    keep = tab["y"].notna() & tab["pi_hat"].notna()
    if "rho_hat" in tab.columns:
        keep = keep & tab["rho_hat"].notna()
    return tab[keep]


def _strip(ax, y0: np.ndarray, y1: np.ndarray, rng: np.random.Generator) -> None:
    for x, vals, color in ((0, y0, "#4C78A8"), (1, y1, "#E45756")):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=12, alpha=0.35, c=color, linewidths=0, zorder=2)
        q = np.nanpercentile(vals, [25, 50, 75])
        ax.plot([x - 0.18, x + 0.18], [q[1], q[1]], color="k", lw=2, zorder=3)
        ax.plot([x, x], [q[0], q[2]], color="k", lw=1.4, zorder=3)


def main() -> None:
    rng = np.random.default_rng(0)
    pairs = {key: _load_pair(*key) for key in STRIP_PAIRS}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, key in zip(axes[:2], STRIP_PAIRS):
        tab = pairs[key]
        y = tab["y"].to_numpy(float)
        pi = tab["pi_hat"].to_numpy(float)
        g0, g1 = pi[y == 0], pi[y == 1]
        _strip(ax, g0, g1, rng)
        ax.set_xticks([0, 1], [r"$Y_r=0$", r"$Y_r=1$"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"positive-window density  $\hat{\pi}_r$")
        auroc = roc_auc_score(y, pi)
        rho = float(spearmanr(pi, y).statistic)
        r = float(pearsonr(pi, y).statistic)
        ax.set_title(
            f"{TITLES[key]}\nAUROC={auroc:.3f}   Spearman={rho:.3f}   Pearson={r:.3f}",
            fontsize=10,
        )
        ax.axhline(g0.mean(), color="#4C78A8", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(g1.mean(), color="#E45756", ls="--", lw=0.8, alpha=0.7)

    roc_ax = axes[2]
    for key, color in zip(STRIP_PAIRS, ("#4C78A8", "#F58518")):
        tab = pairs[key]
        fpr, tpr, _ = roc_curve(tab["y"], tab["pi_hat"])
        roc_ax.plot(fpr, tpr, color=color, lw=2, label=f"{key[1]}  {auc(fpr, tpr):.3f}")
    roc_ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="chance  0.50")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title(r"ROC:  $\hat{\pi}_r$  predicts  $Y_r$")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.set_aspect("equal")
    roc_ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(
        r"Selected pairs (noisy)   $\hat{\pi}_r$ vs bag label $Y_r$"
        "\nhighest-AUROC control and spatial",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    plot_all_rocs()
    plot_all_pr()
    plot_neg_rate()
    plot_neg_rocs()


def plot_all_rocs() -> None:
    curves = []
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        tab = _load_pair(dataset, motif)
        y = tab["y"].to_numpy(float)
        pi = tab["pi_hat"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, pi)
        tag = TAGS.get(dataset, dataset)
        curves.append((float(auc(fpr, tpr)), motif in CONTROLS, f"{tag}  {motif}", fpr, tpr))
    curves.sort(reverse=True)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.4, label="chance  0.50", zorder=0)
    for i, (auroc, control, name, fpr, tpr) in enumerate(curves):
        ax.plot(
            fpr, tpr,
            color=cmap(i % 10),
            lw=2.4 if control else 1.6,
            label=f"{name}  {auroc:.3f}",
            zorder=3 if control else 2,
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(r"10 selected pairs (noisy)"
                 "\n" r"ROC:  $\hat{\pi}_r$  predicts  $Y_r$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_ALL_ROC, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_ALL_ROC}")


def plot_all_pr() -> None:
    curves = []
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        tab = _load_pair(dataset, motif)
        y = tab["y"].to_numpy(float)
        pi = tab["pi_hat"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, pi)
        ap = float(average_precision_score(y, pi))
        prev = float(y.mean())
        tag = TAGS.get(dataset, dataset)
        curves.append((ap, motif in CONTROLS, f"{tag}  {motif}", rec, prec))
    curves.sort(reverse=True)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.axhline(0.5, color="0.55", ls="--", lw=1.4, label="chance  ~0.50", zorder=0)
    for i, (ap, control, name, rec, prec) in enumerate(curves):
        ax.plot(
            rec, prec,
            color=cmap(i % 10),
            lw=2.4 if control else 1.6,
            label=f"{name}  {ap:.3f}",
            zorder=3 if control else 2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(r"10 selected pairs (noisy)"
                 "\n" r"PR:  $\hat{\pi}_r$  predicts  $Y_r$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_ALL_PR, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_ALL_PR}")


def plot_neg_rate() -> None:
    """Negative-window rate ρ̂_r = (# label==0) / (# labeled) vs Y_r."""
    rng = np.random.default_rng(0)
    pairs = {key: _load_pair(*key) for key in STRIP_PAIRS}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, key in zip(axes[:2], STRIP_PAIRS):
        tab = pairs[key]
        y = tab["y"].to_numpy(float)
        rho = tab["rho_hat"].to_numpy(float)
        g0, g1 = rho[y == 0], rho[y == 1]
        _strip(ax, g0, g1, rng)
        ax.set_xticks([0, 1], [r"$Y_r=0$", r"$Y_r=1$"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"negative-window rate  $\hat{\rho}_r$")
        # High ρ̂ should recover negative bags.
        auroc = roc_auc_score((y == 0).astype(float), rho)
        spr = float(spearmanr(rho, y).statistic)
        r = float(pearsonr(rho, y).statistic)
        ax.set_title(
            f"{TITLES[key]}\nAUROC(ρ̂→Y=0)={auroc:.3f}   Spearman={spr:.3f}   Pearson={r:.3f}",
            fontsize=9,
        )
        ax.axhline(g0.mean(), color="#4C78A8", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(g1.mean(), color="#E45756", ls="--", lw=0.8, alpha=0.7)

    roc_ax = axes[2]
    for key, color in zip(STRIP_PAIRS, ("#4C78A8", "#F58518")):
        tab = pairs[key]
        fpr, tpr, _ = roc_curve((tab["y"] == 0).astype(int), tab["rho_hat"])
        roc_ax.plot(fpr, tpr, color=color, lw=2, label=f"{key[1]}  {auc(fpr, tpr):.3f}")
    roc_ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="chance  0.50")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title(r"ROC:  $\hat{\rho}_r$  predicts  $Y_r=0$")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.set_aspect("equal")
    roc_ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        r"Selected pairs (noisy)   $\hat{\rho}_r=1-\hat{\pi}_r$ vs bag label $Y_r$",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT_RHO, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_RHO}")


def plot_neg_rocs() -> None:
    curves = []
    for dataset, motif in SELECTED_MOTIF_SOURCES:
        tab = _load_pair(dataset, motif)
        y0 = (tab["y"] == 0).astype(int).to_numpy()
        rho = tab["rho_hat"].to_numpy(float)
        if len(np.unique(y0)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y0, rho)
        tag = TAGS.get(dataset, dataset)
        curves.append((float(auc(fpr, tpr)), motif in CONTROLS, f"{tag}  {motif}", fpr, tpr))
    curves.sort(reverse=True)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.4, label="chance  0.50", zorder=0)
    for i, (auroc, control, name, fpr, tpr) in enumerate(curves):
        ax.plot(
            fpr, tpr, color=cmap(i % 10),
            lw=2.4 if control else 1.6,
            label=f"{name}  {auroc:.3f}",
            zorder=3 if control else 2,
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(r"10 selected pairs (noisy)"
                 "\n" r"ROC:  $\hat{\rho}_r$  predicts  $Y_r=0$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_RHO_ROC, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_RHO_ROC}")


if __name__ == "__main__":
    main()
