from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark.models.linear import LinearClassifier
from benchmark.motifs.recovery import RULES, rank_recovery


class MotifRecoveryTest(unittest.TestCase):
    def test_control_recovers_composition_tumor(self):
        ranked = [
            "composition::Tumor",
            "composition::CD8 T cell",
            "point-pattern::CD8 T cell_K_r50",
        ]
        rec = rank_recovery(ranked, RULES["motif_tumor_high"], k=5)
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["best_hit_rank"], 1)
        self.assertFalse(rec["miss_as_top1"])

    def test_clustering_fails_when_abundance_is_top1(self):
        ranked = [
            "composition::CD8 T cell",
            "point-pattern::CD8 T cell_K_r50",
            "composition::Tumor",
        ]
        rec = rank_recovery(ranked, RULES["motif_cd8_clustering"], k=5)
        self.assertTrue(rec["recovered"])
        self.assertTrue(rec["miss_as_top1"])
        self.assertFalse(rec["passed"])

    def test_clustering_passes_on_type_specific_k(self):
        ranked = [
            "point-pattern::CD8 T cell_K_r50",
            "point-pattern::CD8 T cell_L_r100",
            "composition::Tumor",
        ]
        rec = rank_recovery(ranked, RULES["motif_cd8_clustering"], k=5)
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["best_hit_rank"], 1)
        self.assertFalse(rec["miss_as_top1"])

    def test_t_tumor_mixing_fails_on_t_cell_fraction(self):
        rec = rank_recovery(
            ["composition::T cell", "point-pattern::T cell_K_r50"],
            RULES["motif_t_tumor_mixing"],
            k=5,
        )
        self.assertTrue(rec["miss_as_top1"])
        self.assertFalse(rec["passed"])
        rec_ok = rank_recovery(
            ["point-pattern::T cell_K_r50", "composition::Tumor"],
            RULES["motif_t_tumor_mixing"],
            k=5,
        )
        self.assertTrue(rec_ok["passed"])

    def test_exclusion_hits_tumor_density_not_fraction(self):
        rec = rank_recovery(
            [
                "density::tumor_density::CD8 T cell",
                "density::tissue_density::CD8 T cell",
                "composition::Stroma",
            ],
            RULES["motif_immune_exclusion"],
            k=5,
        )
        self.assertTrue(rec["passed"])
        rec_fail = rank_recovery(
            ["composition::CD8 T cell", "density::tumor_density::CD8 T cell"],
            RULES["motif_immune_exclusion"],
            k=5,
        )
        self.assertTrue(rec_fail["miss_as_top1"])
        self.assertFalse(rec_fail["passed"])

    def test_mixing_requires_spatial_not_stroma_fraction(self):
        rec = rank_recovery(
            ["composition::Stroma", "composition::Tumor"],
            RULES["motif_tumor_stroma_mixing"],
            k=5,
        )
        self.assertFalse(rec["passed"])
        rec_ok = rank_recovery(
            ["point-pattern::Stroma_K_r50", "point-pattern::Tumor_L_r20"],
            RULES["motif_tumor_stroma_mixing"],
            k=5,
        )
        self.assertTrue(rec_ok["passed"])

    def test_lasso_uses_l1_penalty(self):
        rng = np.random.default_rng(0)
        y = pd.Series(np.array([0, 1] * 20), index=[f"r{i}" for i in range(40)])
        X = pd.DataFrame(
            {
                "signal": y.to_numpy() + rng.normal(0, 0.05, 40),
                "noise_a": rng.normal(size=40),
                "noise_b": rng.normal(size=40),
                "noise_c": rng.normal(size=40),
            },
            index=y.index,
        )
        model = LinearClassifier(seed=0, C=0.2, l1_ratio=1.0).fit(X, y)
        self.assertEqual(model._clf.penalty, "l1")


class UTAGRecoveryTest(unittest.TestCase):
    def test_control_hits_panck_mean_or_tumor_domain(self):
        from benchmark.motifs.recovery import UTAG_RULES

        rec = rank_recovery(
            ["utag_mean__PanCK", "utag_mean__CD8"],
            UTAG_RULES["motif_tumor_high"],
            k=5,
        )
        self.assertTrue(rec["passed"])
        rec_domain = rank_recovery(
            ["utag_domain::Tumor__03", "utag_mean__CD45"],
            UTAG_RULES["motif_tumor_high"],
            k=5,
        )
        self.assertTrue(rec_domain["passed"])

    def test_clustering_requires_cd8_domain_not_marker_mean(self):
        from benchmark.motifs.recovery import UTAG_RULES

        rec_fail = rank_recovery(
            ["utag_mean__CD8", "utag_domain::CD8 T cell__02"],
            UTAG_RULES["motif_cd8_clustering"],
            k=5,
        )
        self.assertTrue(rec_fail["miss_as_top1"])
        self.assertFalse(rec_fail["passed"])
        rec_ok = rank_recovery(
            ["utag_domain::CD8 T cell__02", "utag_mean__PanCK"],
            UTAG_RULES["motif_cd8_clustering"],
            k=5,
        )
        self.assertTrue(rec_ok["passed"])


class DummyCatalog:
    cell_sets = {
        "tumor": ("Tumor", "Tumor (Proliferating)"),
        "cd8": ("CD8 T cell",),
        "stroma": ("Stroma",),
        "immune": ("CD8 T cell", "CD4 T cell", "B cell", "Macrophage"),
    }


class NativePortraitTest(unittest.TestCase):
    def test_tumor_portrait_hits_panck_or_tumor_majority(self):
        from benchmark.motifs.recovery import PORTRAIT_SPECS, portrait_matches

        spec = PORTRAIT_SPECS["motif_tumor_high"]
        hit = portrait_matches("Tumor", ["PanCK", "Ki67"], {"tumor": 0.8, "cd8": 0.0}, DummyCatalog, spec)
        self.assertTrue(hit["hit"])
        self.assertFalse(hit["miss"])
        hit_marker = portrait_matches("Stroma", ["PanCK", "Vimentin"], {"tumor": 0.1, "stroma": 0.6}, DummyCatalog, spec)
        self.assertTrue(hit_marker["hit"])

    def test_clustering_rejects_tumor_majority(self):
        from benchmark.motifs.recovery import PORTRAIT_SPECS, portrait_matches

        spec = PORTRAIT_SPECS["motif_cd8_clustering"]
        miss = portrait_matches("Tumor", ["PanCK"], {"tumor": 0.7, "cd8": 0.05}, DummyCatalog, spec)
        self.assertTrue(miss["miss"])
        ok = portrait_matches("CD8 T cell", ["CD8", "CD45"], {"tumor": 0.05, "cd8": 0.4}, DummyCatalog, spec)
        self.assertTrue(ok["hit"])
        self.assertFalse(ok["miss"])


if __name__ == "__main__":
    unittest.main()
