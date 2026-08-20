#!/usr/bin/env python
"""Write per-dataset motif catalogs that inherit the HNC recipes.

Cell-set names follow each dataset's native ``cell_type`` / ``cell_type_uniform``
vocabulary. Motifs whose required sets are empty are dropped at load time.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "motifs"


def dump(name: str, payload: dict) -> None:
    path = OUT / f"{name}.yaml"
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=88)
    path.write_text(text)
    print(f"wrote {path}")


def main() -> None:
    # Breast — Jackson 2020 (no CD8/CD4 split; T cell is the CD8 proxy)
    dump("bc_jackson2020", {
        "dataset": "bc_jackson2020",
        "cell_type_col": "cell_type_uniform",
        "cv_filter": "cohort == 'Basel'",
        "inherit_motifs": "hnc_wu2022",
        "cd8_definition": "T cell (panel has no CD8 vs CD4 split)",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": [
                "Tumor (Apoptotic)", "Tumor (Basal CK)", "Tumor (CK lo HR hi p53+)",
                "Tumor (CK lo HR lo)", "Tumor (CK+ HR hi)", "Tumor (CK+ HR lo)",
                "Tumor (CK+ HR+)", "Tumor (CK7+ CK hi Ecadh hi)", "Tumor (CK7+)",
                "Tumor (Epithelial low)", "Tumor (Hypoxic)", "Tumor (Myoepithelial)",
                "Tumor (Proliferative)", "Tumor (p53+ EGFR+)",
            ],
            "t_cell": ["T cell"],
            "cd8": ["T cell"],
            "b_cell": ["B cell"],
            "macrophage": ["Macrophage"],
            "apc": ["Macrophage"],
            "vessel": ["Endothelial"],
            "stroma": ["Stromal cells"],
            "immune": ["B cell", "T cell", "Macrophage", "T & B cells"],
        },
    })

    # Breast — METABRIC Ali 2020 (T cells as CD8 proxy)
    dump("bc_metabric_ali2020", {
        "dataset": "bc_metabric_ali2020",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "cd8_definition": "T cells (panel has no CD8 vs CD4 split)",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": [
                "Basal CKlow", "HER2+", "HR+ CK7-", "HR+ CK7- Ki67+", "HR+ CK7- Slug+",
                "HR- CK7+", "HR- CK7-", "HR- CKlow CK5+", "HR- Ki67+", "HRlow CKlow",
                "Hypoxia", "Myoepithelial",
            ],
            "t_cell": ["T cells"],
            "cd8": ["T cells"],
            "b_cell": ["B cells"],
            "macrophage": [
                "Macrophages Vim+ CD45low", "Macrophages Vim+ Slug+", "Macrophages Vim+ Slug-",
            ],
            "apc": [
                "Macrophages Vim+ CD45low", "Macrophages Vim+ Slug+", "Macrophages Vim+ Slug-",
            ],
            "vessel": ["Endothelial", "Vascular SMA+"],
            "stroma": ["Fibroblasts", "Fibroblasts CD68+", "Myofibroblasts"],
            "immune": [
                "B cells", "T cells",
                "Macrophages Vim+ CD45low", "Macrophages Vim+ Slug+", "Macrophages Vim+ Slug-",
            ],
        },
    })

    # CRC — Schürch 2020
    dump("crc_schurch2020", {
        "dataset": "crc_schurch2020",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": ["tumor cells"],
            "t_cell": [
                "CD3+ T cells", "CD4+ T cells", "CD4+ T cells CD45RO+",
                "CD4+ T cells GATA3+", "CD8+ T cells", "Tregs",
            ],
            "cd8": ["CD8+ T cells"],
            "cd4": ["CD4+ T cells", "CD4+ T cells CD45RO+", "CD4+ T cells GATA3+", "Tregs"],
            "b_cell": ["B cells"],
            "macrophage": [
                "CD11b+CD68+ macrophages", "CD163+ macrophages", "CD68+ macrophages",
                "CD68+ macrophages GzmB+", "CD68+CD163+ macrophages",
            ],
            "apc": [
                "CD11c+ DCs", "CD11b+CD68+ macrophages", "CD163+ macrophages",
                "CD68+ macrophages", "CD68+ macrophages GzmB+", "CD68+CD163+ macrophages",
            ],
            "granulocyte": ["granulocytes"],
            "vessel": ["vasculature", "lymphatics"],
            "stroma": ["stroma", "smooth muscle", "adipocytes"],
            "immune": [
                "B cells", "CD3+ T cells", "CD4+ T cells", "CD4+ T cells CD45RO+",
                "CD4+ T cells GATA3+", "CD8+ T cells", "Tregs", "NK cells",
                "plasma cells", "CD11b+ monocytes", "CD11b+CD68+ macrophages",
                "CD11c+ DCs", "CD163+ macrophages", "CD68+ macrophages",
                "CD68+ macrophages GzmB+", "CD68+CD163+ macrophages", "granulocytes",
                "immune cells",
            ],
        },
    })

    # CRC — Wu 2022
    dump("crc_wu2022", {
        "dataset": "crc_wu2022",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": [
                "Tumor 1", "Tumor 2 (Ki67 Proliferating)", "Tumor 3", "Tumor 4",
                "Tumor 5", "Tumor 7",
            ],
            "t_cell": ["CD4 T cell", "CD8 T cell"],
            "cd8": ["CD8 T cell"],
            "cd4": ["CD4 T cell"],
            "b_cell": ["B cell"],
            "macrophage": ["Macrophage"],
            "apc": ["Macrophage", "Tumor 6 / DC"],
            "granulocyte": ["Granulocyte"],
            "vessel": ["Blood vessel"],
            "stroma": ["Stroma"],
            "immune": ["B cell", "CD4 T cell", "CD8 T cell", "Macrophage", "Granulocyte"],
        },
    })

    # LUAD — Sorin 2023 (no fibroblast/stroma class)
    dump("luad_sorin2023", {
        "dataset": "luad_sorin2023",
        "cell_type_col": "cell_type",
        "cv_filter": "cohort == 'Discovery'",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": ["Cancer"],
            "t_cell": ["Tc", "Th", "Treg", "T other"],
            "cd8": ["Tc"],
            "cd4": ["Th", "Treg"],
            "b_cell": ["B cell"],
            "macrophage": ["Alt MAC", "Cl MAC", "Cl Mo", "Int Mo", "Non-Cl Mo"],
            "apc": ["DCs cell", "Alt MAC", "Cl MAC", "Cl Mo", "Int Mo", "Non-Cl Mo"],
            "granulocyte": ["Neutrophils"],
            "vessel": ["Endothelial cell"],
            "immune": [
                "B cell", "Tc", "Th", "Treg", "T other", "NK cell", "Mast cell",
                "DCs cell", "Alt MAC", "Cl MAC", "Cl Mo", "Int Mo", "Non-Cl Mo",
                "Neutrophils",
            ],
        },
    })

    # NSCLC — Aung 2025
    dump("nsclc_aung2025", {
        "dataset": "nsclc_aung2025",
        "cell_type_col": "cell_type",
        "cv_filter": "cohort == 'Yale'",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": ["Tumour", "Ki67"],
            "t_cell": ["CD4", "CD4_TFH", "CD8_cells", "Cytotoxic CD8", "Exhausted CD8", "Treg"],
            "cd8": ["CD8_cells", "Cytotoxic CD8", "Exhausted CD8"],
            "cd4": ["CD4", "CD4_TFH", "Treg"],
            "b_cell": ["B_Cells"],
            "macrophage": ["M1_mac", "M2_mac"],
            "apc": ["DCs", "M1_mac", "M2_mac"],
            "granulocyte": ["Granulocytes"],
            "vessel": ["Vessels"],
            "stroma": ["Stroma"],
            "immune": [
                "B_Cells", "CD4", "CD4_TFH", "CD8_cells", "Cytotoxic CD8", "Exhausted CD8",
                "Treg", "DCs", "M1_mac", "M2_mac", "Granulocytes",
            ],
        },
    })

    # NSCLC — Hoebel 2026 (limited 8-class panel)
    dump("nsclc_gnn_hoebel2026", {
        "dataset": "nsclc_gnn_hoebel2026",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": ["PD-L1+ tumor", "PD-L1- tumor"],
            "t_cell": ["FOXP3+", "PD-1+ (CD8-)", "PD-1+ CD8+", "PD-1- CD8+"],
            "cd8": ["PD-1+ CD8+", "PD-1- CD8+"],
            "immune": [
                "FOXP3+", "PD-1+ (CD8-)", "PD-1+ CD8+", "PD-1- CD8+", "PD-L1+ immune",
            ],
        },
    })

    # NSCLC — Monkman 2024
    dump("nsclc_ici_monkman2024", {
        "dataset": "nsclc_ici_monkman2024",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": ["Tumour", "CD44 Tumour", "HLADR Tumour", "Proliferating Tumour"],
            "t_cell": [
                "CD4 Cells", "CD8 Cells", "Effector CD4", "Treg",
                "CCR7+ CD8/CD4 Cells", "Proliferating Lymphocytes",
            ],
            "cd8": ["CD8 Cells"],
            "cd4": ["CD4 Cells", "Effector CD4", "Treg"],
            "b_cell": ["B Cells"],
            "macrophage": ["Macrophages", "Monocytes"],
            "apc": ["Macrophages", "Monocytes"],
            "granulocyte": ["Granulocytes"],
            "vessel": ["Blood Vessels", "Lymphatics"],
            "stroma": ["Stroma"],
            "immune": [
                "B Cells", "CD4 Cells", "CD8 Cells", "Effector CD4", "Treg",
                "CCR7+ CD8/CD4 Cells", "Lymphocytes", "Proliferating Lymphocytes",
                "Macrophages", "Monocytes", "Mast Cells", "Granulocytes",
            ],
        },
    })

    # TNBC — Wang 2023
    dump("tnbc_wang2023", {
        "dataset": "tnbc_wang2023",
        "cell_type_col": "cell_type",
        "inherit_motifs": "hnc_wu2022",
        "radius_um": 50.0,
        "cell_sets": {
            "tumor": [
                "AR^{+}LAR", "Apoptosis", "Basal", "CA9^+", "CA9^{+}Hypoxia",
                "CD56^{+}NE", "CK8/18^{med}", "CK^{hi}GATA3^{+}", "CK^{lo}GATA3^{+}",
                "PD-L1^{+}GZMB^{+}", "PD-L1^{+}IDO^{+}", "TCF1^{+}",
                "Vimentin^{+}EMT", "pH2AX^{+}DSB", "panCK^{med}",
            ],
            "t_cell": [
                "CD4^+PD1^+T", "CD4^+TCF1^+T", "CD8^+GZMB^+T", "CD8^+PD1^+T_{Ex}",
                "CD8^+T", "CD8^+TCF1^+T", "Treg",
            ],
            "cd8": ["CD8^+GZMB^+T", "CD8^+PD1^+T_{Ex}", "CD8^+T", "CD8^+TCF1^+T"],
            "cd4": ["CD4^+PD1^+T", "CD4^+TCF1^+T", "Treg"],
            "b_cell": ["CD20^+B"],
            "macrophage": ["M2 Mac"],
            "apc": ["DCs", "PD-L1^+APCs", "PD-L1^+IDO^+APCs", "M2 Mac"],
            "granulocyte": ["Neutrophils", "CD15^{+}"],
            "vessel": ["Endothelial"],
            "stroma": ["Fibroblasts", "Myofibroblasts", "PDPN^+Stromal"],
            "immune": [
                "CD20^+B", "CD79a^+Plasma", "CD4^+PD1^+T", "CD4^+TCF1^+T",
                "CD8^+GZMB^+T", "CD8^+PD1^+T_{Ex}", "CD8^+T", "CD8^+TCF1^+T",
                "Treg", "Helios^{+}", "CD56^+NK", "DCs", "M2 Mac",
                "PD-L1^+APCs", "PD-L1^+IDO^+APCs", "Neutrophils", "CD15^{+}",
            ],
        },
    })


if __name__ == "__main__":
    main()
