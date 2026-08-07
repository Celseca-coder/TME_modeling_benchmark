"""Curated dataset/task/validation-scheme selection shared by benchmark runners."""

# dataset, task, validation scheme, expected primary metric
SELECTED_RUNS = (
    ("bc_jackson2020", "clinical_type", "Basel_to_Zurich", "balanced_acc"),
    ("bc_jackson2020", "clinical_type", "cv", "balanced_acc"),
    ("bc_jackson2020", "response", "cv", "auc_roc"),
    ("bc_metabric_ali2020", "DSS", "cv", "c_index"),
    ("bc_metabric_ali2020", "ER_status", "cv", "auc_roc"),
    ("crc_schurch2020", "CLR_DII", "cv", "auc_roc"),
    ("crc_schurch2020", "OS", "cv", "c_index"),
    ("hnc_wu2022", "OS", "cv", "c_index"),
    ("hnc_wu2022", "hpv_status", "cv", "auc_roc"),
    ("hnc_wu2022", "primary_outcome", "UPMC_to_DFCI", "auc_roc"),
    ("hnc_wu2022", "primary_outcome", "cv", "auc_roc"),
    ("luad_sorin2023", "OS", "cv", "c_index"),
    ("nsclc_aung2025", "immunotherapy_response", "Yale_to_UQ", "auc_roc"),
    ("nsclc_aung2025", "immunotherapy_response", "Yale_to_YaleExt", "auc_roc"),
    ("nsclc_gnn_hoebel2026", "OS", "cv", "c_index"),
    ("nsclc_gnn_hoebel2026", "stage_binary", "cv", "auc_roc"),
    ("tnbc_wang2023", "pCR_all", "cv", "auc_roc"),
)

SELECTED_TRIPLES = frozenset(row[:3] for row in SELECTED_RUNS)
SELECTED_DATASETS = tuple(dict.fromkeys(row[0] for row in SELECTED_RUNS))
