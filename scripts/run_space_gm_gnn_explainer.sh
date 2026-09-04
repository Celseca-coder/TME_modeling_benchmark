#!/bin/bash
# SPACE-GM node attributions (noisy v2) -> interpreter recovery.
#
# Step A runs in the space_gm env (official GNN_pred + PyG).
# Step B runs in p3 (verify_pseudo_label_explanations.py).
#
# Do not source this file; run it from the repo root, or copy blocks.
set -euo pipefail
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SPACE_PY=/autofs/nas8/tywang/tjzou/Miniconda3/envs/space_gm/bin/python
P3_PY=/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python
LABEL_ROOT=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset
CKPT=model_results/SPACE_GM_pseudo
FEAT=results/pseudo_label_explanations_panel/features
LOG=results/nohup_logs/gnn_explainer
mkdir -p "$FEAT" "$LOG"

# 10 selected motifs only (no extra tumor_high / cd8_high).
declare -A TASKS=(
    [bc_jackson2020]="motif_tumor_high motif_t_tumor_mixing motif_cd8_tumor_contact motif_macrophage_tumor_niche motif_apc_t_contact"
    [hnc_wu2022]="motif_cd8_clustering motif_immune_exclusion"
    [bc_metabric_ali2020]="motif_tumor_stroma_mixing motif_interface_immune"
    [tnbc_wang2023]="motif_cd8_high"
)

# Default: one checkpoint per task (seed0 fold0). Set ALL_FOLDS=1 to average all 15.
SEED_ARGS=(--seeds 0 --folds 0)
if [ "${ALL_FOLDS:-0}" = "1" ]; then
    SEED_ARGS=()
fi

export_one() {
    local ds="$1" method="$2"
    local csv="$FEAT/${ds}_${method//-/_}.csv"
    echo "[$(date '+%F %T')] export $ds  method=$method  ${SEED_ARGS[*]:-all folds}"
    "$SPACE_PY" -u scripts/export_space_gm_node_importance.py \
        --dataset "$ds" \
        --tasks ${TASKS[$ds]} \
        --method "$method" \
        --ckpt-root "$CKPT" \
        --data-root "$DATA_ROOT" \
        --device cuda \
        "${SEED_ARGS[@]}" \
        --output "$csv"
}

verify_one() {
    local ds="$1" method="$2"
    local tag="${method//-/_}"
    local csv="$FEAT/${ds}_${tag}.csv"
    echo "[$(date '+%F %T')] verify $ds  $method"
    "$P3_PY" -u scripts/verify_pseudo_label_explanations.py \
        --dataset "$ds" \
        --tasks ${TASKS[$ds]} \
        --labels "$LABEL_ROOT/${ds}_v2_noisy.csv" \
        --label-version v2 \
        --mode gnn-explainer \
        --explainer-csv "$csv" \
        --data-root "$DATA_ROOT" \
        --output-dir "results/pseudo_label_explanations_panel/${tag}_noisy"
}

# ---- smoke (one model, few regions) ----
# "$SPACE_PY" -u scripts/export_space_gm_node_importance.py \
#     --dataset hnc_wu2022 --tasks motif_cd8_clustering \
#     --method gnn-explainer --max-models 1 --max-regions 2 \
#     --data-root "$DATA_ROOT" --device cuda \
#     --output "$FEAT/hnc_wu2022_gnn_explainer_smoke.csv"

# ---- full export: GNNExplainer (default) ----
for ds in bc_jackson2020 hnc_wu2022 bc_metabric_ali2020 tnbc_wang2023; do
    export_one "$ds" gnn-explainer
done

# ---- optional: IG / occlusion (same checkpoints, separate CSVs) ----
# for ds in bc_jackson2020 hnc_wu2022 bc_metabric_ali2020 tnbc_wang2023; do
#     export_one "$ds" ig
#     # occlusion is a forward pass per cell; skip unless you need it
#     # export_one "$ds" occlusion
# done

# ---- verify (p3) ----
for ds in bc_jackson2020 hnc_wu2022 bc_metabric_ali2020 tnbc_wang2023; do
    verify_one "$ds" gnn-explainer
done

# for ds in bc_jackson2020 hnc_wu2022 bc_metabric_ali2020 tnbc_wang2023; do
#     verify_one "$ds" ig
#     # verify_one "$ds" occlusion
# done

echo "[$(date '+%F %T')] done"
echo "CSVs:     $FEAT/<dataset>_gnn_explainer.csv"
echo "recovery: results/pseudo_label_explanations_panel/gnn_explainer_noisy/<dataset>/"
