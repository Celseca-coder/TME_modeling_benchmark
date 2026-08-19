#!/bin/bash
# 1) Expression leakage screen on composition-residualized v2 labels (all 13 motifs).
# 2) Re-run tabular Lasso/SHAP, KRONOS/Eva probes, and native UTAG portraits.
set -euo pipefail
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data

CONDA_SH=/autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate p3

LABELS=results/pseudo_labels/hnc_wu2022_v2.csv
ROUND=results/pseudo_label_explanations_round2
mkdir -p logs "$ROUND"/kronos_embedding "$ROUND"/eva_embedding "$ROUND"/utag_native_portraits "$ROUND"/tabular

echo "[$(date '+%F %T')] [1/5] expression screen of composition-generated v2 labels"
python -u scripts/run_pseudo_label_benchmark.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels "$LABELS" \
    --label-version v2 \
    --feature-sources expression \
    --output results/pseudo_label_benchmark_expression_v2.csv

echo "[$(date '+%F %T')] [2/5] tabular Lasso + linear SHAP"
python -u scripts/verify_pseudo_label_explanations.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels "$LABELS" \
    --label-version v2 \
    --mode tabular \
    --feature-sources composition density point-pattern \
    --by-type \
    --cache-features results/pseudo_label_explanations/tabular_features.csv \
    --output-dir "$ROUND"/tabular

echo "[$(date '+%F %T')] [3/5] KRONOS embedding probe"
python -u scripts/verify_pseudo_label_explanations.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels "$LABELS" \
    --label-version v2 \
    --mode embedding \
    --embeddings-csv results/pseudo_label_explanations/hnc_wu2022_kronos.csv \
    --output-dir "$ROUND"/kronos_embedding

echo "[$(date '+%F %T')] [4/5] Eva embedding probe"
python -u scripts/verify_pseudo_label_explanations.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels "$LABELS" \
    --label-version v2 \
    --mode embedding \
    --embeddings-csv results/pseudo_label_explanations/hnc_wu2022_eva.csv \
    --output-dir "$ROUND"/eva_embedding

echo "[$(date '+%F %T')] [5/5] native UTAG domain portraits"
python -u models/utag/verify_native_domain_portraits.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels "$LABELS" \
    --label-version v2 \
    --output-dir "$ROUND"/utag_native_portraits

echo "[$(date '+%F %T')] done"
echo "expression: results/pseudo_label_benchmark_expression_v2.csv"
echo "interpretability: $ROUND"
