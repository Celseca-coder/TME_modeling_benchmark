#!/bin/bash
# Verify Lasso / linear-SHAP recovery on selected HNC v2 pseudo-labels.
# Point-pattern by_type is slow on first run; later runs reuse the feature cache.
set -euo pipefail
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data

CONDA_SH=/autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate p3

mkdir -p logs results/pseudo_label_explanations

python -u scripts/verify_pseudo_label_explanations.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --labels results/pseudo_labels/hnc_wu2022_v2.csv \
    --label-version v2 \
    --mode tabular \
    --feature-sources composition density point-pattern \
    --by-type \
    --cache-features results/pseudo_label_explanations/tabular_features.csv \
    --output-dir results/pseudo_label_explanations
