#!/bin/bash
# Refine existing HNC motif labels with patient-bootstrap tertiles.
# Does not regenerate scores.
set -euo pipefail
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark

CONDA_SH=/autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate p3

python scripts/refine_pseudo_labels.py \
    --dataset hnc_wu2022 \
    --primary-only \
    --n-boot 1000 \
    --confidence 0.90 \
    --output results/pseudo_labels/hnc_wu2022_v2.csv
