#!/bin/bash
# Motif verification using the combination-study spatial features:
# density (broad) and point_pattern (spatial organization).
# Skips mixing (slow, narrowest clinically) and does not regenerate labels.
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

LABELS=results/pseudo_labels/hnc_wu2022.csv
if [ ! -f "$LABELS" ]; then
    echo "Missing $LABELS — run scripts/generate_pseudo_labels.py first." >&2
    exit 1
fi

python scripts/run_pseudo_label_benchmark.py \
    --dataset hnc_wu2022 \
    --data-root "$DATA_ROOT" \
    --feature-sources density point-pattern \
    --primary-tasks \
    --output results/pseudo_label_benchmark_spatial.csv
