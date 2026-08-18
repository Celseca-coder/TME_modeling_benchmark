#!/bin/bash
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

python scripts/generate_pseudo_labels.py --dataset hnc_wu2022 --data-root "$DATA_ROOT"
python scripts/run_pseudo_label_benchmark.py \
    --dataset hnc_wu2022 --data-root "$DATA_ROOT" \
    --feature-sources composition expression composition-expression
