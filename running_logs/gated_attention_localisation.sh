#!/usr/bin/env bash
set -euo pipefail
source /autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
conda activate p3
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark

for ds in bc_jackson2020 bc_metabric_ali2020 hnc_wu2022 tnbc_wang2023; do
  echo "=== start ${ds} $(date -Is) ==="
  python scripts/eval_ig_mil_window_importance.py \
    --dataset "$ds" \
    --families composition density mixing point_pattern \
    --out "results/pseudo_labels/mil_windows/gated_attention_localisation_${ds}.csv"
  echo "=== done ${ds} $(date -Is) ==="
done
echo "=== all done $(date -Is) ==="
