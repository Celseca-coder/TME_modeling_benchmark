#!/bin/bash
# Gated AttnMIL instance explanations on motif labels.
# Localization (window vs geometry) and faithfulness (MORF / LERF / insertion).
#
#   bash scripts/run_mil_explanations.sh
#   NOISY=1 bash scripts/run_mil_explanations.sh
# Smoke (one dataset, one seed, no IG):
#   DATASETS=hnc_wu2022 SEEDS="0" EXPLAINERS="attention single one_removed random" \
#     bash scripts/run_mil_explanations.sh
set -euo pipefail
ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
DATA_ROOT="${DATA_ROOT:-/autofs/bal14/zqwu/CellularTables/TME_benchmark_data}"
PY="${PY:-/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python}"
LABEL_ROOT=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset
NOISY="${NOISY:-0}"
GPU="${GPU:-0}"
SEQUENTIAL="${SEQUENTIAL:-1}"
SEEDS="${SEEDS:-0 1 2}"
EXPLAINERS="${EXPLAINERS:-attention single one_removed ig random}"
DATASETS="${DATASETS:-hnc_wu2022 bc_jackson2020 bc_metabric_ali2020 tnbc_wang2023}"
LOG_DIR="$ROOT/results/nohup_logs/mil_explain"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$LOG_DIR"

if [[ "$NOISY" == "1" ]]; then
  TAG="_noisy"
  LABEL_FLAG=(--labels "$LABEL_ROOT/{dataset}_v2_noisy.csv")
  OUT=results/pseudo_label_explanations_panel/mil_noisy
else
  TAG=""
  LABEL_FLAG=(--labels results/pseudo_labels/{dataset}_v2.csv)
  OUT=results/pseudo_label_explanations_panel/mil
fi

declare -A TASKS=(
  [bc_jackson2020]="motif_tumor_high motif_cd8_high motif_t_tumor_mixing motif_cd8_tumor_contact motif_macrophage_tumor_niche motif_apc_t_contact"
  [hnc_wu2022]="motif_cd8_clustering motif_immune_exclusion motif_tumor_high motif_cd8_high"
  [bc_metabric_ali2020]="motif_tumor_stroma_mixing motif_interface_immune motif_tumor_high motif_cd8_high"
  [tnbc_wang2023]="motif_cd8_high motif_tumor_high"
)

for ds in $DATASETS; do
  name="mil_${ds}${TAG}"
  echo "Starting $name"
  nohup "$PY" -u scripts/verify_pseudo_label_explanations.py \
    --dataset "$ds" \
    --tasks ${TASKS[$ds]} \
    "${LABEL_FLAG[@]}" \
    --label-version v2 \
    --mode mil \
    --feature-groups composition mixing \
    --mil-explainers $EXPLAINERS \
    --seeds $SEEDS \
    --data-root "$DATA_ROOT" \
    --device cuda \
    --output-dir "$OUT" \
    > "$LOG_DIR/${name}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
  if [[ "$SEQUENTIAL" == "1" ]]; then
    wait
  fi
done

echo "Logs: $LOG_DIR"
echo "Output: $OUT/<dataset>/{fold_recovery,summary}.csv"
