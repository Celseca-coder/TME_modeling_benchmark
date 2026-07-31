#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_ROOT=/path/to/data bash scripts/run_pairwise_combinations_batch.sh
# Optional: PYTHON=/path/to/python SEEDS="0 1 2" DATASETS="hnc_wu2022 luad_sorin2023"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/pairwise}"
LOG_DIR="${LOG_DIR:-$ROOT/results/nohup_logs/pairwise}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

combinations=(
  composition+density composition+expression composition+distance
  composition+point_pattern composition+mixing density+expression
  density+distance density+point_pattern density+mixing expression+distance
  expression+point_pattern expression+mixing distance+point_pattern
  distance+mixing point_pattern+mixing
)

for combination in "${combinations[@]}"; do
  safe_name="${combination/+/_}"
  command=("$PYTHON" -u "$ROOT/scripts/run_pairwise_feature_combinations.py"
           --combinations "$combination" --seeds $SEEDS
           --output "$RESULT_DIR/${safe_name}.csv")
  if [[ -n "$DATA_ROOT" ]]; then command+=(--data-root "$DATA_ROOT"); fi
  if [[ -n "$DATASETS" ]]; then command+=(--datasets $DATASETS); fi
  echo "Running $combination"
  "${command[@]}" > "$LOG_DIR/${safe_name}.log" 2>&1
done

echo "Completed all 15 combinations. Results: $RESULT_DIR"
