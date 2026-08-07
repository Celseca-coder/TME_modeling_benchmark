#!/usr/bin/env bash
set -euo pipefail

# Sequentially benchmark four spatial-distance configurations:
#   k=1, k=2, k=5, and early-fused k=1+2+5.
#
# Usage:
#   DATA_ROOT=/path/to/TME_benchmark_data \
#     bash scripts/run_spatial_distance_k_sweep.sh
#
# Optional arguments are forwarded to every Python run, for example:
#   DATA_ROOT=/path/to/data \
#     bash scripts/run_spatial_distance_k_sweep.sh \
#       --datasets bc_jackson2020 hnc_wu2022 --seeds 0 1 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/autofs/bal14/zqwu/CellularTables/TME_benchmark_data}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/spatial_distance_k_sweep}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/spatial_distance_k_sweep}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

run_one() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] Starting $label (k-values: $*)"
  "$PYTHON" -u "$ROOT/scripts/run_spatial_distance_baseline.py" \
    --data-root "$DATA_ROOT" \
    --k-values "$@" \
    --output "$RESULT_DIR/${label}.csv" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/${label}.log" 2>&1
  echo "[$(date '+%F %T')] Completed $label"
}

# Forward any script arguments (e.g. --datasets / --seeds) to all four runs.
EXTRA_ARGS=("$@")

run_one spatial_distance_k1 1
run_one spatial_distance_k2 2
run_one spatial_distance_k5 5
run_one spatial_distance_k1_k2_k5 1 2 5

echo "All spatial-distance k configurations completed."
echo "Results: $RESULT_DIR"
echo "Logs:    $LOG_DIR"
