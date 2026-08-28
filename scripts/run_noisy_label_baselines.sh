#!/usr/bin/env bash
# Linear baselines on PseudoNoisyDataset *labels*.
# DATA_ROOT stays on the imaging tables; labels come from LABEL_ROOT.
# SPACE-GM is not launched.
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
LABEL_ROOT=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset
LOG_DIR="$ROOT/results/nohup_logs/noisy_labels"
OUT_DIR="$ROOT/results/noisy_label_baselines"
PY=/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local method="$1"
  echo "Starting $method"
  nohup "$PY" -u scripts/run_noisy_label_baselines.py \
    --method "$method" \
    --data-root "$DATA_ROOT" \
    --label-root "$LABEL_ROOT" \
    --output "$OUT_DIR/${method}.csv" \
    > "$LOG_DIR/${method}.log" 2>&1 &
  echo $! > "$LOG_DIR/${method}.pid"
}

# Existing linear / patch jobs (already launched on this machine):
# run_one composition
# run_one expression
# run_one density
# run_one spatial-distance
# run_one point-pattern
# run_one mixing
# run_one patch-composition
# run_one patch-expression

# run_one composition-expression
# run_one attention-composition
# run_one attention-expression
# run_one attention-composition-expression

run_one patch-composition-expression

echo "Launched (no SPACE-GM). PIDs:"
ls -1 "$LOG_DIR"/*.pid
echo "Logs: $LOG_DIR"
echo "Outputs: $OUT_DIR"
