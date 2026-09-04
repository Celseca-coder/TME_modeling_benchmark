#!/usr/bin/env bash
# Linear_mean + attention MIL on motif v2 labels:
# composition / mixing / celltype_density and the three pairwise combos.
#
# Window features are extracted once per dataset×combo and cached.
# At most MAX_JOBS processes run at once (default 2).
#
#   bash scripts/run_mil_motif_panel.sh
#   NOISY=1 bash scripts/run_mil_motif_panel.sh
#   SKIP_EXISTING=1 NOISY=1 MAX_JOBS=2 bash scripts/run_mil_motif_panel.sh
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
DATA_ROOT="${DATA_ROOT:-/autofs/bal14/zqwu/CellularTables/TME_benchmark_data}"
PY="${PY:-/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python}"
LOG_DIR="$ROOT/results/nohup_logs/mil_motif"
OUT_DIR="$ROOT/results/mil_motif_baselines"
NOISY="${NOISY:-0}"
GPU="${GPU:-0}"
MAX_JOBS="${MAX_JOBS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

noisy_flag=()
tag=""
if [[ "$NOISY" == "1" ]]; then
  noisy_flag=(--noisy)
  tag="_noisy"
fi

wait_for_slot() {
  while true; do
    running=0
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    if (( running < MAX_JOBS )); then
      return
    fi
    sleep 30
  done
}

PIDS=()

run_one() {
  local method="$1"
  shift
  local groups=("$@")
  local name="${method}_$(IFS=+; echo "${groups[*]}")${tag}"
  local extra=()
  if [[ "$method" == "attention" ]]; then
    extra=(--device "cuda:0")
    export CUDA_VISIBLE_DEVICES="$GPU"
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$OUT_DIR/${name}.csv" ]]; then
    echo "Skip existing $name"
    return
  fi
  wait_for_slot
  echo "Starting $name"
  nohup "$PY" -u scripts/run_mil_motif_baselines.py \
    --panel \
    --method "$method" \
    --feature-groups "${groups[@]}" \
    --data-root "$DATA_ROOT" \
    --output "$OUT_DIR/${name}.csv" \
    "${noisy_flag[@]}" \
    "${extra[@]}" \
    > "$LOG_DIR/${name}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
  PIDS+=("$!")
}

combos=(
  "composition"
  "celltype_density"
  "composition celltype_density"
  "mixing"
  "composition mixing"
  "mixing celltype_density"
)

for combo in "${combos[@]}"; do
  # shellcheck disable=SC2086
  run_one linear_mean $combo
done
for combo in "${combos[@]}"; do
  # shellcheck disable=SC2086
  run_one attention $combo
done

echo "Launched MIL motif jobs (max $MAX_JOBS). PIDs:"
ls -1 "$LOG_DIR"/*.pid
echo "Logs: $LOG_DIR"
echo "Outputs: $OUT_DIR"
echo "Waiting for remaining jobs..."
for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
  wait "$pid" || true
done
echo "All launched jobs finished."
