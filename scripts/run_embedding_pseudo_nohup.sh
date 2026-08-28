#!/usr/bin/env bash
# Frozen Eva / KRONOS / UTAG on PseudoNoisyDataset labels.
# Imaging stays on TME_benchmark_data.
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
CONDA=/autofs/nas8/tywang/tjzou/Miniconda3
DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
LABEL_ROOT=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset
LOG_DIR="$ROOT/results/nohup_logs/pseudo_embeddings"
OUT_DIR="$ROOT/results"

EVA_PY="$CONDA/envs/Eva/bin/python"
KRONOS_PY="$CONDA/envs/kronos_cpython/bin/python"
UTAG_PY="$CONDA/envs/utag_benchmark/bin/python"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "Starting eva on GPU 0"
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  bash -c "
    set -euo pipefail
    cd '$ROOT'
    '$EVA_PY' -u models/Eva/process_local_data.py precompute \
      --image-mode auto --device cuda --batch-size 1 \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --results-root '$ROOT/model_results/Eva'
    '$EVA_PY' -u models/Eva/process_local_data.py benchmark \
      --image-mode auto --device cuda --batch-size 1 \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --output '$OUT_DIR/eva_pseudo_benchmark.csv'
  " > "$LOG_DIR/eva.log" 2>&1 &
echo $! | tee "$LOG_DIR/eva.pid"

echo "Starting kronos on GPU 1"
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
  bash -c "
    set -euo pipefail
    cd '$ROOT'
    '$KRONOS_PY' -u models/KRONOS/process_local_data.py precompute \
      --image-mode auto --device cuda --batch-size 1 \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --results-root '$ROOT/model_results/KRONOS'
    '$KRONOS_PY' -u models/KRONOS/process_local_data.py benchmark \
      --image-mode auto --device cuda --batch-size 1 \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --output '$OUT_DIR/kronos_pseudo_benchmark.csv'
  " > "$LOG_DIR/kronos.log" 2>&1 &
echo $! | tee "$LOG_DIR/kronos.pid"

echo "Starting utag on CPU"
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 \
  bash -c "
    set -euo pipefail
    cd '$ROOT'
    '$UTAG_PY' -u models/utag/process_local_data.py precompute \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --results-root '$ROOT/model_results/UTAG'
    '$UTAG_PY' -u models/utag/process_local_data.py benchmark \
      --feature-mode combined \
      --data-roots '$DATA_ROOT' --label-root '$LABEL_ROOT' \
      --output '$OUT_DIR/utag_pseudo_benchmark.csv'
  " > "$LOG_DIR/utag.log" 2>&1 &
echo $! | tee "$LOG_DIR/utag.pid"

echo "Launched. PIDs:"
cat "$LOG_DIR"/{eva,kronos,utag}.pid
echo "Logs: $LOG_DIR"
echo "Outputs: $OUT_DIR/{eva,kronos,utag}_pseudo_benchmark.csv"
