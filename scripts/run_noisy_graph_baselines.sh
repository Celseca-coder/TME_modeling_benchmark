#!/usr/bin/env bash
# In-repo CytoCommunity / Cell-Graph Signature / Cellular Neighborhood
# on PseudoNoisyDataset labels. Imaging stays on TME_benchmark_data.
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
CONDA=/autofs/nas8/tywang/tjzou/Miniconda3
DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
LABEL_ROOT=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset
LOG_DIR="$ROOT/results/nohup_logs/noisy_labels"
OUT_DIR="$ROOT/results/noisy_label_baselines"

CN_PY="$CONDA/envs/p3/bin/python"
CGS_PY="$CONDA/envs/cell_graph/bin/python"
CYTO_PY="$CONDA/envs/cyto_community/bin/python"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Cellular Neighborhood: CPU
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 \
  "$CN_PY" -u scripts/run_noisy_graph_baselines.py \
    --method cellular-neighborhood \
    --data-root "$DATA_ROOT" \
    --label-root "$LABEL_ROOT" \
    --output "$OUT_DIR/cellular-neighborhood.csv" \
  > "$LOG_DIR/cellular-neighborhood.log" 2>&1 &
echo $! | tee "$LOG_DIR/cellular-neighborhood.pid"
echo "Started cellular-neighborhood PID $(cat "$LOG_DIR/cellular-neighborhood.pid")"

# Cell-Graph Signature: this machine only has GPU 0/1 (Eva is on 0).
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
  "$CGS_PY" -u scripts/run_noisy_graph_baselines.py \
    --method cell-graph-signature \
    --data-root "$DATA_ROOT" \
    --label-root "$LABEL_ROOT" \
    --output "$OUT_DIR/cell-graph-signature.csv" \
  > "$LOG_DIR/cell-graph-signature.log" 2>&1 &
echo $! | tee "$LOG_DIR/cell-graph-signature.pid"
echo "Started cell-graph-signature PID $(cat "$LOG_DIR/cell-graph-signature.pid")"

# CytoCommunity: requested GPU 2 does not exist here; share GPU 1.
nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=1 CYTO_COMMUNITY_DEVICE=cuda \
  "$CYTO_PY" -u scripts/run_noisy_graph_baselines.py \
    --method cyto-community \
    --device cuda \
    --data-root "$DATA_ROOT" \
    --label-root "$LABEL_ROOT" \
    --output "$OUT_DIR/cyto-community.csv" \
  > "$LOG_DIR/cyto-community.log" 2>&1 &
echo $! | tee "$LOG_DIR/cyto-community.pid"
echo "Started cyto-community PID $(cat "$LOG_DIR/cyto-community.pid")"

# SORBET: in-repo adapter, same overlay. Do not uncomment while GPU 1 is
# already running cell-graph + cyto. Env: sorbet_gnn.
# nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
#   "$CONDA/envs/sorbet_gnn/bin/python" -u scripts/run_noisy_graph_baselines.py \
#     --method sorbet \
#     --device cuda \
#     --data-root "$DATA_ROOT" \
#     --label-root "$LABEL_ROOT" \
#     --output "$OUT_DIR/sorbet.csv" \
#   > "$LOG_DIR/sorbet.log" 2>&1 &
# echo $! | tee "$LOG_DIR/sorbet.pid"

echo
echo "PIDs:"
echo "  CN    $(cat "$LOG_DIR/cellular-neighborhood.pid")"
echo "  CGS   $(cat "$LOG_DIR/cell-graph-signature.pid")"
echo "  Cyto  $(cat "$LOG_DIR/cyto-community.pid")"
echo "Logs: $LOG_DIR/{cellular-neighborhood,cell-graph-signature,cyto-community}.log"
echo "Outputs: $OUT_DIR"
