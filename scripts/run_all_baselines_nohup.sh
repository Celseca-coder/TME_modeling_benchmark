#!/usr/bin/env bash
# Keep this script in Unix LF format; it is executed on Linux.
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
LOG_DIR="$ROOT/results/nohup_logs"

mkdir -p "$LOG_DIR"

source /autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
CONDA_BASE=$(conda info --base)
P3_PY="$CONDA_BASE/envs/p3/bin/python"
SPACEGM_PY="$CONDA_BASE/envs/space_gm/bin/python"

check_space_gm_deps() {
  "$SPACEGM_PY" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "torch_geometric", "lifelines", "sklearn") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing dependencies for SPACE-GM baseline:", ", ".join(missing))
    print("Install them first, for example: pip install 'torch>=2.2.2' 'torch-geometric>=2.4.0' lifelines scikit-learn")
    raise SystemExit(1)
PY
}

check_cyto_community_deps() {
  "$SPACEGM_PY" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "torch_geometric") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing dependencies for Cyto-Community baseline:", ", ".join(missing))
    print("Install them first, for example: pip install 'torch>=2.2.2' 'torch-geometric>=2.4.0'")
    raise SystemExit(1)
PY
}

cd "$ROOT"

# echo "Starting composition baseline..."
# nohup python -u scripts/run_global_composition_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/composition_benchmark.csv" \
#   > "$LOG_DIR/composition_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/composition_baseline.pid"

# echo "Starting density baseline..."
# nohup python -u scripts/run_celltype_density_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/celltype_density_benchmark.csv" \
#   > "$LOG_DIR/density_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/density_baseline.pid"

# echo "Starting expression baseline..."
# nohup python -u scripts/run_global_expression_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/expression_benchmark.csv" \
#   > "$LOG_DIR/expression_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/expression_baseline.pid"

# echo "All jobs launched."
# echo "Logs:"
# ls -1 "$LOG_DIR"

# ---- 新增：距离特征基线 ----
# echo "Starting spatial distance baseline..."
# nohup python -u scripts/run_spatial_distance_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/spatial_distance_benchmark.csv" \
#   > "$LOG_DIR/spatial_distance_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/spatial_distance_baseline.pid"

# ---- 新增：点模式统计基线 ----
# echo "Starting point pattern baseline..."
# nohup python -u scripts/run_point_pattern_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/point_pattern_benchmark.csv" \
#   > "$LOG_DIR/point_pattern_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/point_pattern_baseline.pid"

# ---- 新增：混合/浸润评分基线 ----
# echo "Starting mixing baseline..."
# nohup python -u scripts/run_mixing_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/mixing_benchmark.csv" \
#   > "$LOG_DIR/mixing_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/mixing_baseline.pid"

# echo "Starting patch baseline..."
# nohup python -u scripts/run_patch_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/patch_benchmark.csv" \
#   > "$LOG_DIR/patch_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/patch_baseline.pid"

# echo "Starting SPACE-GM baseline..."
# check_space_gm_deps
# nohup "$SPACEGM_PY" -u scripts/run_space_gm_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/space_gm_benchmark.csv" \
#   > "$LOG_DIR/space_gm_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/space_gm_baseline.pid"

# echo "Starting Cellular Neighborhood baseline..."
# nohup "$P3_PY" -u scripts/run_cellular_neighborhood_baseline.py \
#   --data-root "$DATA_ROOT" \
#   --output "$ROOT/results/cellular_neighborhood_benchmark.csv" \
#   > "$LOG_DIR/cellular_neighborhood_baseline.log" 2>&1 &
# echo $! > "$LOG_DIR/cellular_neighborhood_baseline.pid"

echo "Starting Cyto-Community baseline..."
check_cyto_community_deps
nohup "$SPACEGM_PY" -u scripts/run_cyto_community_baseline.py \
  --data-root "$DATA_ROOT" \
  --output "$ROOT/results/cyto_community_benchmark.csv" \
  > "$LOG_DIR/cyto_community_baseline.log" 2>&1 &
echo $! > "$LOG_DIR/cyto_community_baseline.pid"

echo "All jobs launched."
echo "Logs:"
ls -1 "$LOG_DIR"
