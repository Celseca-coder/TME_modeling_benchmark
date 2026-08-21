#!/usr/bin/env bash
# Generate v2 pseudo labels and run composition / expression / density screening
# on every motif-catalog dataset except HNC (already frozen) and Hoebel (2863 ROIs).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/autofs/bal14/zqwu/CellularTables/TME_benchmark_data}"
N_JOBS="${N_JOBS:-8}"
LOG_DIR="$ROOT/logs/motif_screening"
mkdir -p "$LOG_DIR" results/pseudo_labels results/motif_screening

DATASETS=(
  nsclc_ici_monkman2024
  nsclc_aung2025
  crc_schurch2020
  crc_wu2022
  luad_sorin2023
  bc_metabric_ali2020
  bc_jackson2020
  tnbc_wang2023
)

BENCH="$ROOT/results/motif_screening/benchmark.csv"

if [[ ! -f "$BENCH" ]]; then
  "$PY" - "$ROOT" "$BENCH" << 'PY'
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
files = [
    root / "results/pseudo_label_benchmark_v2.csv",
    root / "results/pseudo_label_benchmark_expression_v2.csv",
    root / "results/pseudo_label_benchmark_spatial.csv",
]
frames = [pd.read_csv(p) for p in files if p.exists()]
if frames:
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["dataset", "task", "feature_source", "scheme"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"seeded {out} with {len(df)} HNC rows", flush=True)
PY
fi

for ds in "${DATASETS[@]}"; do
  log="$LOG_DIR/${ds}.log"
  echo "===== $ds =====" | tee -a "$log"
  "$PY" scripts/generate_pseudo_labels.py \
    --dataset "$ds" --data-root "$DATA_ROOT" \
    --n-jobs "$N_JOBS" --skip-spatial-null \
    2>&1 | tee -a "$log"
  "$PY" scripts/refine_pseudo_labels.py --dataset "$ds" \
    2>&1 | tee -a "$log"
  "$PY" scripts/run_pseudo_label_benchmark.py \
    --dataset "$ds" --data-root "$DATA_ROOT" \
    --labels "$ROOT/results/pseudo_labels/${ds}_v2.csv" \
    --label-version v2 \
    --feature-sources composition expression density \
    --seeds 0 1 2 \
    --output "$BENCH" --append \
    2>&1 | tee -a "$log"
done

"$PY" scripts/select_motif_panel.py --benchmark "$BENCH"
echo "Done. See results/motif_screening/"
