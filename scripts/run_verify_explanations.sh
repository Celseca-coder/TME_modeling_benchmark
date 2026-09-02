#!/bin/bash
# Separate interpreters on the 10 selected motifs:
#   1) Lasso on named features (composition + density + type-specific K/L)
#   2) linear SHAP on the same named features (scored independently of Lasso)
#   3) UTAG cellular: native domain portraits (no Lasso/SHAP on UTAG)
#   4) embedding linear probes (KRONOS, Eva)
#   5) GNN-explainer node attributions, if CSVs are present
#
# Same-dataset tumor_high / cd8_high are attached as matched controls for the
# interpreter protocol even when the globally selected control lives elsewhere.
#
# Do not source this file; run it (or copy blocks) from the repo root.
set -euo pipefail
cd /autofs/nas8/tywang/tjzou/TME_modeling_benchmark
export DATA_ROOT=/autofs/bal14/zqwu/CellularTables/TME_benchmark_data
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

CONDA_SH=/autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate p3
PY=/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python

PANEL=results/pseudo_labels_all/selected_catalog.csv
OUT=results/pseudo_label_explanations_panel
FEAT="$OUT/features"
CACHE_UTAG=model_results/UTAG/message_passing_cache
mkdir -p logs "$OUT" "$FEAT"

DATASETS=(bc_jackson2020 hnc_wu2022 bc_metabric_ali2020 tnbc_wang2023)
declare -A DISPLAY=(
    [bc_jackson2020]=BC-Jackson2020
    [hnc_wu2022]=HNC-Wu2022
    [bc_metabric_ali2020]=BC-METABRIC-Ali2020
    [tnbc_wang2023]=TNBC-Wang2023
)

pick_cache() {
    local root="$1"
    "$PY" - "$root" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"missing cache root: {root}")
cands = [p for p in root.iterdir() if p.is_dir()]
if not cands:
    raise SystemExit(f"no signature directories in {root}")
best = max(cands, key=lambda p: sum(1 for _ in p.rglob("*.npz")))
print(best)
PY
}

embedding_root() {
    local kind="$1"   # KRONOS or Eva
    local display="$2"
    local raster="model_results/${kind}/embeddings/rasterized/${display}"
    local native="model_results/${kind}/embeddings/native/${display}"
    if [ -d "$raster" ] && find "$raster" -name '*.npz' -print -quit | grep -q .; then
        echo "$raster"
    elif [ -d "$native" ]; then
        echo "$native"
    else
        echo ""
    fi
}

tasks_for() {
    local ds="$1"
    "$PY" - "$ds" "$PANEL" <<'PY'
import sys
from benchmark.motifs.panel import jobs_by_dataset, load_selected_panel, run_tasks_for
ds, panel = sys.argv[1], sys.argv[2]
print(" ".join(run_tasks_for(jobs_by_dataset(load_selected_panel(panel))[ds])))
PY
}

echo "[$(date '+%F %T')] [0/4] export KRONOS / Eva embeddings"
for ds in "${DATASETS[@]}"; do
    name="${DISPLAY[$ds]}"
    for kind in KRONOS Eva; do
        src=$(echo "$kind" | tr 'A-Z' 'a-z')
        root="$(embedding_root "$kind" "$name")"
        out="$FEAT/${ds}_${src}.csv"
        if [ -z "$root" ]; then
            echo "WARN: no $kind cache for $ds ($name); skip export"
            continue
        fi
        sig="$(pick_cache "$root")"
        echo "export $kind $ds <- $sig"
        "$PY" -u scripts/export_cached_model_features.py \
            --source "$src" \
            --dataset "$ds" \
            --data-root "$DATA_ROOT" \
            --input-root "$sig" \
            --output "$out"
    done
done

echo "[$(date '+%F %T')] [1/4] tabular named features (Lasso and SHAP scored separately)"
"$PY" -u scripts/verify_pseudo_label_explanations.py \
    --panel "$PANEL" \
    --data-root "$DATA_ROOT" \
    --label-version v2 \
    --mode tabular \
    --feature-sources composition density mixing point-pattern \
    --by-type \
    --output-dir "$OUT/tabular"

echo "[$(date '+%F %T')] [2/4] embedding probes (KRONOS, then Eva)"
"$PY" -u scripts/verify_pseudo_label_explanations.py \
    --panel "$PANEL" \
    --data-root "$DATA_ROOT" \
    --label-version v2 \
    --mode embedding \
    --embeddings-csv "$FEAT/{dataset}_kronos.csv" \
    --output-dir "$OUT/kronos_embedding"

"$PY" -u scripts/verify_pseudo_label_explanations.py \
    --panel "$PANEL" \
    --data-root "$DATA_ROOT" \
    --label-version v2 \
    --mode embedding \
    --embeddings-csv "$FEAT/{dataset}_eva.csv" \
    --output-dir "$OUT/eva_embedding"

echo "[$(date '+%F %T')] [3/4] UTAG cellular (native domain portraits; no Lasso/SHAP)"
for ds in "${DATASETS[@]}"; do
    mkdir -p "$OUT/utag_native_portraits/$ds"
    "$PY" -u models/utag/verify_native_domain_portraits.py \
        --dataset "$ds" \
        --data-root "$DATA_ROOT" \
        --labels "results/pseudo_labels/${ds}_v2.csv" \
        --label-version v2 \
        --tasks $(tasks_for "$ds") \
        --n-domains 10 \
        --cache-dir "$CACHE_UTAG" \
        --output-dir "$OUT/utag_native_portraits/$ds"
done

echo "[$(date '+%F %T')] [4/4] GNN-explainer (skipped unless node-attribution CSVs exist)"
GNN_READY=1
for ds in "${DATASETS[@]}"; do
    csv="$OUT/features/${ds}_gnn_explainer.csv"
    if [ ! -f "$csv" ]; then
        echo "WARN: missing $csv"
        GNN_READY=0
    fi
done
if [ "$GNN_READY" -eq 1 ]; then
    "$PY" -u scripts/verify_pseudo_label_explanations.py \
        --panel "$PANEL" \
        --data-root "$DATA_ROOT" \
        --label-version v2 \
        --mode gnn-explainer \
        --explainer-csv "$OUT/features/{dataset}_gnn_explainer.csv" \
        --output-dir "$OUT/gnn_explainer"
else
    echo "GNN-explainer not run. Export region_id,cell_type,importance CSVs to:"
    echo "  $OUT/features/<dataset>_gnn_explainer.csv"
fi

echo "[$(date '+%F %T')] score interpreters separately"
"$PY" -u scripts/summarize_interpreter_panel.py \
    --panel "$PANEL" \
    --input-dir "$OUT"

echo "[$(date '+%F %T')] done"
echo "lasso/shap folds: $OUT/tabular/summary.csv"
echo "kronos:           $OUT/kronos_embedding/embedding_probe_summary.csv"
echo "eva:              $OUT/eva_embedding/embedding_probe_summary.csv"
echo "utag cellular:    $OUT/utag_native_portraits/<dataset>/summary.csv"
echo "gnn:              $OUT/gnn_explainer/gnn_summary.csv"
echo "method table:     $OUT/method_scores/method_summary.csv"
