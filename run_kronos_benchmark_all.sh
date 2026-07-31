#!/usr/bin/env bash

set -uo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

REPO="/autofs/nas8/tywang/tjzou/TME_modeling_benchmark"
DATA_ROOT="/autofs/bal14/zqwu/CellularTables/TME_benchmark_data"

SNAPSHOT="$REPO/models/KRONOS/model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516"
CHECKPOINT="$SNAPSHOT/kronos_vits16_model.pt"
CONFIG="$SNAPSHOT/config.json"
MARKER_METADATA="$REPO/models/KRONOS/model_assets/marker_metadata.csv"

# 必须与 precompute 使用同一个 embedding cache。
CACHE_DIR="$REPO/model_results/KRONOS/embeddings"

LOG_ROOT="$REPO/logs/kronos/benchmark"
OUTPUT_ROOT="$REPO/results/kronos/by_dataset"
SUMMARY="$LOG_ROOT/run_summary.tsv"

DATASETS=(
    bc_jackson2020
    bc_metabric_ali2020
    crc_schurch2020
    crc_wu2022
    hnc_wu2022
    luad_sorin2023
    nsclc_aung2025
    nsclc_gnn_hoebel2026
    nsclc_ici_monkman2024
    tnbc_wang2023
)

get_image_mode() {
    case "$1" in
        bc_jackson2020)
            echo "native"
            ;;
        bc_metabric_ali2020)
            echo "native"
            ;;
        luad_sorin2023)
            echo "native"
            ;;
        nsclc_ici_monkman2024)
            echo "native"
            ;;
        *)
            echo "rasterized"
            ;;
    esac
}

cd "$REPO" || exit 1

mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_ROOT"

printf "dataset\tmode\tstarted\tfinished\texit_code\tresult_rows\toutput\n" \
  > "$SUMMARY"

echo "============================================================"
echo "KRONOS formal sequential benchmark"
echo "Started: $(date)"
echo "Physical GPU: $CUDA_VISIBLE_DEVICES"
echo "Seeds: 0 1 2"
echo "Embedding cache: $CACHE_DIR"
echo "Datasets: ${#DATASETS[@]}"
echo "============================================================"

for dataset in "${DATASETS[@]}"; do
    mode=$(get_image_mode "$dataset")
    started=$(date '+%Y-%m-%d %H:%M:%S')

    output="$OUTPUT_ROOT/${dataset}_benchmark.csv"
    log_file="$LOG_ROOT/${dataset}.log"

    extra_args=()

    if [ "$dataset" = "tnbc_wang2023" ]; then
        extra_args+=(--available-regions-only)
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "Mode: $mode"
    echo "Started: $started"
    echo "Output: $output"
    echo "Log: $log_file"
    echo "Extra arguments: ${extra_args[*]:-none}"
    echo "------------------------------------------------------------"

    # 删除同名旧结果，避免失败时误把旧 CSV 当成新结果。
    rm -f "$output"

    python -u models/KRONOS/process_local_data.py benchmark \
        --datasets "$dataset" \
        --data-roots "$DATA_ROOT" \
        --image-mode "$mode" \
        --checkpoint "$CHECKPOINT" \
        --cfg-path "$CONFIG" \
        --marker-metadata "$MARKER_METADATA" \
        --cache-dir "$CACHE_DIR" \
        --device cuda:0 \
        --batch-size 1 \
        --seeds 0 1 2 \
        --output "$output" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1

    exit_code=$?
    finished=$(date '+%Y-%m-%d %H:%M:%S')
    result_rows=0

    if [ -f "$output" ]; then
        line_count=$(wc -l < "$output")

        if [ "$line_count" -gt 1 ]; then
            result_rows=$((line_count - 1))
        fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" \
        "$mode" \
        "$started" \
        "$finished" \
        "$exit_code" \
        "$result_rows" \
        "$output" \
        >> "$SUMMARY"

    echo "Finished: $dataset"
    echo "Time: $finished"
    echo "Exit code: $exit_code"
    echo "Result rows: $result_rows"

    if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: command failed; inspect $log_file"
    elif [ "$result_rows" -eq 0 ]; then
        echo "WARNING: command finished but produced no benchmark rows"
        echo "Inspect: $log_file"
    else
        echo "Result: $output"
    fi
done

echo
echo "============================================================"
echo "All KRONOS benchmarks processed"
echo "Finished: $(date)"
echo "Summary: $SUMMARY"
echo "============================================================"
