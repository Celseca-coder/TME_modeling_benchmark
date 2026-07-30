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

LOG_ROOT="$REPO/logs/kronos/precompute"
RESULT_ROOT="$REPO/model_results/KRONOS/precompute_by_dataset"
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
mkdir -p "$RESULT_ROOT"

printf "dataset\tmode\tstarted\tfinished\texit_code\tok\terror\tmanifest\n" \
  > "$SUMMARY"

echo "============================================================"
echo "KRONOS full sequential precompute"
echo "Started: $(date)"
echo "Physical GPU: $CUDA_VISIBLE_DEVICES"
echo "Datasets: ${#DATASETS[@]}"
echo "============================================================"

for dataset in "${DATASETS[@]}"; do
    mode=$(get_image_mode "$dataset")
    started=$(date '+%Y-%m-%d %H:%M:%S')

    dataset_results="$RESULT_ROOT/$dataset"
    manifest="$dataset_results/precompute_manifest.csv"
    log_file="$LOG_ROOT/${dataset}.log"

    mkdir -p "$dataset_results"

    extra_args=()

    if [ "$dataset" = "tnbc_wang2023" ]; then
        extra_args+=(--available-regions-only)
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "Mode: $mode"
    echo "Started: $started"
    echo "Log: $log_file"
    echo "Results: $dataset_results"
    echo "Extra arguments: ${extra_args[*]:-none}"
    echo "------------------------------------------------------------"

    python -u models/KRONOS/process_local_data.py precompute \
        --datasets "$dataset" \
        --data-roots "$DATA_ROOT" \
        --image-mode "$mode" \
        --checkpoint "$CHECKPOINT" \
        --cfg-path "$CONFIG" \
        --marker-metadata "$MARKER_METADATA" \
        --device cuda:0 \
        --batch-size 1 \
        --results-root "$dataset_results" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1

    exit_code=$?
    finished=$(date '+%Y-%m-%d %H:%M:%S')

    ok_count=0
    error_count=0

    if [ -f "$manifest" ]; then
        ok_count=$(
            awk -F, 'NR > 1 && $3 == "ok" {count++} END {print count+0}' \
              "$manifest"
        )

        error_count=$(
            awk -F, 'NR > 1 && $3 == "error" {count++} END {print count+0}' \
              "$manifest"
        )
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" \
        "$mode" \
        "$started" \
        "$finished" \
        "$exit_code" \
        "$ok_count" \
        "$error_count" \
        "$manifest" \
        >> "$SUMMARY"

    echo "Finished: $dataset"
    echo "Time: $finished"
    echo "Exit code: $exit_code"
    echo "Successful regions: $ok_count"
    echo "Failed regions: $error_count"

    if [ "$exit_code" -ne 0 ] || [ "$error_count" -gt 0 ]; then
        echo "WARNING: inspect $log_file"
    fi
done

echo
echo "============================================================"
echo "All datasets processed"
echo "Finished: $(date)"
echo "Summary: $SUMMARY"
echo "============================================================"
