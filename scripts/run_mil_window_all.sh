#!/usr/bin/env bash
# Window MIL localisation on every label cut.
#
# Directed cuts train and score all five methods: L1 MIL, gated attention,
# window MLP, RF, and Additive MIL. Noisy labels only.
#   - Extreme20: lowest 20% = 0, highest 20% = 1.
#   - Tertile: lowest third = 0, highest third = 1, in PseudoNoisyDataset.
# Undirected cuts keep the original pair: gated attention and RF.
#   - 1in3 NoDirect and 1in5 NoDirect. Noisy labels only.
#
# Physical GPU 1 is the only visible CUDA device. RF stays on CPU.
#
# The first launch moves existing result CSVs in each output directory to
# archive_before_rerun/, because those files are the previous scores and the
# drivers would otherwise skip them. A later launch leaves the new CSVs in
# place, so a restarted job resumes finished dataset/motif/method rows.
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
TERTILE_WINDOW=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/mil_windows/per_dataset
TERTILE_BAG=/autofs/nas8/tywang/tjzou/PseudoNoisyDataset/per_dataset
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

source /autofs/nas8/tywang/tjzou/Miniconda3/etc/profile.d/conda.sh
conda activate p3
cd "$ROOT"

OUT="$ROOT/results/pseudo_labels/mil_windows"

stash_once() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    local marker="$dir/.rerun_archived"
    if [[ -f "$marker" ]]; then
        return 0
    fi
    mkdir -p "$dir/archive_before_rerun"
    local moved=0
    local f
    for f in "$dir"/*.csv; do
        [[ -e "$f" ]] || continue
        mv "$f" "$dir/archive_before_rerun/"
        moved=$((moved + 1))
    done
    touch "$marker"
    echo "archived ${moved} csvs from ${dir}"
}

run_job() {
    local name="$1"
    shift
    echo "start $(date '+%F %T') ${name}"
    python -u "$@"
    echo "done $(date '+%F %T') ${name}"
}

stash_once "$OUT/extreme20_noisy"
run_job "extreme20 noisy" \
    scripts/eval_extreme20_window_importance.py --method all --label noisy

stash_once "$OUT/tertile_noisy"
run_job "tertile noisy" \
    scripts/eval_extreme20_window_importance.py --method all --label noisy \
    --window-dir "$TERTILE_WINDOW" \
    --bag-dir "$TERTILE_BAG" \
    --out-dir "$OUT/tertile_noisy"

for pack in 1in3 1in5; do
    stash_once "$OUT/nodirect_${pack}_noisy"
    run_job "nodirect ${pack} noisy" \
        scripts/eval_nodirect_1in3_window_importance.py \
        --pack "$pack" --method attention rf --label noisy
done

echo "all packs finished $(date '+%F %T')"
