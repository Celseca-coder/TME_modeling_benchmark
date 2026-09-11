#!/usr/bin/env bash
# Tabular Lasso + linear SHAP on named-feature family combinations.
#
# The verify script already accepts several --feature-sources. This runner
# only chooses which subsets to run and gives each combo its own --output-dir
# so cached tabular_features.csv files never collide.
#
#   bash scripts/run_tabular_family_combos.sh              # default: 6 pairs
#   MODE=singles bash scripts/run_tabular_family_combos.sh
#   MODE=pairs SKIP_EXISTING=1 bash scripts/run_tabular_family_combos.sh
#
# Do not source this file; run it from the repo root (or let it cd itself).
set -euo pipefail

ROOT=/autofs/nas8/tywang/tjzou/TME_modeling_benchmark
DATA_ROOT="${DATA_ROOT:-/autofs/bal14/zqwu/CellularTables/TME_benchmark_data}"
PY="${PY:-/autofs/nas8/tywang/tjzou/Miniconda3/envs/p3/bin/python}"
PANEL="${PANEL:-results/pseudo_labels_all/selected_catalog.csv}"
OUT="${OUT:-results/pseudo_label_explanations_panel}"
MODE="${MODE:-pairs}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs "$OUT"

FAMILIES=(composition density mixing point-pattern)

combo_tag() {
  local spec="$1"
  echo "${spec// /_}" | tr '-' '_'
}

emit_combos() {
  case "$MODE" in
    singles)
      printf '%s\n' "${FAMILIES[@]}"
      ;;
    pairs)
      # Cheap pairs first; anything with point-pattern last.
      echo "composition density"
      echo "composition mixing"
      echo "density mixing"
      echo "composition point-pattern"
      echo "density point-pattern"
      echo "mixing point-pattern"
      ;;
    all)
      emit_combos_from_python all
      ;;
    *)
      echo "Unknown MODE=$MODE (use singles, pairs, or all)" >&2
      exit 1
      ;;
  esac
}

emit_combos_from_python() {
  "$PY" - "$1" <<'PY'
import itertools
import sys
families = ["composition", "density", "mixing", "point-pattern"]
mode = sys.argv[1]
sizes = range(1, 5) if mode == "all" else (2,)
for k in sizes:
    for combo in itertools.combinations(families, k):
        print(" ".join(combo))
PY
}

run_one() {
  local spec="$1"
  local tag
  tag="$(combo_tag "$spec")"
  local dest="$OUT/tabular_${tag}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$dest/summary.csv" ]]; then
    echo "======== SKIP ${spec} (found ${dest}/summary.csv) ========"
    return
  fi
  echo "======== START ${spec} $(date) ========"
  # --by-type only changes point-pattern; harmless for the other families.
  # shellcheck disable=SC2086
  "$PY" -u scripts/verify_pseudo_label_explanations.py \
    --panel "$PANEL" \
    --data-root "$DATA_ROOT" \
    --label-version v2 \
    --mode tabular \
    --rules tabular \
    --by-type \
    --feature-sources ${spec} \
    --output-dir "$dest"
  echo "======== END ${spec} $(date) ========"
}

echo "MODE=${MODE}  DATA_ROOT=${DATA_ROOT}  OUT=${OUT}"
while IFS= read -r spec; do
  [[ -z "$spec" ]] && continue
  run_one "$spec"
done < <(emit_combos)
echo "======== ALL ${MODE} DONE $(date) ========"
