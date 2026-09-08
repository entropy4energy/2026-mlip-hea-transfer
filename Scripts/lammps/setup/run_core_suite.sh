#!/usr/bin/env bash
# Run preflight, 0 K relaxation, EOS points, and elastic finite differences.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${SCRIPT_DIR}/run_lammps.sh"
PYTHON_EXE="${PYTHON_EXE:-python3}"

if (( $# != 4 )); then
    printf 'Usage: %s ARCHITECTURE FROZEN_MODEL INITIAL_DATA OUTPUT_ROOT\n' "$0" >&2
    exit 2
fi

ARCHITECTURE="$1"
MODEL="$(realpath -e -- "$2")"
INITIAL_DATA="$(realpath -e -- "$3")"
OUTPUT_ROOT="$4"

case "$ARCHITECTURE" in
    DPA2|DPA3|DPA4) ;;
    *) printf 'ERROR: architecture must be DPA2, DPA3, or DPA4\n' >&2; exit 2 ;;
esac

mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"

"$RUNNER" "${PROJECT_ROOT}/inputs/01_preflight.in" "${OUTPUT_ROOT}/preflight" \
    -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$INITIAL_DATA"

"$RUNNER" "${PROJECT_ROOT}/inputs/02_relax.in" "${OUTPUT_ROOT}/relax" \
    -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$INITIAL_DATA" \
    -var CELL_RELAX "${CELL_RELAX:-tri}" -var CELL_CYCLES "${CELL_CYCLES:-3}" \
    -var FTOL "${FTOL:-1.0e-4}" -var MAX_FINAL_FORCE "${MAX_FINAL_FORCE:-0.01}" \
    -var MAX_PRESSURE_ERROR_GPA "${MAX_PRESSURE_ERROR_GPA:-0.05}" \
    -var MAX_STRESS_ERROR_GPA "${MAX_STRESS_ERROR_GPA:-0.05}"

RELAXED_DATA="${OUTPUT_ROOT}/relax/relaxed.data"
EOS_SCALES="${EOS_SCALES:-0.970 0.980 0.990 1.000 1.010 1.020 1.030}"
for scale in $EOS_SCALES; do
    tag="${scale//./p}"
    "$RUNNER" "${PROJECT_ROOT}/inputs/03_eos_point.in" "${OUTPUT_ROOT}/eos/scale_${tag}" \
        -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$RELAXED_DATA" \
        -var SCALE "$scale" -var FTOL "${FTOL:-1.0e-4}"
done

ELASTIC_STRAINS="${ELASTIC_STRAINS:-0.0025 0.0050}"
for magnitude in $ELASTIC_STRAINS; do
    mag_tag="${magnitude//./p}"
    for mode in 1 2 3 4 5 6; do
        for sign in -1 1; do
            if (( sign < 0 )); then
                strain="-${magnitude}"
                sign_tag=minus
            else
                strain="$magnitude"
                sign_tag=plus
            fi
            "$RUNNER" "${PROJECT_ROOT}/inputs/04_elastic_point.in" \
                "${OUTPUT_ROOT}/elastic/strain_${mag_tag}/mode_${mode}_${sign_tag}" \
                -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$RELAXED_DATA" \
                -var MODE "$mode" -var STRAIN "$strain" -var FTOL "${FTOL:-1.0e-4}"
        done
    done
done

"$PYTHON_EXE" "${PROJECT_ROOT}/analysis/analyze_eos.py" \
    --input "${OUTPUT_ROOT}/eos" --output "${OUTPUT_ROOT}/eos_analysis"
"$PYTHON_EXE" "${PROJECT_ROOT}/analysis/analyze_elastic.py" \
    --input "${OUTPUT_ROOT}/elastic" --output "${OUTPUT_ROOT}/elastic_analysis"

printf 'Core property suite completed for %s in %s\n' "$ARCHITECTURE" "$OUTPUT_ROOT"
