#!/usr/bin/env bash
# Run one uniaxial tension/compression trajectory and reduce the curve.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${SCRIPT_DIR}/run_lammps.sh"
PYTHON_EXE="${PYTHON_EXE:-python3}"

if (( $# != 7 )); then
    printf 'Usage: %s ARCHITECTURE MODEL DATA AXIS RATE_PS_INV TARGET_STRAIN OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

ARCHITECTURE="$1"
MODEL="$(realpath -e -- "$2")"
DATA="$(realpath -e -- "$3")"
AXIS="$4"
RATE="$5"
TARGET_STRAIN="$6"
OUTPUT_DIRECTORY="$7"
DT="${DT:-0.001}"

case "$ARCHITECTURE" in DPA2|DPA3|DPA4) ;; *) printf 'ERROR: invalid architecture\n' >&2; exit 2 ;; esac
case "$AXIS" in x|y|z) ;; *) printf 'ERROR: axis must be x, y, or z\n' >&2; exit 2 ;; esac

N_STEPS="$(awk -v target="$TARGET_STRAIN" -v rate="$RATE" -v dt="$DT" 'BEGIN {
    if (rate == 0 || target == 0 || target*rate < 0) exit 2;
    value = target/(rate*dt);
    if (value < 0) value = -value;
    printf "%d", int(value+0.5);
}')" || {
    printf 'ERROR: RATE and TARGET_STRAIN must be nonzero and have the same sign\n' >&2
    exit 2
}

"$RUNNER" "${PROJECT_ROOT}/inputs/07_uniaxial.in" "$OUTPUT_DIRECTORY" \
    -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$DATA" \
    -var AXIS "$AXIS" -var RATE "$RATE" -var N_STEPS "$N_STEPS" -var DT "$DT" \
    -var TEMP "${TEMP:-300.0}" -var SEED "${SEED:-20260911}" \
    -var N_EQ "${N_EQ:-50000}"

"$PYTHON_EXE" "${PROJECT_ROOT}/analysis/analyze_uniaxial.py" \
    --input "${OUTPUT_DIRECTORY}/stress_strain.dat" \
    --output "${OUTPUT_DIRECTORY}/analysis" \
    --elastic-limit "${ELASTIC_LIMIT:-0.01}" --offset "${YIELD_OFFSET:-0.002}"

