#!/usr/bin/env bash
# Run independent NPT temperature points and optional transport calculations.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${SCRIPT_DIR}/run_lammps.sh"
PYTHON_EXE="${PYTHON_EXE:-python3}"

if (( $# < 4 || $# > 5 )); then
    printf 'Usage: %s ARCHITECTURE FROZEN_MODEL RELAXED_DATA OUTPUT_ROOT ["T1 T2 ..."]\n' "$0" >&2
    exit 2
fi

ARCHITECTURE="$1"
MODEL="$(realpath -e -- "$2")"
RELAXED_DATA="$(realpath -e -- "$3")"
OUTPUT_ROOT="$4"
TEMPERATURE_LIST="${5:-${TEMPERATURES:-300 600 900 1200}}"
RUN_DIFFUSION="${RUN_DIFFUSION:-1}"
RUN_KAPPA="${RUN_KAPPA:-0}"

case "$ARCHITECTURE" in
    DPA2|DPA3|DPA4) ;;
    *) printf 'ERROR: architecture must be DPA2, DPA3, or DPA4\n' >&2; exit 2 ;;
esac

mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
replica=0
for temperature in $TEMPERATURE_LIST; do
    replica=$((replica + 1))
    seed=$((20261000 + replica * 97))
    npt_dir="${OUTPUT_ROOT}/T_${temperature}/npt"
    "$RUNNER" "${PROJECT_ROOT}/inputs/05_npt_thermo.in" "$npt_dir" \
        -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$RELAXED_DATA" \
        -var TEMP "$temperature" -var SEED "$seed" \
        -var N_EQ "${NPT_N_EQ:-50000}" -var N_PROD "${NPT_N_PROD:-200000}"

    npt_data="${npt_dir}/npt_T${temperature}.data"
    if [[ "$RUN_DIFFUSION" == 1 ]]; then
        diffusion_dir="${OUTPUT_ROOT}/T_${temperature}/diffusion"
        "$RUNNER" "${PROJECT_ROOT}/inputs/06_diffusion_rdf.in" "$diffusion_dir" \
            -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$npt_data" \
            -var TEMP "$temperature" -var SEED "$((seed + 1))" \
            -var N_EQ "${DIFFUSION_N_EQ:-20000}" -var N_PROD "${DIFFUSION_N_PROD:-500000}"
        "$PYTHON_EXE" "${PROJECT_ROOT}/analysis/analyze_transport.py" \
            --input "${diffusion_dir}/msd.dat" --output "${diffusion_dir}/analysis" \
            --fit-fraction "${MSD_FIT_FRACTION:-0.5}"
    fi

    if [[ "$RUN_KAPPA" == 1 ]]; then
        kappa_dir="${OUTPUT_ROOT}/T_${temperature}/kappa"
        "$RUNNER" "${PROJECT_ROOT}/inputs/08_thermal_conductivity.in" "$kappa_dir" \
            -var ARCH "$ARCHITECTURE" -var MODEL "$MODEL" -var DATA "$npt_data" \
            -var TEMP "$temperature" -var SEED "$((seed + 2))" \
            -var N_EQ "${KAPPA_N_EQ:-50000}" -var N_PROD "${KAPPA_N_PROD:-1000000}"
    fi
done

"$PYTHON_EXE" "${PROJECT_ROOT}/analysis/analyze_thermo.py" \
    --input "$OUTPUT_ROOT" --output "${OUTPUT_ROOT}/thermo_analysis"
printf 'Thermal suite completed for %s in %s\n' "$ARCHITECTURE" "$OUTPUT_ROOT"

