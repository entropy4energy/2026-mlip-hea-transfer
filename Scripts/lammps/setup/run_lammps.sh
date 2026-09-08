#!/usr/bin/env bash
# Run one LAMMPS input in an isolated output directory and record provenance.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FINALIZER="${SCRIPT_DIR}/finalize_run.py"
PYTHON_EXE="${PYTHON_EXE:-python3}"
# shellcheck source=deepmd_env.sh
source "${SCRIPT_DIR}/deepmd_env.sh"

if (( $# < 2 )); then
    printf 'Usage: %s INPUT_FILE OUTPUT_DIRECTORY [LAMMPS options, e.g. -var MODEL /abs/model.pth]\n' "$0" >&2
    exit 2
fi

INPUT_FILE="$(realpath -e -- "$1")"
OUTPUT_REQUEST="$2"
shift 2

# Command-line index variables override the portable defaults inside inputs.
# Supply the plugin derived from DEEPMD_ROOT unless the caller did so explicitly.
LAMMPS_ARGS=("$@")
plugin_is_set=0
for ((i=0; i+2<${#LAMMPS_ARGS[@]}; i++)); do
    if [[ "${LAMMPS_ARGS[$i]}" == -var && "${LAMMPS_ARGS[$((i+1))]}" == PLUGIN ]]; then
        plugin_is_set=1
        break
    fi
done
if (( plugin_is_set == 0 )); then
    LAMMPS_ARGS+=(-var PLUGIN "$DEEPMD_PLUGIN")
fi

if [[ -d "$OUTPUT_REQUEST" ]] && find "$OUTPUT_REQUEST" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    if [[ "${ALLOW_EXISTING:-0}" != 1 ]]; then
        printf 'ERROR: output directory is not empty (set ALLOW_EXISTING=1 only for an intentional continuation): %s\n' "$OUTPUT_REQUEST" >&2
        exit 2
    fi
    if [[ -z "${RESTART_PARENT_RECEIPT:-}" || ! -r "${RESTART_PARENT_RECEIPT}" ]]; then
        printf 'ERROR: an intentional continuation requires RESTART_PARENT_RECEIPT\n' >&2
        exit 2
    fi
fi
mkdir -p -- "$OUTPUT_REQUEST"
OUTPUT_DIR="$(cd -- "$OUTPUT_REQUEST" && pwd -P)"

MPI_RANKS="${MPI_RANKS:-1}"
if ! [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: MPI_RANKS must be a positive integer\n' >&2
    exit 2
fi
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-}"
RUN_TIMEOUT_KILL_AFTER_SECONDS="${RUN_TIMEOUT_KILL_AFTER_SECONDS:-120}"
if [[ -n "$RUN_TIMEOUT_SECONDS" ]] && ! [[ "$RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: RUN_TIMEOUT_SECONDS must be empty or a positive integer\n' >&2
    exit 2
fi
if ! [[ "$RUN_TIMEOUT_KILL_AFTER_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: RUN_TIMEOUT_KILL_AFTER_SECONDS must be a positive integer\n' >&2
    exit 2
fi
if [[ -n "$RUN_TIMEOUT_SECONDS" ]] && ! command -v timeout >/dev/null 2>&1; then
    printf 'ERROR: RUN_TIMEOUT_SECONDS requires GNU timeout\n' >&2
    exit 2
fi

MANIFEST="${OUTPUT_DIR}/run_manifest.tsv"
{
    printf 'key\tvalue\n'
    printf 'started\t%s\n' "$(date --iso-8601=seconds)"
    printf 'started_utc\t%s\n' "$(date --utc --iso-8601=seconds)"
    printf 'host\t%s\n' "$(hostname -s)"
    printf 'uid\t%s\n' "$(id -u)"
    printf 'campaign_id\t%s\n' "${CAMPAIGN_ID:-not_set}"
    printf 'run_id\t%s\n' "${RUN_ID:-not_set}"
    printf 'model_id\t%s\n' "${MODEL_ID:-not_set}"
    printf 'phase\t%s\n' "${PHASE:-not_set}"
    printf 'training_parent\t%s\n' "${TRAINING_PARENT:-not_set}"
    printf 'atom_type_contract\t1=Hf,2=Mo,3=Ta,4=Ti,5=Zr\n'
    printf 'chemical_seed\t%s\n' "${CHEMICAL_SEED:-not_set}"
    printf 'velocity_seed\t%s\n' "${VELOCITY_SEED:-not_set}"
    printf 'input\t%s\n' "$INPUT_FILE"
    printf 'input_sha256\t%s\n' "$(sha256sum "$INPUT_FILE" | awk '{print $1}')"
    printf 'output_directory\t%s\n' "$OUTPUT_DIR"
    printf 'lammps_executable\t%s\n' "$LAMMPS_EXE"
    printf 'lammps_executable_sha256\t%s\n' "$(sha256sum "$LAMMPS_EXE" | awk '{print $1}')"
    printf 'lammps_version\t%s\n' "$("$LAMMPS_EXE" -h 2>&1 | sed -n '1,2p' | tr '\n\t' '  ')"
    printf 'dp_version\t%s\n' "$($DP_EXE --version 2>&1 | tail -n 1)"
    printf 'deepmd_plugin\t%s\n' "$DEEPMD_PLUGIN"
    printf 'deepmd_plugin_sha256\t%s\n' "$(sha256sum "$DEEPMD_PLUGIN" | awk '{print $1}')"
    printf 'runner_sha256\t%s\n' "$(sha256sum "${SCRIPT_DIR}/run_lammps.sh" | awk '{print $1}')"
    printf 'finalizer_sha256\t%s\n' "$(sha256sum "$FINALIZER" | awk '{print $1}')"
    printf 'environment_script_sha256\t%s\n' "$(sha256sum "${SCRIPT_DIR}/deepmd_env.sh" | awk '{print $1}')"
    printf 'mpi_ranks\t%s\n' "$MPI_RANKS"
    printf 'run_timeout_seconds\t%s\n' "${RUN_TIMEOUT_SECONDS:-none}"
    printf 'run_timeout_kill_after_seconds\t%s\n' "${RUN_TIMEOUT_KILL_AFTER_SECONDS}"
    printf 'cuda_visible_devices\t%s\n' "${CUDA_VISIBLE_DEVICES:-not_set}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'gpu_inventory\t%s\n' "$(nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader | tr '\n\t' '; ')"
        selected_gpu_index="${CUDA_VISIBLE_DEVICES%%,*}"
        if [[ "$selected_gpu_index" =~ ^[0-9]+$ ]]; then
            printf 'selected_gpu\t%s\n' "$(nvidia-smi -i "$selected_gpu_index" --query-gpu=index,name,uuid,driver_version --format=csv,noheader | tr '\n\t' '; ')"
        fi
    fi
    printf 'omp_num_threads\t%s\n' "$OMP_NUM_THREADS"
    if [[ -n "${PROTOCOL_PATH:-}" && -r "${PROTOCOL_PATH}" ]]; then
        printf 'protocol_path\t%s\n' "$(realpath -e -- "$PROTOCOL_PATH")"
        printf 'protocol_sha256\t%s\n' "$(sha256sum "$PROTOCOL_PATH" | awk '{print $1}')"
    fi
    if [[ -n "${MODEL_MANIFEST:-}" && -r "${MODEL_MANIFEST}" ]]; then
        printf 'model_manifest\t%s\n' "$(realpath -e -- "$MODEL_MANIFEST")"
        printf 'model_manifest_sha256\t%s\n' "$(sha256sum "$MODEL_MANIFEST" | awk '{print $1}')"
    fi
    if [[ -n "${RESTART_PARENT_RECEIPT:-}" && -r "${RESTART_PARENT_RECEIPT}" ]]; then
        printf 'restart_parent_receipt\t%s\n' "$(realpath -e -- "$RESTART_PARENT_RECEIPT")"
        printf 'restart_parent_receipt_sha256\t%s\n' "$(sha256sum "$RESTART_PARENT_RECEIPT" | awk '{print $1}')"
    fi
    printf 'arguments\t'
    printf '%q ' "${LAMMPS_ARGS[@]}"
    printf '\n'
} > "$MANIFEST"

ARGS=("${LAMMPS_ARGS[@]}")
for ((i=0; i+2<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == -var ]]; then
        var_name="${ARGS[$((i+1))]}"
        var_value="${ARGS[$((i+2))]}"
        printf 'var_%s\t%s\n' "${var_name,,}" "$var_value" >> "$MANIFEST"
        if [[ "$var_name" == MODEL* || "$var_name" == DATA || "$var_name" == PLUGIN ]] && [[ -f "$var_value" ]]; then
            resolved_value="$(realpath -e -- "$var_value")"
            printf '%s\t%s\n' "${var_name,,}_sha256" "$(sha256sum "$resolved_value" | awk '{print $1}')" >> "$MANIFEST"
            printf '%s\t%s\n' "${var_name,,}_path" "$resolved_value" >> "$MANIFEST"
        fi
    fi
done

cd -- "$OUTPUT_DIR"
if (( MPI_RANKS == 1 )); then
    LAMMPS_COMMAND=("$LAMMPS_EXE" -in "$INPUT_FILE" -log log.lammps -screen screen.log "${LAMMPS_ARGS[@]}")
else
    LAMMPS_COMMAND=("${MPI_LAUNCHER:-mpirun}" -np "$MPI_RANKS" "$LAMMPS_EXE" \
        -in "$INPUT_FILE" -log log.lammps -screen screen.log "${LAMMPS_ARGS[@]}")
fi
set +e
if [[ -n "$RUN_TIMEOUT_SECONDS" ]]; then
    timeout --signal=TERM --kill-after="${RUN_TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${RUN_TIMEOUT_SECONDS}s" "${LAMMPS_COMMAND[@]}"
    return_code=$?
else
    "${LAMMPS_COMMAND[@]}"
    return_code=$?
fi
set -e

printf 'finished\t%s\n' "$(date --iso-8601=seconds)" >> "$MANIFEST"
printf 'finished_utc\t%s\n' "$(date --utc --iso-8601=seconds)" >> "$MANIFEST"
printf 'return_code\t%s\n' "$return_code" >> "$MANIFEST"
set +e
"$PYTHON_EXE" "$FINALIZER" --output "$OUTPUT_DIR" --manifest "$MANIFEST" \
    --return-code "$return_code"
finalizer_code=$?
set -e
if (( return_code != 0 )); then
    printf 'LAMMPS failed with return code %d; see %s\n' "$return_code" "${OUTPUT_DIR}/screen.log" >&2
fi
if (( return_code != 0 )); then
    exit "$return_code"
fi
exit "$finalizer_code"
