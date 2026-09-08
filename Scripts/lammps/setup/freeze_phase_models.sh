#!/usr/bin/env bash
# Export both immutable phase-parent branches for the two-phase campaign.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=deepmd_env.sh
source "${SCRIPT_DIR}/deepmd_env.sh"

TRAINING_RUNS="${TRAINING_RUNS:-./training/runs}"
OUTPUT_REQUEST="${1:-${PROJECT_ROOT}/models/hea_two_phase_v1_20260827}"
mkdir -p -- "$OUTPUT_REQUEST"
OUTPUT_DIR="$(cd -- "$OUTPUT_REQUEST" && pwd -P)"

declare -A EXPECTED_HASHES
EXPECTED_HASHES[DPA2__bcc_parent]=88e5092032e2f61783a8f84529096754a9d33a20b269d56c78bae54df7a29e76
EXPECTED_HASHES[DPA3__bcc_parent]=bb48380ac6110241daafda0b1303f499f539d5850d78203a533fae576150c149
EXPECTED_HASHES[DPA4__bcc_parent]=dde2b2f6830bf09599538e00903cee48cf8f3f786cf3bff0742226962339534c
EXPECTED_HASHES[DPA2__fcc_parent]=0049f9370c72cd940a7b7ce3b493336e8ef88678c0667732534c89ea80d38e92
EXPECTED_HASHES[DPA3__fcc_parent]=6fef25ab2ae602148e3ecce0d4ad6f4a9dd46c647d8fa7d822590c8b8387c6c7
EXPECTED_HASHES[DPA4__fcc_parent]=18fb6b51863a374343849833b979adb7ac96ebf81e21099806f1ef5c3f16ca82

model_ids=(
    DPA2__bcc_parent DPA3__bcc_parent DPA4__bcc_parent
    DPA2__fcc_parent DPA3__fcc_parent DPA4__fcc_parent
)

checkpoint_for() {
    local model_id="$1" architecture parent branch filename
    architecture="${model_id%%__*}"
    parent="${model_id##*__}"
    parent="${parent%_parent}"
    if [[ "$parent" == bcc ]]; then
        branch=bcc_to_fcc
    else
        branch=fcc_to_bcc
    fi
    filename=model.ckpt-500000.pt
    [[ "$architecture" == DPA4 ]] && filename=model_ema.ckpt-500000.pt
    printf '%s/%s__%s__pure_pocc_aimd/ckpt/%s\n' \
        "$TRAINING_RUNS" "$architecture" "$branch" "$filename"
}

run_for() {
    dirname "$(dirname "$(checkpoint_for "$1")")"
}

for model_id in "${model_ids[@]}"; do
    architecture="${model_id%%__*}"
    extension=pth
    [[ "$architecture" == DPA4 ]] && extension=pt2
    checkpoint="$(checkpoint_for "$model_id")"
    run_directory="$(run_for "$model_id")"
    output_model="${OUTPUT_DIR}/${model_id}.${extension}"
    if [[ ! -r "$checkpoint" ]]; then
        printf 'ERROR: frozen checkpoint is absent: %s\n' "$checkpoint" >&2
        exit 4
    fi
    if [[ ! -r "${run_directory}/run_status.json" || ! -r "${run_directory}/input.json" ]]; then
        printf 'ERROR: training provenance is incomplete for %s\n' "$model_id" >&2
        exit 4
    fi
    if ! "$DEEPMD_PYTHON" - "$run_directory/run_status.json" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
if status.get("state") != "completed" or status.get("return_code") != 0:
    raise SystemExit(1)
PY
    then
        printf 'ERROR: training run is not completed with return code zero: %s\n' "$run_directory" >&2
        exit 4
    fi
    actual_hash="$(sha256sum "$checkpoint" | awk '{print $1}')"
    if [[ "$actual_hash" != "${EXPECTED_HASHES[$model_id]}" ]]; then
        printf 'ERROR: checkpoint hash mismatch for %s: %s\n' "$model_id" "$actual_hash" >&2
        exit 4
    fi
    if [[ -e "$output_model" ]]; then
        printf 'ERROR: refusing to overwrite frozen model: %s\n' "$output_model" >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT_DIR}/manifest.tsv" ]]; then
    printf 'ERROR: refusing to overwrite model manifest: %s\n' "${OUTPUT_DIR}/manifest.tsv" >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
    printf 'ERROR: DPA4 .pt2 artifacts must be frozen on the deployment GPU\n' >&2
    exit 2
fi
cuda_available="$("$DEEPMD_PYTHON" -c 'import torch; print(int(torch.cuda.is_available()))')"
if [[ "$cuda_available" != 1 ]]; then
    printf 'ERROR: model freeze cannot see a CUDA device\n' >&2
    exit 2
fi
freeze_device="$("$DEEPMD_PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.name};sm_{p.major}{p.minor};uuid_unavailable_in_torch")')"
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | head -n 1)"

STAGE_DIR="$(mktemp -d "${OUTPUT_DIR}/.freeze-two-phase.XXXXXX")"
cleanup() {
    rm -rf -- "$STAGE_DIR"
}
trap cleanup EXIT

EXPECTED_MAP="['Hf', 'Mo', 'Ta', 'Ti', 'Zr']"
printf 'model_id\tarchitecture\ttraining_parent\tprimary_phase\ttraining_run\ttraining_status_sha256\ttraining_input_sha256\tcheckpoint\tcheckpoint_sha256\tfrozen_model\tfrozen_sha256\ttype_map\tdp_version\tformat\tfreeze_device\tgpu_uuid\tcxx\tdp_compile_infer\tdp_tf32_infer\tdp_amp_infer\tdp_triton_infer\tartifact_validation\tcreated\n' > "${STAGE_DIR}/manifest.tsv"

for model_id in "${model_ids[@]}"; do
    architecture="${model_id%%__*}"
    parent="${model_id##*__}"
    parent="${parent%_parent}"
    checkpoint="$(checkpoint_for "$model_id")"
    run_directory="$(run_for "$model_id")"
    extension=pth
    [[ "$architecture" == DPA4 ]] && extension=pt2
    staged_model="${STAGE_DIR}/${model_id}.${extension}"
    freeze_target="$staged_model"
    model_device=portable_torchscript
    if [[ "$architecture" == DPA4 ]]; then
        freeze_target="${STAGE_DIR}/${model_id}"
        model_device="$freeze_device"
    fi

    if ! checkpoint_show="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$checkpoint" type-map 2>&1)"; then
        printf 'ERROR: dp show failed for %s checkpoint\n%s\n' "$model_id" "$checkpoint_show" >&2
        exit 3
    fi
    if [[ "$checkpoint_show" != *"$EXPECTED_MAP"* ]]; then
        printf 'ERROR: %s checkpoint type map is not %s\n%s\n' \
            "$model_id" "$EXPECTED_MAP" "$checkpoint_show" >&2
        exit 3
    fi

    printf 'Freezing %s from %s\n' "$model_id" "$checkpoint"
    "$DP_EXE" freeze -c "$checkpoint" -o "$freeze_target"
    if [[ ! -s "$staged_model" ]]; then
        printf 'ERROR: freeze did not produce expected artifact: %s\n' "$staged_model" >&2
        exit 3
    fi

    artifact_validation=dp_show_type_map
    if [[ "$architecture" == DPA4 ]]; then
        if ! unzip -tqq "$staged_model"; then
            printf 'ERROR: DPA4 .pt2 archive integrity check failed: %s\n' "$staged_model" >&2
            exit 3
        fi
        artifact_validation=checkpoint_type_map_plus_zip_integrity
    else
        if ! show_output="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$staged_model" type-map 2>&1)"; then
            printf 'ERROR: dp show failed for %s frozen model\n%s\n' "$model_id" "$show_output" >&2
            exit 3
        fi
        if [[ "$show_output" != *"$EXPECTED_MAP"* ]]; then
            printf 'ERROR: frozen %s type map is not %s\n%s\n' \
                "$model_id" "$EXPECTED_MAP" "$show_output" >&2
            exit 3
        fi
    fi

    output_model="${OUTPUT_DIR}/${model_id}.${extension}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model_id" "$architecture" "$parent" "$parent" \
        "$(basename "$run_directory")" \
        "$(sha256sum "${run_directory}/run_status.json" | awk '{print $1}')" \
        "$(sha256sum "${run_directory}/input.json" | awk '{print $1}')" \
        "$checkpoint" "${EXPECTED_HASHES[$model_id]}" \
        "$output_model" "$(sha256sum "$staged_model" | awk '{print $1}')" \
        'Hf,Mo,Ta,Ti,Zr' "$($DP_EXE --version 2>&1 | tail -n 1)" \
        "$extension" "$model_device" "$gpu_uuid" "$CXX" \
        "$DP_COMPILE_INFER" "$DP_TF32_INFER" "$DP_AMP_INFER" \
        "$DP_TRITON_INFER" "$artifact_validation" "$(date --iso-8601=seconds)" \
        >> "${STAGE_DIR}/manifest.tsv"
done

for model_id in "${model_ids[@]}"; do
    architecture="${model_id%%__*}"
    extension=pth
    [[ "$architecture" == DPA4 ]] && extension=pt2
    mv -- "${STAGE_DIR}/${model_id}.${extension}" "${OUTPUT_DIR}/${model_id}.${extension}"
done
mv -- "${STAGE_DIR}/manifest.tsv" "${OUTPUT_DIR}/manifest.tsv"
printf 'Frozen six-model two-phase set and immutable manifest in %s\n' "$OUTPUT_DIR"
