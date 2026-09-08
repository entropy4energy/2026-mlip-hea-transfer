#!/usr/bin/env bash
# Export only the preregistered final BCC-trained study checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=deepmd_env.sh
source "${SCRIPT_DIR}/deepmd_env.sh"

TRAINING_RUNS="${TRAINING_RUNS:-./training/runs}"
OUTPUT_REQUEST="${1:-${PROJECT_ROOT}/models}"
mkdir -p -- "$OUTPUT_REQUEST"
OUTPUT_DIR="$(cd -- "$OUTPUT_REQUEST" && pwd -P)"

declare -A CHECKPOINTS
CHECKPOINTS[DPA2]="${TRAINING_RUNS}/DPA2__bcc_to_fcc__pure_pocc_aimd/ckpt/model.ckpt-500000.pt"
CHECKPOINTS[DPA3]="${TRAINING_RUNS}/DPA3__bcc_to_fcc__pure_pocc_aimd/ckpt/model.ckpt-500000.pt"
CHECKPOINTS[DPA4]="${TRAINING_RUNS}/DPA4__bcc_to_fcc__pure_pocc_aimd/ckpt/model_ema.ckpt-500000.pt"
declare -A FILENAMES
FILENAMES[DPA2]=DPA2.pth
FILENAMES[DPA3]=DPA3.pth
FILENAMES[DPA4]=DPA4.pt2

for architecture in DPA2 DPA3 DPA4; do
    checkpoint="${CHECKPOINTS[$architecture]}"
    if [[ ! -r "$checkpoint" ]]; then
        printf 'NOT READY: preregistered %s checkpoint is absent: %s\n' "$architecture" "$checkpoint" >&2
        exit 4
    fi
    if [[ -e "${OUTPUT_DIR}/${FILENAMES[$architecture]}" ]]; then
        printf 'ERROR: refusing to overwrite frozen model: %s\n' \
            "${OUTPUT_DIR}/${FILENAMES[$architecture]}" >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT_DIR}/manifest.tsv" ]]; then
    printf 'ERROR: refusing to overwrite model manifest: %s\n' "${OUTPUT_DIR}/manifest.tsv" >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
    printf 'ERROR: DPA4 .pt2 must be frozen on its deployment GPU; set CUDA_VISIBLE_DEVICES\n' >&2
    exit 2
fi
cuda_available="$("$DEEPMD_PYTHON" -c 'import torch; print(int(torch.cuda.is_available()))')"
if [[ "$cuda_available" != 1 ]]; then
    printf 'ERROR: DPA4 .pt2 freeze cannot see a CUDA device\n' >&2
    exit 2
fi
DPA4_DEVICE="$("$DEEPMD_PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.name};sm_{p.major}{p.minor}")')"

STAGE_DIR="$(mktemp -d "${OUTPUT_DIR}/.freeze.XXXXXX")"
cleanup() {
    rm -rf -- "$STAGE_DIR"
}
trap cleanup EXIT

EXPECTED_MAP="['Hf', 'Mo', 'Ta', 'Ti', 'Zr']"
printf 'architecture\ttraining_run\tcheckpoint\tcheckpoint_sha256\tfrozen_model\tfrozen_sha256\ttype_map\tdp_version\tformat\tfreeze_device\tcxx\tdp_compile_infer\tdp_tf32_infer\tdp_amp_infer\tdp_triton_infer\tartifact_validation\n' > "${STAGE_DIR}/manifest.tsv"

for architecture in DPA2 DPA3 DPA4; do
    checkpoint="${CHECKPOINTS[$architecture]}"
    if ! checkpoint_show="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$checkpoint" type-map 2>&1)"; then
        printf 'ERROR: dp show failed for %s checkpoint\n%s\n' \
            "$architecture" "$checkpoint_show" >&2
        exit 3
    fi
    if [[ "$checkpoint_show" != *"$EXPECTED_MAP"* ]]; then
        printf 'ERROR: %s checkpoint type map is not %s\n%s\n' \
            "$architecture" "$EXPECTED_MAP" "$checkpoint_show" >&2
        exit 3
    fi
    frozen="${STAGE_DIR}/${FILENAMES[$architecture]}"
    freeze_target="$frozen"
    format=pth
    freeze_device=portable_torchscript
    if [[ "$architecture" == DPA4 ]]; then
        freeze_target="${STAGE_DIR}/DPA4"
        format=pt2
        freeze_device="$DPA4_DEVICE"
    fi
    printf 'Freezing %s from %s\n' "$architecture" "$checkpoint"
    "$DP_EXE" freeze -c "$checkpoint" -o "$freeze_target"
    if [[ ! -s "$frozen" ]]; then
        printf 'ERROR: freeze did not produce expected artifact: %s\n' "$frozen" >&2
        exit 3
    fi
    artifact_validation=dp_show_type_map
    if [[ "$architecture" == DPA4 ]]; then
        if ! unzip -tqq "$frozen"; then
            printf 'ERROR: DPA4 .pt2 archive integrity check failed: %s\n' "$frozen" >&2
            exit 3
        fi
        artifact_validation=checkpoint_type_map_plus_zip_integrity
    else
        if ! show_output="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$frozen" type-map 2>&1)"; then
            printf 'ERROR: dp show failed for %s\n%s\n' "$architecture" "$show_output" >&2
            exit 3
        fi
        if [[ "$show_output" != *"$EXPECTED_MAP"* ]]; then
            printf 'ERROR: %s type map is not %s\n%s\n' "$architecture" "$EXPECTED_MAP" "$show_output" >&2
            exit 3
        fi
    fi
    run_name="$(basename "$(dirname "$(dirname "$checkpoint")")")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$architecture" "$run_name" "$checkpoint" \
        "$(sha256sum "$checkpoint" | awk '{print $1}')" \
        "${OUTPUT_DIR}/${FILENAMES[$architecture]}" \
        "$(sha256sum "$frozen" | awk '{print $1}')" \
        'Hf,Mo,Ta,Ti,Zr' "$($DP_EXE --version 2>&1 | tail -n 1)" \
        "$format" "$freeze_device" "$CXX" "$DP_COMPILE_INFER" "$DP_TF32_INFER" \
        "$DP_AMP_INFER" "$DP_TRITON_INFER" "$artifact_validation" \
        >> "${STAGE_DIR}/manifest.tsv"
done

for architecture in DPA2 DPA3 DPA4; do
    mv -- "${STAGE_DIR}/${FILENAMES[$architecture]}" \
        "${OUTPUT_DIR}/${FILENAMES[$architecture]}"
done
mv -- "${STAGE_DIR}/manifest.tsv" "${OUTPUT_DIR}/manifest.tsv"
printf 'Frozen models and immutable hashes written to %s\n' "$OUTPUT_DIR"
