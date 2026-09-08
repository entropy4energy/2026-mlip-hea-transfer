#!/usr/bin/env bash
# Freeze one explicit checkpoint without guessing which checkpoint is final.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=deepmd_env.sh
source "${SCRIPT_DIR}/deepmd_env.sh"

if (( $# != 3 )); then
    printf 'Usage: %s DPA2|DPA3|DPA4 CHECKPOINT OUTPUT.pth|OUTPUT.pt2\n' "$0" >&2
    exit 2
fi
ARCHITECTURE="$1"
CHECKPOINT="$(realpath -e -- "$2")"
OUTPUT_REQUEST="$3"
case "$ARCHITECTURE" in DPA2|DPA3|DPA4) ;; *) printf 'ERROR: invalid architecture\n' >&2; exit 2 ;; esac
case "$ARCHITECTURE" in
    DPA4)
        expected_extension=pt2
        ;;
    *)
        expected_extension=pth
        ;;
esac
if [[ "${OUTPUT_REQUEST##*.}" != "$expected_extension" ]]; then
    printf 'ERROR: %s frozen model output must end in .%s\n' \
        "$ARCHITECTURE" "$expected_extension" >&2
    exit 2
fi
if [[ -e "$OUTPUT_REQUEST" || -e "${OUTPUT_REQUEST}.provenance.tsv" ]]; then
    printf 'ERROR: refusing to overwrite existing model or provenance sidecar\n' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$OUTPUT_REQUEST")"
OUTPUT_DIR="$(cd -- "$(dirname -- "$OUTPUT_REQUEST")" && pwd -P)"
OUTPUT="${OUTPUT_DIR}/$(basename -- "$OUTPUT_REQUEST")"
STAGE_DIR="$(mktemp -d "${OUTPUT_DIR}/.freeze-one.XXXXXX")"
cleanup() { rm -rf -- "$STAGE_DIR"; }
trap cleanup EXIT

expected="['Hf', 'Mo', 'Ta', 'Ti', 'Zr']"
if ! checkpoint_show="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$CHECKPOINT" type-map 2>&1)"; then
    printf 'ERROR: dp show failed for source checkpoint %s\n%s\n' \
        "$CHECKPOINT" "$checkpoint_show" >&2
    exit 3
fi
if [[ "$checkpoint_show" != *"$expected"* ]]; then
    printf 'ERROR: checkpoint type map is not %s\n%s\n' "$expected" "$checkpoint_show" >&2
    exit 3
fi

freeze_device=portable_torchscript
if [[ "$ARCHITECTURE" == DPA4 ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
        printf 'ERROR: DPA4 .pt2 must be frozen on its deployment GPU; set CUDA_VISIBLE_DEVICES\n' >&2
        exit 2
    fi
    cuda_available="$("$DEEPMD_PYTHON" -c 'import torch; print(int(torch.cuda.is_available()))')"
    if [[ "$cuda_available" != 1 ]]; then
        printf 'ERROR: DPA4 .pt2 freeze cannot see a CUDA device\n' >&2
        exit 2
    fi
    freeze_device="$("$DEEPMD_PYTHON" -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.name};sm_{p.major}{p.minor}")')"
    FREEZE_TARGET="${STAGE_DIR}/model"
    STAGED="${FREEZE_TARGET}.pt2"
else
    STAGED="${STAGE_DIR}/model.pth"
    FREEZE_TARGET="$STAGED"
fi

"$DP_EXE" freeze -c "$CHECKPOINT" -o "$FREEZE_TARGET"
if [[ ! -s "$STAGED" ]]; then
    printf 'ERROR: freeze did not produce expected artifact: %s\n' "$STAGED" >&2
    exit 3
fi
artifact_validation=dp_show_type_map
if [[ "$ARCHITECTURE" == DPA4 ]]; then
    # `dp show` in this pinned development build does not decode binary AOT
    # extras in .pt2. Validate its source map above and its archive integrity.
    if ! unzip -tqq "$STAGED"; then
        printf 'ERROR: DPA4 .pt2 archive integrity check failed: %s\n' "$STAGED" >&2
        exit 3
    fi
    artifact_validation=checkpoint_type_map_plus_zip_integrity
else
    if ! show_output="$(OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 "$DP_EXE" show "$STAGED" type-map 2>&1)"; then
        printf 'ERROR: dp show failed for frozen model %s\n%s\n' "$STAGED" "$show_output" >&2
        exit 3
    fi
    if [[ "$show_output" != *"$expected"* ]]; then
        printf 'ERROR: frozen model type map is not %s\n%s\n' "$expected" "$show_output" >&2
        exit 3
    fi
fi

{
    printf 'key\tvalue\n'
    printf 'architecture\t%s\n' "$ARCHITECTURE"
    printf 'checkpoint\t%s\n' "$CHECKPOINT"
    printf 'checkpoint_sha256\t%s\n' "$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    printf 'frozen_model\t%s\n' "$OUTPUT"
    printf 'frozen_model_sha256\t%s\n' "$(sha256sum "$STAGED" | awk '{print $1}')"
    printf 'type_map\tHf,Mo,Ta,Ti,Zr\n'
    printf 'format\t%s\n' "$expected_extension"
    printf 'freeze_device\t%s\n' "$freeze_device"
    printf 'cxx\t%s\n' "$CXX"
    printf 'dp_compile_infer\t%s\n' "$DP_COMPILE_INFER"
    printf 'dp_tf32_infer\t%s\n' "$DP_TF32_INFER"
    printf 'dp_amp_infer\t%s\n' "$DP_AMP_INFER"
    printf 'dp_triton_infer\t%s\n' "$DP_TRITON_INFER"
    printf 'artifact_validation\t%s\n' "$artifact_validation"
    printf 'dp_version\t%s\n' "$($DP_EXE --version 2>&1 | tail -n 1)"
    printf 'created\t%s\n' "$(date --iso-8601=seconds)"
} > "${STAGE_DIR}/provenance.tsv"

mv -- "$STAGED" "$OUTPUT"
mv -- "${STAGE_DIR}/provenance.tsv" "${OUTPUT}.provenance.tsv"
printf 'Frozen %s model: %s\n' "$ARCHITECTURE" "$OUTPUT"
