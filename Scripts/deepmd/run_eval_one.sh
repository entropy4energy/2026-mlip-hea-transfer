#!/usr/bin/env bash
# Run one hash-locked DeePMD evaluation on its preregistered Saiph GPU.

set -euo pipefail

EVALUATION_ID="${1:?usage: run_eval_one.sh EVALUATION_ID}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST="$ROOT/evaluation_manifest.csv"

found=false
while IFS=, read -r evaluation_id run_id architecture training_fold regime \
    checkpoint_variant checkpoint_path evaluation_variant system_set_id \
    system_map systems_file system_count frame_count queue gpu_index gpu_uuid \
    cpuset numa_node training_input_sha256 output_dir; do
  if [[ "$evaluation_id" == "$EVALUATION_ID" ]]; then
    RUN_ID="$run_id"
    GPU_INDEX="$gpu_index"
    GPU_UUID="$gpu_uuid"
    CPUSET="$cpuset"
    NUMA_NODE="$numa_node"
    # The frozen CSV uses CRLF line endings; remove the record terminator from
    # its final field before using it as a shell path.
    OUTPUT_DIR="${output_dir%$'\r'}"
    found=true
    break
  fi
done < <(sed '1d' "$MANIFEST")
if [[ "$found" != true ]]; then
  printf 'ERROR: evaluation ID is absent from manifest: %s\n' "$EVALUATION_ID" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "saiph" ]]; then
  printf 'ERROR: %s is pinned to saiph\n' "$EVALUATION_ID" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec 9> "$OUTPUT_DIR/.evaluation.lock"
if ! flock -n 9; then
  printf 'ERROR: another launcher holds %s/.evaluation.lock\n' "$OUTPUT_DIR" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | awk -v uuid="$GPU_UUID" '{gsub(/[[:space:]]/, ""); if ($0 == uuid) found=1} END {exit !found}'; then
  printf 'ERROR: GPU %s assigned to %s is occupied\n' "$GPU_INDEX" "$EVALUATION_ID" >&2
  exit 5
fi

export DEEPMD_ROOT="${DEEPMD_BUILD_ROOT:?set to your DeePMD-kit build}"
export DEEPMD_ENV_PREFIX="${DEEPMD_CONDA_ENV:?set to your DeePMD conda environment}"
export CUDA_HOME="${CUDA_INSTALL_DIR:?set to your CUDA 12.9 installation}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export NVHPC_ROOT="${NVHPC_INSTALL_DIR:?set to your NVIDIA HPC SDK installation}"
export DEEPMD_AOT_GCC_ROOT="/data/apps/extern/easybuild/GCCcore/12.3.0"
export DEEPMD_RUNTIME_LIBSTDCXX="$DEEPMD_ENV_PREFIX/lib/libstdc++.so.6"
export LAMMPS_PLUGIN_PATH="$DEEPMD_ROOT/lib"
export DP_BACKEND="pytorch"

if [[ ! -x "$DEEPMD_ENV_PREFIX/bin/python" || ! -x "$DEEPMD_ENV_PREFIX/bin/dp" || \
      ! -r "$DEEPMD_RUNTIME_LIBSTDCXX" ]]; then
  printf 'ERROR: validated Build_02 runtime is unavailable\n' >&2
  exit 2
fi
export PATH="$DEEPMD_ENV_PREFIX/bin:$CUDA_HOME/bin:$DEEPMD_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$DEEPMD_ENV_PREFIX/lib:$CUDA_HOME/lib64:$DEEPMD_ROOT/lib:${LD_LIBRARY_PATH:-}"
case ":${LD_PRELOAD:-}:" in
  *":$DEEPMD_RUNTIME_LIBSTDCXX:"*) ;;
  *) export LD_PRELOAD="$DEEPMD_RUNTIME_LIBSTDCXX${LD_PRELOAD:+:$LD_PRELOAD}" ;;
esac
export CC="$DEEPMD_AOT_GCC_ROOT/bin/gcc"
export CXX="$DEEPMD_AOT_GCC_ROOT/bin/g++"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=1
export TORCH_NUM_THREADS=4
export TORCH_NUM_INTEROP_THREADS=1
export DP_PT_NUM_WORKERS=0
export NUM_WORKERS=0
export PYTHONUNBUFFERED=1

{
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  printf 'evaluation=%s\n' "$EVALUATION_ID"
  printf 'training_run=%s\n' "$RUN_ID"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  "$DEEPMD_ENV_PREFIX/bin/python" -c \
    'import deepmd, torch; print("deepmd=" + deepmd.__version__); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
} > "$OUTPUT_DIR/environment.txt"

printf '\n[%s] Starting evaluation %s on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$EVALUATION_ID" "$GPU_INDEX"
exec numactl --membind="$NUMA_NODE" taskset -c "$CPUSET" \
  "$DEEPMD_ENV_PREFIX/bin/python" "$ROOT/evaluate_study_models.py" run \
    --evaluation-id "$EVALUATION_ID" \
    --dp-executable "$DEEPMD_ENV_PREFIX/bin/dp"
