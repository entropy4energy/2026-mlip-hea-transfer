#!/usr/bin/env bash
# Run one hash-locked, restartable DeePMD calculation on its assigned L40S.

set -euo pipefail

RUN_DIR="${1:?usage: run_one.sh RUN_DIRECTORY}"
RUN_DIR="$(cd -- "$RUN_DIR" && pwd -P)"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# shellcheck source=/dev/null
source "$RUN_DIR/run.env"

if [[ "$(hostname -s)" != "saiph" ]]; then
  printf 'ERROR: %s is pinned to saiph\n' "$CALCULATION_ID" >&2
  exit 2
fi
if [[ "$RUN_STATE" != "READY" ]]; then
  printf 'BLOCKED: %s is %s: %s\n' "$CALCULATION_ID" "$RUN_STATE" "$BLOCK_REASON" >&2
  exit 4
fi

cd "$RUN_DIR"
mkdir -p logs ckpt
exec 9> .training.lock
if ! flock -n 9; then
  printf 'ERROR: another launcher holds %s/.training.lock\n' "$RUN_DIR" >&2
  exit 3
fi

actual_hash="$(sha256sum input.json | awk '{print $1}')"
if [[ "$actual_hash" != "$INPUT_SHA256" ]]; then
  printf 'ERROR: input hash changed for %s\n' "$CALCULATION_ID" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
    | grep -Fqx "$GPU_UUID"; then
  printf 'ERROR: GPU %s assigned to %s is occupied\n' "$GPU_INDEX" "$CALCULATION_ID" >&2
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

if [[ ! -x "$DEEPMD_ENV_PREFIX/bin/python" || ! -r "$DEEPMD_RUNTIME_LIBSTDCXX" ]]; then
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
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_WORKER_START_METHOD=spawn
export PYTHONUNBUFFERED=1

{
  printf 'started=%s\n' "$(date --iso-8601=seconds)"
  printf 'calculation=%s\n' "$CALCULATION_ID"
  printf 'input_sha256=%s\n' "$INPUT_SHA256"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  "$DEEPMD_ENV_PREFIX/bin/python" -c \
    'import deepmd, torch; print("deepmd=" + deepmd.__version__); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
} > environment.txt

printf '\n[%s] Starting %s on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$CALCULATION_ID" "$GPU_INDEX"
"$DEEPMD_ENV_PREFIX/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print("visible GPU:", torch.cuda.get_device_name(0))'

exec numactl --membind="$NUMA_NODE" taskset -c "$CPUSET" \
  "$DEEPMD_ENV_PREFIX/bin/python" "$ROOT/runDP_restartable.py" \
    --input input.json --logdir logs --restart auto
