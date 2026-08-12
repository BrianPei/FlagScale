#!/usr/bin/env bash

set -euo pipefail

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CUDA_PATH="${CUDA_PATH:-/opt/dtk/cuda/cuda-12}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH}}"
export DEVICE_HOME="${DEVICE_HOME:-${CUDA_PATH}}"
export CCL_HOME="${CCL_HOME:-${CUDA_PATH}}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"

flagcx_source="${FLAGSCALE_DEPS:-/opt/flagscale/deps}/FlagCX"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("DTK devices are unavailable; mount /opt/hyhal and /dev/kfd,/dev/dri")
print(f"DTK devices available: {torch.cuda.device_count()}")
PY

if python -c "from flagcx import _C" >/dev/null 2>&1; then
    echo "FlagCX extension is already installed"
    exit 0
fi

rm -rf "${flagcx_source}/build"
python -m pip install \
    --force-reinstall \
    --no-deps \
    --no-build-isolation \
    -v "${flagcx_source}"

python - <<'PY'
import torch
import flagcx
from flagcx import _C

print(f"FlagCX ready on {torch.cuda.device_count()} DTK devices")
PY
