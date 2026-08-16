#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-2}"
smoke_nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-$expected_devices}"

if [ "$phase" != post ]; then
    exit 0
fi

docker_args=(
    --rm
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    --volume /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro
    --volume /usr/local/Ascend/nnal:/usr/local/Ascend/nnal:ro
    --volume /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
    --privileged
)

if [ "$task" = all ]; then
    docker run "${docker_args[@]}" \
        --env EXPECTED_DEVICE_COUNT="$expected_devices" \
        --env EXPECTED_WORLD_SIZE="$smoke_nproc" \
        --entrypoint bash "$candidate" -lc '
set -euo pipefail
export FLAGSCALE_RUNTIME_ROOT=/opt/flagscale/runtimes/train
export HCCL_NPU_SOCKET_PORT_RANGE=41000-41099
. "$FLAGSCALE_RUNTIME_ROOT/activate.sh"
python - <<"PY"
import os
import torch
import torch_npu
import transformer_engine
from megatron.core.models.gpt import GPTModel

assert torch.npu.device_count() >= int(os.environ["EXPECTED_DEVICE_COUNT"])
value = torch.ones(16, device="npu:0")
assert value.sum().item() == 16
print("train runtime:", torch.__version__, transformer_engine.__file__, GPTModel)
PY
cat >/tmp/collective.py <<"PY"
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
value = torch.tensor([rank + 1.0], device=f"npu:{rank}")
dist.all_reduce(value)
assert value.item() == world * (world + 1) / 2, value
dist.destroy_process_group()
PY
torchrun --nnodes=1 --nproc-per-node="${EXPECTED_WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port=29500 /tmp/collective.py
'
    docker run "${docker_args[@]}" \
        --env EXPECTED_DEVICE_COUNT="$expected_devices" \
        --env EXPECTED_WORLD_SIZE="$smoke_nproc" \
        --entrypoint bash "$candidate" -lc '
set -euo pipefail
export FLAGSCALE_RUNTIME_ROOT=/opt/flagscale/runtimes/inference
export HCCL_NPU_SOCKET_PORT_RANGE=41100-41199
. "$FLAGSCALE_RUNTIME_ROOT/activate.sh"
python - <<"PY"
import os
import torch
import torch_npu
import vllm_fl
from vllm.platforms import current_platform

assert torch.npu.device_count() >= int(os.environ["EXPECTED_DEVICE_COUNT"])
assert type(current_platform).__module__ == "vllm_fl.platform"
assert current_platform.device_type == "npu"
assert current_platform.dist_backend == "hccl"
value = torch.ones(16, device="npu:0")
assert value.sum().item() == 16
print("inference runtime:", torch.__version__, type(current_platform))
PY
cat >/tmp/collective.py <<"PY"
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank)
dist.init_process_group("hccl")
value = torch.tensor([rank + 1.0], device=f"npu:{rank}")
dist.all_reduce(value)
assert value.item() == world * (world + 1) / 2, value
dist.destroy_process_group()
PY
torchrun --nnodes=1 --nproc-per-node="${EXPECTED_WORLD_SIZE}" \
    --master-addr=127.0.0.1 --master-port=29500 /tmp/collective.py
'
    exit 0
fi

[ "$task" = inference ] || exit 0

docker run "${docker_args[@]}" \
    --entrypoint python \
    "$candidate" -c '
import flag_gems
import vllm
import vllm_fl
from vllm.platforms import current_platform

print("platform:", type(current_platform).__module__, type(current_platform).__name__)
print("device_type:", current_platform.device_type)
print("dist_backend:", current_platform.dist_backend)
assert type(current_platform).__module__ == "vllm_fl.platform"
assert type(current_platform).__name__ == "PlatformFL"
assert current_platform.device_type == "npu"
assert current_platform.dist_backend == "hccl"
'
