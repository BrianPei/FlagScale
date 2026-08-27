#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-1}"

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
    --volume /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
    --privileged
)

if [ "$task" = train ]; then
    docker run "${docker_args[@]}" \
        --env EXPECTED_DEVICE_COUNT="$expected_devices" \
        --env TE_FL_SKIP_CUDA=1 \
        --entrypoint python \
        "$candidate" -c '
import importlib.metadata as metadata
import pathlib
import os

import torch
import torch_npu
import transformer_engine
import megatron.core
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from transformer_engine.pytorch import Linear

expected_devices = int(os.environ["EXPECTED_DEVICE_COUNT"])
assert torch.npu.is_available()
assert torch.npu.device_count() >= expected_devices
assert HAVE_TE
assert TESpecProvider is not None

value = torch.ones(16, device="npu:0")
assert value.sum().item() == 16

distribution = metadata.distribution("transformer-engine")
native = sorted(
    str(pathlib.Path(distribution.locate_file(path)))
    for path in distribution.files or ()
    if str(path).endswith(".so")
)
print("Ascend train runtime:", torch.__version__)
print("Megatron:", metadata.version("megatron-core"), megatron.core.__file__)
print("TransformerEngine:", distribution.version, transformer_engine.__file__)
print("TransformerEngine native SO:", native or "none (Python vendor backend)")
print("TE Linear:", Linear)
'
    exit 0
fi

[ "$task" = inference ] || {
    echo "Unsupported Ascend image task: $task" >&2
    exit 1
}

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
