#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
base_image="${IMAGE_BUILD_BASE_IMAGE:?IMAGE_BUILD_BASE_IMAGE is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-8}"

[ "$task" = inference ] || exit 0

case "$phase" in
    pre) image="$base_image" ;;
    post) image="$candidate" ;;
    *) exit 0 ;;
esac

[ "$phase" != pre ] || docker pull "$image"

docker run --rm \
    --privileged \
    --ipc=host \
    --network host \
    --volume /dev:/dev \
    --volume /sys:/sys \
    --env EXPECTED_DEVICE_COUNT="$expected_devices" \
    --env VLLM_PLUGINS=fl \
    --env VLLM_FL_PLATFORM=enflame \
    --entrypoint python \
    "$image" -c '
import importlib.metadata as metadata
import os
import torch

device_count = torch.gcu.device_count()
platform_plugins = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
general_plugins = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.general_plugins")
}

print("torch:", torch.__version__)
print("devices:", device_count)
print("vllm:", metadata.version("vllm"))
print("vllm-plugin-fl:", metadata.version("vllm-plugin-fl"))
print("platform plugins:", platform_plugins)
print("general plugins:", general_plugins)

assert device_count >= int(os.environ["EXPECTED_DEVICE_COUNT"])
assert metadata.version("vllm").startswith("0.20.2")
assert platform_plugins.get("fl") == "vllm_fl:register"
assert general_plugins.get("fl") == "vllm_fl:register_model"

value = torch.ones(16, device="gcu:0")
assert value.sum().item() == 16
'
