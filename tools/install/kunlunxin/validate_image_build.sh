#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
base_image="${IMAGE_BUILD_BASE_IMAGE:?IMAGE_BUILD_BASE_IMAGE is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-8}"
smoke_nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-2}"

case "$task" in
    train|inference|all) ;;
    *) exit 0 ;;
esac

validate_cuda_runtime() {
    local image="$1"
    local runtime_task="$2"
    local expected_world_size="$3"

    docker run --rm \
        --privileged \
        --ipc=host \
        --shm-size=64g \
        --env EXPECTED_WORLD_SIZE="$expected_world_size" \
        --env FLAGSCALE_RUNTIME_TASK="$runtime_task" \
        --entrypoint bash "$image" -lc '
set -euo pipefail
python - "$FLAGSCALE_RUNTIME_TASK" <<"PY"
import os
import sys

import torch

task = sys.argv[1]
expected_world_size = int(os.environ["EXPECTED_WORLD_SIZE"])

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= expected_world_size
assert torch.tensor([1.0], device="cuda").item() == 1.0

if task == "train":
    import flagcx
    import megatron.core
    import transformer_engine
    import transformer_engine_torch
    from megatron.core.jit import disable_jit_fuser
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

    assert callable(disable_jit_fuser)
    assert callable(get_gpt_decoder_layer_specs)
    assert flagcx is not None
    assert transformer_engine is not None
    assert transformer_engine_torch is not None
    print("Kunlunxin train runtime:", torch.__version__, megatron.core.__file__)
elif task == "inference":
    import sentencepiece
    import tiktoken
    import transformers

    assert sentencepiece is not None
    assert tiktoken is not None
    assert transformers is not None
    print("Kunlunxin inference runtime:", torch.__version__, transformers.__version__)
elif task == "all":
    import flagcx
    import megatron.core
    import sentencepiece
    import tiktoken
    import transformers
    import transformer_engine
    import transformer_engine_torch

    assert flagcx is not None
    assert megatron.core is not None
    assert sentencepiece is not None
    assert tiktoken is not None
    assert transformers is not None
    assert transformer_engine is not None
    assert transformer_engine_torch is not None
    print("Kunlunxin all runtime:", torch.__version__, megatron.core.__file__)
else:
    raise SystemExit(f"Unsupported Kunlunxin runtime task: {task}")
PY
'
}

case "$phase" in
    pre)
        if [[ "$base_image" == */* ]]; then
            docker pull "$base_image"
        elif ! docker image inspect "$base_image" >/dev/null 2>&1; then
            echo "Kunlunxin base image is not available locally: $base_image" >&2
            echo "Use a registry-qualified base image or load the vendor image on the P800 runner." >&2
            exit 1
        fi
        validate_cuda_runtime "$base_image" "$task" "$expected_devices"
        ;;
    post)
        validate_cuda_runtime "$candidate" "$task" "$smoke_nproc"
        ;;
    *)
        exit 0
        ;;
esac
