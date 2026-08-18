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
    local runtime_phase="$4"

    docker run --rm \
        --privileged \
        --ipc=host \
        --shm-size=64g \
        --env EXPECTED_WORLD_SIZE="$expected_world_size" \
        --env FLAGSCALE_RUNTIME_TASK="$runtime_task" \
        --env FLAGSCALE_RUNTIME_PHASE="$runtime_phase" \
        --entrypoint bash "$image" -lc '
set -euo pipefail
export FLAGSCALE_CONDA="${FLAGSCALE_CONDA:-/root/miniconda}"
export FLAGSCALE_ENV_NAME="${FLAGSCALE_ENV_NAME:-python310_torch29_cuda}"
if [ -f "$FLAGSCALE_CONDA/etc/profile.d/conda.sh" ]; then
    . "$FLAGSCALE_CONDA/etc/profile.d/conda.sh"
    conda activate "$FLAGSCALE_ENV_NAME"
fi
if [ -f /etc/profile.d/flagscale-env.sh ]; then
    . /etc/profile.d/flagscale-env.sh
else
    export PYTHONPATH="/opt/Megatron-LM-FL:${PYTHONPATH:-}"
fi
python - "$FLAGSCALE_RUNTIME_TASK" "$FLAGSCALE_RUNTIME_PHASE" <<"PY"
import os
import sys
from pathlib import Path

import torch

task = sys.argv[1]
phase = sys.argv[2]
expected_world_size = int(os.environ["EXPECTED_WORLD_SIZE"])

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= expected_world_size
assert torch.tensor([1.0], device="cuda").item() == 1.0
if phase == "pre":
    print("Kunlunxin base runtime:", torch.__version__, torch.cuda.device_count())
    raise SystemExit(0)

if task == "train":
    import flagcx
    import megatron.core
    import transformer_engine
    import transformer_engine_torch

    assert flagcx is not None
    assert transformer_engine is not None
    assert transformer_engine_torch is not None
    expected = Path(os.environ.get("FLAGSCALE_MEGATRON_PATH", "/opt/flagscale/deps/Megatron-LM-FL")).resolve()
    actual = Path(megatron.core.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
    print("Kunlunxin train runtime:", torch.__version__, megatron.core.__file__)
elif task == "inference":
    import sentencepiece
    import tiktoken
    import transformers

    assert sentencepiece is not None
    assert tiktoken is not None
    assert transformers is not None
    print("Kunlunxin inference runtime:", torch.__version__, transformers.__version__)

    # Verify the triton.autotune compat shim lets flag_gems import on P800 and
    # that vLLM loads the fl plugin onto the Kunlunxin platform. Without the
    # shim, `import flag_gems` raises TypeError (generate_configs) and vLLM
    # falls back to UnspecifiedPlatform ("Device string must not be empty").
    import flag_gems
    os.environ.setdefault("VLLM_PLUGINS", "fl")
    os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
    os.environ.setdefault("USE_FLAGGEMS", "false")
    from vllm.platforms import current_platform
    platform_module = type(current_platform).__module__
    platform_class = type(current_platform).__name__
    print("flag_gems:", getattr(flag_gems, "__file__", "built-in"))
    print("platform_module:", platform_module, "platform_class:", platform_class)
    assert "vllm_fl" in platform_module, (
        f"vLLM did not load fl plugin; platform={platform_module}.{platform_class}"
    )
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
    expected = Path(os.environ.get("FLAGSCALE_MEGATRON_PATH", "/opt/flagscale/deps/Megatron-LM-FL")).resolve()
    actual = Path(megatron.core.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)

    # Same flag_gems / vLLM platform probe as the inference task. Catches the
    # triton.autotune(generate_configs) import failure at build time instead of
    # at the inference functional test.
    import flag_gems
    os.environ.setdefault("VLLM_PLUGINS", "fl")
    os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
    os.environ.setdefault("USE_FLAGGEMS", "false")
    from vllm.platforms import current_platform
    platform_module = type(current_platform).__module__
    platform_class = type(current_platform).__name__
    print("flag_gems:", getattr(flag_gems, "__file__", "built-in"))
    print("platform_module:", platform_module, "platform_class:", platform_class)
    assert "vllm_fl" in platform_module, (
        f"vLLM did not load fl plugin; platform={platform_module}.{platform_class}"
    )
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
        validate_cuda_runtime "$base_image" "$task" "$expected_devices" pre
        ;;
    post)
        validate_cuda_runtime "$candidate" "$task" "$smoke_nproc" post
        ;;
    *)
        exit 0
        ;;
esac
