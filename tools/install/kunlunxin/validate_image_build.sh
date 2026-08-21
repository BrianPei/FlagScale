#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# Mirrors tools/install/ascend/validate_image_build.sh: only the post phase
# is validated, and only the inference task. The all/train images are
# exercised by their own functional tests, NOT by importing flag_gems here --
# a previous version imported flag_gems in the all branch, which crashed at
# fused/geglu.py:12 (pow = tl_extra_shim.pow) because GEMS_VENDOR was unset,
# so flag_gems' DeviceDetector singleton picked the wrong vendor and
# tl_extra_shim fell back to triton.language.math (no pow on Triton 3.0.0).
# The real vllm serve gets GEMS_VENDOR=kunlunxin from the case-yaml env, so
# it never hit this; the crash was a validate-script artefact. Like ascend,
# we skip all/train and keep the inference check minimal.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"

if [ "$phase" != post ]; then
    exit 0
fi
if [ "$task" != inference ]; then
    exit 0
fi

# GEMS_VENDOR must be in the environment before the first `import flag_gems`
# below: flag_gems' DeviceDetector (runtime/backend/device.py) is a singleton
# that reads it once to pick the vendor backend, and kunlunxin is NOT in its
# quick-cmd probe dict (ascend IS, which is why ascend's validate sets no env
# here). Passing it via docker --env keeps it out of the heredoc body (single
# quotes inside the -lc '...' wrapper get stripped by bash quote-removal).
# VLLM_PLUGINS/VLLM_FL_PLATFORM mirror the case-yaml serve env so this check
# replicates the real import path the functional test exercises.
docker run --rm \
    --privileged \
    --ipc=host \
    --shm-size=64g \
    --env GEMS_VENDOR=kunlunxin \
    --env VLLM_PLUGINS=fl \
    --env VLLM_FL_PLATFORM=kunlunxin \
    --entrypoint bash "$candidate" -lc '
set -euo pipefail
export FLAGSCALE_CONDA="${FLAGSCALE_CONDA:-/root/miniconda}"
export FLAGSCALE_ENV_NAME="${FLAGSCALE_ENV_NAME:-python310_torch29_cuda}"
if [ -f "$FLAGSCALE_CONDA/etc/profile.d/conda.sh" ]; then
    . "$FLAGSCALE_CONDA/etc/profile.d/conda.sh"
    conda activate "$FLAGSCALE_ENV_NAME"
fi
python - <<"PY"
import flag_gems
import vllm
import vllm_fl
from vllm.platforms import current_platform

print("platform:", type(current_platform).__module__, type(current_platform).__name__)
print("device_type:", current_platform.device_type)
print("dist_backend:", current_platform.dist_backend)
assert "vllm_fl" in type(current_platform).__module__, type(current_platform).__module__
assert type(current_platform).__name__ == "PlatformFL", type(current_platform).__name__
PY
'
