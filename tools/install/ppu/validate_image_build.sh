#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
phase=${IMAGE_BUILD_PHASE:?Expected pre or post}
test "${IMAGE_BUILD_TASK:?Expected train}" = train
nproc=${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:?IMAGE_BUILD_RUNTIME_SMOKE_NPROC is required}
device_count=${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:?IMAGE_BUILD_RUNTIME_DEVICE_COUNT is required}
if ! [[ "$nproc" =~ ^[1-9][0-9]*$ ]] || [ "$nproc" -lt 2 ]; then
    echo "Expected a smoke process count of at least two: $nproc" >&2
    exit 2
fi
if ! [[ "$device_count" =~ ^[1-9][0-9]*$ ]] || [ "$device_count" -lt "$nproc" ]; then
    echo "Expected device count >= smoke process count: $device_count" >&2
    exit 2
fi
case "$phase" in
    pre) image=${IMAGE_BUILD_BASE_IMAGE:?}; mode=device; docker pull "$image" ;;
    post) image=${IMAGE_BUILD_CANDIDATE_IMAGE:?}; mode=train ;;
    *) echo "Unknown validation phase: $phase" >&2; exit 2 ;;
esac
timeout 300 bash "$script_dir/run_container.sh" "$image" \
    env EXPECTED_DEVICE_COUNT="$device_count" \
    python -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
    tools/install/ppu/probe.py "$mode"
