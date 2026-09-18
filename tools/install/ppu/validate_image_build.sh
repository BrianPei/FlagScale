#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$script_dir/../../.." && pwd)
phase=${IMAGE_BUILD_PHASE:?Expected pre or post}
test "${IMAGE_BUILD_TASK:?Expected train}" = train
case "$phase" in
    pre) image=${IMAGE_BUILD_BASE_IMAGE:?}; mode=device; docker pull "$image" ;;
    post) image=${IMAGE_BUILD_CANDIDATE_IMAGE:?}; mode=train ;;
    *) echo "Unknown validation phase: $phase" >&2; exit 2 ;;
esac
mkdir -p "$root/ppu-artifacts"
docker image inspect "$image" > "$root/ppu-artifacts/$phase-image.json"
bash "$script_dir/run_container.sh" "$image" \
    python tools/install/ppu/probe.py inventory | tee "$root/ppu-artifacts/$phase-inventory.json"
# The training topology is the single source for smoke-test process count.
nproc=$(${YQ_BIN:-yq} -r '.experiment.runner.nproc_per_node' \
    "$root/tests/functional_tests/train/qwen3/conf/0_6b_ppu.yaml")
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] && [ "$nproc" -ge 2 ]
timeout 300 bash "$script_dir/run_container.sh" "$image" \
    python -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
    tools/install/ppu/probe.py "$mode" | tee "$root/ppu-artifacts/$phase-smoke.log"
