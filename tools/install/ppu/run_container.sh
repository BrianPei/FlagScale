#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
image=${1:?Usage: run_container.sh IMAGE COMMAND...}
shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
config="$root/.github/configs/ppu.yml"
yq_bin=${YQ_BIN:-yq}
options=$($yq_bin -r '.container_options' "$config")
# Options are a whitespace-delimited list, without shell evaluation.
read -r -a docker_args <<< "$options"
if [ "${PPU_WITH_DATA:-0}" = 1 ]; then
    while IFS= read -r volume; do
        test -d "${volume%%:*}" || { echo "Missing host data directory: $volume" >&2; exit 1; }
        docker_args+=(--volume "$volume")
    done < <("$yq_bin" -r '.container_volumes[]' "$config")
fi
name="flagscale-ppu-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run --rm --name "$name" "${docker_args[@]}" \
    --volume "$root:/workspace" --workdir /workspace \
    --user "$(id -u):$(id -g)" --env HOME=/tmp/ppu-home \
    --entrypoint /bin/bash "$image" -c '
        set -euo pipefail
        mkdir -p "$HOME"
        source /workspace/tools/install/ppu/env.sh
        export PYTHONPATH="/workspace:/workspace/flagscale/train:${PYTHONPATH:-}"
        exec "$@"
    ' bash "$@"
