#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"

checkout_source() {
    local repository=$1 revision=$2 target=$3
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || {
        echo "Expected resolved 40-character source SHA, got: $revision" >&2
        return 1
    }
    git init "$target"
    git -C "$target" remote add origin "$repository"
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach FETCH_HEAD
    test "$(git -C "$target" rev-parse HEAD)" = "$revision"
}

: "${FLAGSCALE_MEGATRON_REF:?Resolve Megatron-LM-FL to a commit SHA}"
: "${FLAGSCALE_TE_REF:?Resolve TransformerEngine-FL to a commit SHA}"
deps=/opt/flagscale/deps
mkdir -p "$deps"
checkout_source https://github.com/flagos-ai/Megatron-LM-FL.git \
    "$FLAGSCALE_MEGATRON_REF" "$deps/Megatron-LM-FL"
checkout_source https://github.com/flagos-ai/TransformerEngine-FL.git \
    "$FLAGSCALE_TE_REF" "$deps/TransformerEngine-FL"
git -C "$deps/TransformerEngine-FL" submodule update --init --recursive --depth 1
python -m pip install --no-build-isolation --no-deps "$deps/TransformerEngine-FL"
python -m pip install --no-build-isolation --no-deps "$deps/Megatron-LM-FL"
printf '%s\n' "$FLAGSCALE_TE_REF" > "$deps/.transformer-engine-fl.ref"
printf '%s\n' "$FLAGSCALE_MEGATRON_REF" > "$deps/.megatron-lm-fl.ref"
# The vendor torch exposes PPU through CUDA-compatible APIs and routes the
# PyTorch NCCL backend to PCCL. Never replace it with a public torch wheel.
python -c '
import torch
import torch.distributed as dist
import transformer_engine.pytorch
from megatron.core.models.gpt import GPTModel

assert dist.is_nccl_available()
print("Vendor NCCL/PCCL:", torch.cuda.nccl.version())
'
python -m pip check > /opt/flagscale/ppu/candidate-pip-check.txt 2>&1 || true
new_conflicts=$(comm -13 \
    <(LC_ALL=C sort /opt/flagscale/ppu/vendor-pip-check.txt) \
    <(LC_ALL=C sort /opt/flagscale/ppu/candidate-pip-check.txt))
if [ -n "$new_conflicts" ]; then
    echo "PPU overlay introduced dependency conflicts:" >&2
    printf '%s\n' "$new_conflicts" >&2
    exit 1
fi
python -m pip freeze > /opt/flagscale/ppu/packages.txt
