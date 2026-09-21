#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/retry_utils.sh"
source "$SCRIPT_DIR/env.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
deps="${FLAGSCALE_DEPS:-${FLAGSCALE_HOME:-/opt/flagscale}/deps}"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

: "${FLAGSCALE_MEGATRON_REF:?Resolve Megatron-LM-FL to a commit SHA}"
: "${FLAGSCALE_TE_REF:?Resolve TransformerEngine-FL to a commit SHA}"
for revision in "$FLAGSCALE_MEGATRON_REF" "$FLAGSCALE_TE_REF"; do
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || die "Expected a resolved source SHA: $revision"
done

# Pin the installed vendor runtime while resolving the training requirements.
# This phase must also work when the common installer skips the base phase.
if [ "$DEBUG" != true ]; then
    python -c 'import torch; print("Vendor torch:", torch.__version__, torch.__file__)'
    constraints=$(mktemp)
    trap 'rm -f "$constraints"' EXIT
    python - <<'PY' > "$constraints"
import importlib.metadata as md
for dist in md.distributions():
    name = dist.metadata.get("Name", "")
    if name.lower().replace("_", "-").startswith(("torch", "triton", "flagcx")):
        print(f"{name}=={dist.version}")
PY
    export PIP_CONSTRAINT="$constraints${PIP_CONSTRAINT:+ $PIP_CONSTRAINT}"
fi
set_step "Installing PPU train requirements"
retry_pip_install -d "$DEBUG" "$PROJECT_ROOT/requirements/ppu/train.txt" "$RETRY_COUNT"
run_cmd -d "$DEBUG" mkdir -p "$deps"
retry_git_checkout_ref -d "$DEBUG" https://github.com/flagos-ai/Megatron-LM-FL.git \
    "$FLAGSCALE_MEGATRON_REF" "$deps/Megatron-LM-FL" "$RETRY_COUNT"
retry_git_checkout_ref -d "$DEBUG" --recursive \
    https://github.com/flagos-ai/TransformerEngine-FL.git \
    "$FLAGSCALE_TE_REF" "$deps/TransformerEngine-FL" "$RETRY_COUNT"
pip_cmd=$(get_pip_cmd)
for package in TransformerEngine-FL Megatron-LM-FL; do
    retry -d "$DEBUG" "$RETRY_COUNT" \
        "$pip_cmd install --no-build-isolation --no-deps '$deps/$package'"
done
[ "$DEBUG" = true ] && exit 0
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
log_success "PPU training runtime ready"
