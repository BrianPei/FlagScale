#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# The validated MetaX base image owns Torch, Megatron-LM-FL and
# TransformerEngine-FL. This script installs only FlagScale-level requirements.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/metax/train.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

validate_runtime() {
    [ "$DEBUG" = true ] && return 0
    python - <<'PY'
import torch
import transformer_engine
from megatron.core.models.gpt import GPTModel

print("torch:", torch.__version__)
print("transformer_engine:", transformer_engine.__file__)
print("megatron GPTModel:", GPTModel)
PY
}

main() {
    set_step "Installing MetaX train requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || die "MetaX train pip failed"
    validate_runtime || die "MetaX train runtime validation failed"
    log_success "MetaX train runtime ready"
}

main
