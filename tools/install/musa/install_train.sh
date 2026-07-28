#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Train task (MUSA): requirements/musa/train.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
REQ_FILE="$PROJECT_ROOT/requirements/musa/train.txt"
SRC_DEPS_LIST="megatron-lm"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing MUSA train requirements"
        retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "MUSA train requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing MUSA train pip packages (override)"
        run_cmd -d "$DEBUG" "$(get_pip_cmd)" install --root-user-action=ignore $pkgs || return 1
        log_success "MUSA train pip packages installed"
    fi
}

megatron_lm_ready() {
    python -c '
import megatron.core
from megatron.plugin.platform import get_platform
' &>/dev/null
}

install_megatron_lm() {
    if [ "${FLAGSCALE_FORCE_BUILD:-false}" != true ] && megatron_lm_ready; then
        log_info "Megatron-LM-FL is importable, skipping"
        return 0
    fi

    set_step "Installing Megatron-LM-FL for MUSA"
    mkdir -p "$FLAGSCALE_DEPS"
    retry_git_checkout_ref -d "$DEBUG" "$MEGATRON_REPO" "$MEGATRON_REF" \
        "$FLAGSCALE_DEPS/Megatron-LM-FL" "$RETRY_COUNT" || return 1

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    # The pinned source is verified with the vendor's Python 3.10 runtime, but
    # currently declares Python >=3.12 in package metadata. Keep this exception
    # explicit and fail below if the source stops being Python 3.10 compatible.
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/Megatron-LM-FL' && \
        $pip_cmd install --ignore-requires-python --root-user-action=ignore \
        --no-build-isolation . -v" || return 1
    megatron_lm_ready || return 1
    log_success "Megatron-LM-FL ready"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "megatron-lm" && {
        install_megatron_lm || die "Megatron-LM-FL failed"
    }
}

verify_musa_runtime() {
    set_step "Validating torch_musa runtime"
    "$(get_pip_cmd)" show torch-musa >/dev/null 2>&1 || \
        python -c "import torch_musa" || return 1
    python -c "import torch, torch_musa; assert hasattr(torch, 'musa')" || return 1
    log_success "torch_musa runtime is importable"
}

main() {
    install_pip || die "MUSA train pip failed"
    install_src
    verify_musa_runtime || die "MUSA runtime validation failed"
}

main
