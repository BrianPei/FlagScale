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

# Train task (MUSA): requirements/musa/train.txt + native MUSA TransformerEngine

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
FLAGSCALE_DOWNLOADS="${FLAGSCALE_DOWNLOADS:-$FLAGSCALE_HOME/downloads}"
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
TE_SOURCE_DIR="${FLAGSCALE_MUSA_TE_SOURCE_DIR:-$FLAGSCALE_DEPS/TransformerEngine-MUSA}"
TE_REF="${FLAGSCALE_TE_REF:-e73781e85518ab0046007cef69e95bd258a63900}"
REQ_FILE="$PROJECT_ROOT/requirements/musa/train.txt"
SRC_DEPS_LIST="transformer-engine megatron-lm"

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
    # A device-less Docker build must install the pinned source instead of
    # importing Megatron through an incomplete driver placeholder. At runtime,
    # validate with torch_musa auto-loading enabled.
    [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ] && return 1
    python -c '
import megatron.core
from megatron.plugin.platform import get_platform
' &>/dev/null
}

validate_megatron_lm() {
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        "$(get_pip_cmd)" show megatron-core >/dev/null 2>&1 || return 1
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c '
import importlib.util
assert importlib.util.find_spec("megatron") is not None
' || return 1
        log_success "Megatron-LM-FL package is installed; import validation deferred to runtime"
        return 0
    fi
    python -c '
import megatron.core
from megatron.plugin.platform import get_platform
print("Megatron-LM-FL import validation passed")
'
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
    validate_megatron_lm || return 1
    log_success "Megatron-LM-FL ready"
}

install_transformer_engine() {
    set_step "Building native TransformerEngine for MUSA"
    [ -f "$TE_SOURCE_DIR/setup.py" ] || {
        log_error "MUSA TransformerEngine source not found: $TE_SOURCE_DIR"
        return 1
    }

    local actual_ref
    actual_ref=$(git -C "$TE_SOURCE_DIR" rev-parse HEAD) || return 1
    [ "$actual_ref" = "$TE_REF" ] || {
        log_error "MUSA TransformerEngine source is $actual_ref, expected $TE_REF"
        return 1
    }

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    local wheel_dir="$FLAGSCALE_DOWNLOADS/transformer-engine-musa"
    rm -rf "$wheel_dir"
    mkdir -p "$wheel_dir"

    run_cmd -d "$DEBUG" bash -c "cd '$TE_SOURCE_DIR' && \
        rm -rf build dist transformer_engine.egg-info && \
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 NVTE_FRAMEWORK=musa \
        $pip_cmd wheel . --wheel-dir '$wheel_dir' --no-deps \
        --use-pep517 --no-build-isolation -v" || return 1

    local wheel
    wheel=$(find "$wheel_dir" -maxdepth 1 -type f \
        -name 'transformer_engine-*.whl' -print -quit)
    [ -n "$wheel" ] || {
        log_error "Native MUSA TransformerEngine wheel was not produced"
        return 1
    }
    run_cmd -d "$DEBUG" "$pip_cmd" install --force-reinstall --no-deps \
        --root-user-action=ignore "$wheel" || return 1

    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c '
import importlib.metadata as metadata
files = [str(path) for path in metadata.files("transformer-engine") or ()]
assert any("transformer_engine_torch" in path and path.endswith(".so") for path in files), files
' || return 1
        log_success "Native extension packaged; device validation deferred to runtime"
    else
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c '
import transformer_engine
import transformer_engine_torch as tex
for symbol in ("generic_gemm", "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd"):
    assert hasattr(tex, symbol), symbol
print(transformer_engine.__version__)
' || return 1
    fi
    rm -rf "$wheel_dir"
    log_success "Native MUSA TransformerEngine ready"
}

install_musa_launcher() {
    set_step "Installing MUSA training launcher"
    install -D -m 0755 "$SCRIPT_DIR/launch_megatron_gpt.py" \
        "$FLAGSCALE_HOME/bin/launch_megatron_gpt.py" || return 1
    log_success "MUSA training launcher ready"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "transformer-engine" && {
        install_transformer_engine || die "Native MUSA TransformerEngine failed"
    }
    should_install_src task "megatron-lm" && {
        install_megatron_lm || die "Megatron-LM-FL failed"
    }
}

verify_musa_runtime() {
    set_step "Validating torch_musa runtime"
    "$(get_pip_cmd)" show torch-musa >/dev/null 2>&1 || return 1
    if [ "${FLAGSCALE_MUSA_BUILD_NO_DEVICE:-false}" = true ]; then
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import torch" || return 1
        log_success "torch_musa package is installed; device validation deferred to runtime"
        return 0
    fi
    python -c "import torch, torch_musa; assert hasattr(torch, 'musa')" || return 1
    log_success "torch_musa runtime is importable"
}

main() {
    install_pip || die "MUSA train pip failed"
    install_src
    install_musa_launcher || die "MUSA training launcher failed"
    verify_musa_runtime || die "MUSA runtime validation failed"
}

main
