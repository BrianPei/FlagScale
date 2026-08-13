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

# Serve task (Ascend): install FlagScale dependencies plus pinned FlagOS sources.
# The Ascend runtime image owns vLLM; source installs must not replace it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/ascend/serve.txt"
FLAGGEMS_REPO="${FLAGSCALE_FLAGGEMS_REPO:-https://github.com/flagos-ai/FlagGems.git}"
# This is the first upstream revision containing the Ascend zero-element
# tensor fix from FlagGems PR #5410. No released tag contains that fix yet.
FLAGGEMS_REF="${FLAGSCALE_FLAGGEMS_REF:-61f3ff2773bc9c8e86b97489775ef9668a96a33c}"
VLLM_PLUGIN_REPO="${FLAGSCALE_VLLM_PLUGIN_REPO:-https://github.com/flagos-ai/vllm-plugin-FL.git}"
# This revision contains the consolidated Ascend 910C and vLLM 0.20.2 updates
# from vllm-plugin-FL PR #347. Pin the commit until that support is released.
VLLM_PLUGIN_REF="${FLAGSCALE_VLLM_PLUGIN_REF:-43edeb601f4b8f616f56109de64836529e758deb}"

SRC_DEPS_LIST="flaggems vllm-plugin"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# =============================================================================
# Pip Installation
# =============================================================================
install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "serve.txt not found"; return 0; }
        set_step "Installing serve requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Serve requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing serve pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Serve pip packages installed"
    fi
}

checkout_source() {
    local repo="$1"
    local target="$2"
    local ref="$3"

    retry_git_clone -d "$DEBUG" "$repo" "$target" "$RETRY_COUNT" || return 1
    [ -z "$ref" ] && return 0
    retry -d "$DEBUG" "$RETRY_COUNT" \
        "git -C '$target' checkout --detach '$ref'" || return 1
}

install_flaggems() {
    should_build_package "flag_gems" || return 0
    set_step "Installing pinned FlagGems"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_source "$FLAGGEMS_REPO" "$FLAGSCALE_DEPS/FlagGems" "$FLAGGEMS_REF" || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/FlagGems' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation ." || return 1
    log_success "FlagGems ready at $FLAGGEMS_REF"
}

install_vllm_plugin() {
    should_build_package "vllm_plugin_fl" || return 0
    set_step "Installing pinned vllm-plugin-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_source \
        "$VLLM_PLUGIN_REPO" "$FLAGSCALE_DEPS/vllm-plugin-FL" "$VLLM_PLUGIN_REF" || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && \
        $pip_cmd install --root-user-action=ignore --no-build-isolation --no-deps ." || return 1
    log_success "vllm-plugin-FL ready at $VLLM_PLUGIN_REF"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "flaggems" && { install_flaggems || die "FlagGems failed"; }
    should_install_src task "vllm-plugin" && {
        install_vllm_plugin || die "vllm-plugin-FL failed"
    }
}

main() {
    install_pip || die "Serve pip failed"
    install_src
}

main
