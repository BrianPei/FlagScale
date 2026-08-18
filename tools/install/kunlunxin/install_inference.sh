#!/bin/bash
# Inference task (Kunlunxin): requirements/kunlunxin/inference.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
REQ_FILE="$PROJECT_ROOT/requirements/kunlunxin/inference.txt"
VLLM_PLUGIN_REPO="${FLAGSCALE_VLLM_PLUGIN_REPO:-https://github.com/flagos-ai/vllm-plugin-FL.git}"
VLLM_PLUGIN_REF="${FLAGSCALE_VLLM_PLUGIN_REF:-main}"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "inference.txt not found"; return 0; }
        set_step "Installing inference requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Inference requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing inference pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Inference pip packages installed"
    fi
}

checkout_pinned_ref() {
    local repo=$1
    local ref=$2
    local target=$3

    [ -z "$ref" ] && { log_error "A pinned git ref is required"; return 1; }
    retry -d "$DEBUG" "$RETRY_COUNT" "rm -rf '$target' && \
        git init -q '$target' && \
        git -C '$target' remote add origin '$repo' && \
        git -c http.version=HTTP/1.1 -C '$target' fetch --depth 1 origin '$ref' && \
        git -C '$target' checkout -q --detach FETCH_HEAD"
}

install_vllm_plugin() {
    set_step "Installing resolved vllm-plugin-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$VLLM_PLUGIN_REPO" "$VLLM_PLUGIN_REF" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/vllm-plugin-FL' && $pip_cmd install \
        --root-user-action=ignore --no-build-isolation --no-deps ." || return 1
    log_success "vllm-plugin-FL ready at $VLLM_PLUGIN_REF"
}

validate_vllm_plugin() {
    [ "$DEBUG" = true ] && return 0
    python - <<'PY'
import importlib.metadata as metadata

entrypoints = {
    entry.name: entry.value
    for entry in metadata.entry_points(group="vllm.platform_plugins")
}
print("vllm:", metadata.version("vllm"))
print("vllm-plugin-fl:", metadata.version("vllm-plugin-fl"))
print("platform_plugins:", entrypoints)
assert entrypoints.get("fl") == "vllm_fl:register", entrypoints
PY
}

main() {
    install_pip || die "Inference pip failed"
    install_vllm_plugin || die "vllm-plugin-FL failed"
    validate_vllm_plugin || die "vllm-plugin-FL validation failed"
}

main
