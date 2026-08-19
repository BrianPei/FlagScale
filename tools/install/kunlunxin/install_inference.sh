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
# Pin to the vllm-plugin-FL release that targets vllm 0.13.0 (the version
# preinstalled in the P800 base image). Tracking `main` pulls code adapted for
# vllm 0.20.2, whose `from vllm.v1.attention.backends.registry import ...` does
# not exist on 0.13.0, so vLLM fails to resolve current_platform at import.
VLLM_PLUGIN_REF="${FLAGSCALE_VLLM_PLUGIN_REF:-v0.1.1+vllm0.13.0}"

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

    # Patch upstream vllm-plugin-FL for Kunlunxin P800: register the
    # kunlunxin vendor (VENDOR_DEVICE_MAP + supported_device) and fall back
    # to flag_gems' device_finder DeviceDetector on flag_gems >=5.0.3.
    # See patch_vllm_fl_kunlunxin.py; remove once upstream supports kunlunxin.
    python "$SCRIPT_DIR/patch_vllm_fl_kunlunxin.py" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    # Patch vllm_fl flagcx.py for the FlagCX wrapper API shipped in the P800
    # base image (flagcxGetUniqueId returns an object, flagcxCommInitRank takes
    # the object not a byref). See patch_flagcx_kunlunxin.py; remove once
    # vllm_fl matches the base-image FlagCX wrapper API.
    python "$SCRIPT_DIR/patch_flagcx_kunlunxin.py" \
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

# Install the triton.autotune compat shim into the target conda env so it is
# auto-imported (via the .pth file) before any `import flag_gems`, for every
# Python process in that env -- including vLLM spawn workers. Without it,
# vllm_fl:register -> vllm_fl/utils.py -> import flag_gems crashes at import
# time because the Kunlunxin flag_gems ops use triton.autotune(generate_configs=...),
# a keyword the P800 runtime triton does not accept. See triton_autotune_compat.py.
install_triton_autotune_compat() {
    set_step "Installing triton.autotune compat shim (generate_configs)"
    local conda_path="${FLAGSCALE_CONDA:-/root/miniconda}"
    local env_name="${FLAGSCALE_ENV_NAME:-python310_torch29_cuda}"
    local env_python="$conda_path/envs/$env_name/bin/python"
    [ -x "$env_python" ] || env_python="python"

    local site_dir
    site_dir="$("$env_python" -c 'import site; print(site.getsitepackages()[0])')" \
        || { log_error "Could not resolve site-packages for triton shim"; return 1; }

    install -m 0644 "$SCRIPT_DIR/triton_autotune_compat.py" \
        "$site_dir/flagscale_triton_autotune_compat.py"
    echo "import flagscale_triton_autotune_compat" \
        > "$site_dir/_flagscale_triton_autotune_compat.pth"
    log_success "triton.autotune compat shim installed at $site_dir"
}

main() {
    install_pip || die "Inference pip failed"
    install_vllm_plugin || die "vllm-plugin-FL failed"
    validate_vllm_plugin || die "vllm-plugin-FL validation failed"
    install_triton_autotune_compat || die "triton.autotune compat shim failed"
}

main
