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
# vllm-plugin-FL ref. CI image builds inject the main HEAD sha via the
# FLAGSCALE_VLLM_PLUGIN_REF Dockerfile ARG (.github/configs/kunlunxin.yml
# source_refs -> vllm_plugin_fl, image_sources.yml branch main), so main is
# used in CI. main (PR #268) ships native kunlunxin support (VENDOR_DEVICE_MAP
# + vendor attention backend + is_cuda fast path), replacing the v0.1.1 +
# 4-source-patch path. The P800 base vllm (0.2.0+g38e7dbc...) shares the
# vllm-plugin-FL main tree sha, so the two are the same ecosystem -- main is
# the supported match, not a mismatch. The v0.1.1 default below is only a
# local fallback when the ARG is unset (running this script outside a build).
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

# vLLM ref. The P800 base image ships a vendor vllm that is NOT 0.20.2, which
# breaks vllm-plugin-FL main's register() (the plugin imports vllm 0.20.2 APIs
# the base vllm lacks -> register() throws -> vllm swallows it ->
# current_platform stays UnspecifiedPlatform). The official Kunlunxin record
# (/home/wyi/下载/kunlunxin+vllm0.20.2+vllm-plugin-fl.pdf) uninstalls the image
# vllm and rebuilds v0.20.2+empty before installing the plugin; mirror that.
# VLLM_TARGET_DEVICE=empty skips CUDA/Triton compile (reuses the image's
# torch_xmlir/triton); --no-deps keeps image deps (torch/triton/flag_gems) intact.
VLLM_REF="${FLAGSCALE_VLLM_REF:-v0.20.2}"
VLLM_REPO="${FLAGSCALE_VLLM_REPO:-https://github.com/vllm-project/vllm.git}"

install_vllm() {
    set_step "Installing vLLM $VLLM_REF (VLLM_TARGET_DEVICE=empty)"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$VLLM_REPO" "$VLLM_REF" \
        "$FLAGSCALE_DEPS/vllm" || return 1

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    # Uninstall the image's vendor vllm so the 0.20.2 build takes precedence.
    run_cmd -d "$DEBUG" bash -c "$pip_cmd uninstall -y vllm vllm_xpu" || true
    # setuptools_scm is required to build vllm from source.
    run_cmd -d "$DEBUG" $pip_cmd install --root-user-action=ignore setuptools_scm || return 1
    # empty target = no CUDA/Triton compile; --no-deps keeps image torch/triton.
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/vllm' && VLLM_TARGET_DEVICE=empty $pip_cmd install -v \
        --no-build-isolation --no-deps ." || return 1
    # vllm 0.20.2 requires numpy<2.0 (official record pins this).
    run_cmd -d "$DEBUG" $pip_cmd install --root-user-action=ignore "numpy<2.0" || return 1
    log_success "vLLM $VLLM_REF (VLLM_TARGET_DEVICE=empty) installed"
}

install_vllm_plugin() {
    set_step "Installing resolved vllm-plugin-FL"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$VLLM_PLUGIN_REPO" "$VLLM_PLUGIN_REF" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    # Register the kunlunxin vendor (VENDOR_DEVICE_MAP + supported_device).
    # On main (PR #268) this is already native, so the patch's old anchor is
    # absent and it idempotently skips. Kept for the v0.1.1 local fallback.
    # See patch_vllm_fl_kunlunxin.py.
    python "$SCRIPT_DIR/patch_vllm_fl_kunlunxin.py" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    # Patch vllm_fl flagcx.py for the FlagCX wrapper API shipped in the P800
    # base image (flagcxGetUniqueId returns an object, flagcxCommInitRank takes
    # the object not a byref). See patch_flagcx_kunlunxin.py; remove once
    # vllm_fl matches the base-image FlagCX wrapper API.
    python "$SCRIPT_DIR/patch_flagcx_kunlunxin.py" \
        "$FLAGSCALE_DEPS/vllm-plugin-FL" || return 1

    # Force the FlagGems AttentionFLBackend for kunlunxin (v0.1.1-only patch).
    # On main the anchor is gone -- get_attn_backend_cls now calls the
    # module-level _attention_backend CachedOp, not call_op("attention_backend",
    # ...) -- so it idempotently skips. main uses VLLM_FL_PREFER=vendor (set in
    # the case yaml) -> the native kunlunxin vendor attention backend
    # (torch_xmlir XFlashAttention, not triton), which is the supported path and
    # avoids the libcuda.so.1 assert of the FlagGems triton path. Kept for the
    # v0.1.1 local fallback. See patch_attention_backend_kunlunxin.py.
    python "$SCRIPT_DIR/patch_attention_backend_kunlunxin.py" \
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
    install_vllm || die "vLLM install failed"
    install_vllm_plugin || die "vllm-plugin-FL failed"
    validate_vllm_plugin || die "vllm-plugin-FL validation failed"
    install_triton_autotune_compat || die "triton.autotune compat shim failed"
}

main
