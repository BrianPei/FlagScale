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

# FlagGems ref. The official 20260812 PDF source-installs flagos-ai/FlagGems
# v5.0.0 (commit 4e08b44) OVER the image built-in flag_gems. The image built-in
# is the GENERIC flag_gems: its DeviceDetector selects the cuda backend, so
# torch.zeros dispatches to flag_gems/ops/zeros.py (generic CUDA triton) ->
# triton load_binary -> CUDA_ERROR_NOT_SUPPORTED on the P800, crashing both TP
# workers at init_device. It also crashes import on triton.language.math gaps
# (pow/asin) without a compat shim. The source build ships
# flag_gems._kunlunxin.ops.* (kunlunxin-adapted triton kernels whose load_binary
# succeeds) + a DeviceDetector that selects _kunlunxin, AND a triton-3.0.0-
# compatible tl_extra_shim (no import crash, no shim needed). The official PDF
# server log confirms flag_gems._kunlunxin.ops.* all run. No source_refs entry
# in .github/configs/kunlunxin.yml: the flaggems source policy=latest_release
# would override v5.0.0; the v5.0.0 default here pins the official version.
FLAGGEMS_REPO="${FLAGSCALE_FLAGGEMS_REPO:-https://github.com/flagos-ai/FlagGems.git}"
FLAGGEMS_REF="${FLAGSCALE_FLAGGEMS_REF:-v5.0.0}"

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
    # ...) -- so it idempotently skips. main leaves VLLM_FL_PREFER unset (see
    # the case yaml) so vllm_fl's use_flaggems() stays True and attention
    # dispatches through the flag_gems triton path -- the official 20260812 PDF
    # shows this producing correct decode output on the official runtime image.
    # The VLLM_FL_PREFER=vendor path (vendor XFlashAttention) decoded garbled
    # output and is dropped; 645116a's official runtime image (Triton 3.0.0
    # kunlunxin-adapted) makes the flag_gems triton path no longer assert on
    # libcuda.so.1. Kept for the v0.1.1 local fallback. See
    # patch_attention_backend_kunlunxin.py.
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

# The triton.autotune compat shim (triton_autotune_compat.py) is NO LONGER
# installed. It was a workaround for the image built-in GENERIC flag_gems,
# whose import crashed on triton.language.math gaps (pow/asin) and
# triton.autotune(generate_configs). The proper fix is install_flaggems above:
# source-install flagos-ai/FlagGems v5.0.0 (per the official 20260812 PDF),
# whose tl_extra_shim is triton-3.0.0-compatible (no import crash, no shim
# needed) AND ships flag_gems._kunlunxin.ops.* + a DeviceDetector that selects
# the _kunlunxin backend. The shim also actively broke backend selection once
# flag_gems was dispatched (USE_FLAGGEMS=true after dropping VLLM_FL_PREFER=
# vendor): its _FlagGemsCompatFinder re-patched triton mid-import, corrupting
# DeviceDetector. With the source FlagGems install, neither the import crash
# nor the backend corruption occurs, so the shim is obsolete. If a future base
# image reintroduces the import crash, restore this step and the .pth from
# triton_autotune_compat.py. See git history for the removed
# install_triton_autotune_compat() function.

install_flaggems() {
    set_step "Installing resolved FlagGems $FLAGGEMS_REF (source, over image built-in)"
    mkdir -p "$FLAGSCALE_DEPS"
    checkout_pinned_ref "$FLAGGEMS_REPO" "$FLAGGEMS_REF" \
        "$FLAGSCALE_DEPS/FlagGems" || return 1
    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    # Uninstall the image built-in generic flag_gems so the source build
    # (flag_gems._kunlunxin.ops.* + DeviceDetector) takes precedence. Without
    # this, the generic ops stay on sys.path and torch.zeros dispatches to
    # flag_gems/ops/zeros.py -> triton load_binary -> CUDA_ERROR_NOT_SUPPORTED.
    run_cmd -d "$DEBUG" bash -c \
        "$pip_cmd uninstall -y flag_gems flag-gems" || true
    # Build deps the official PDF installs (FlagGems v5.0.0 ships C extensions
    # built via scikit-build-core).
    run_cmd -d "$DEBUG" $pip_cmd install --root-user-action=ignore \
        "scikit-build-core==0.11" pybind11 ninja cmake sqlalchemy || return 1
    run_cmd -d "$DEBUG" bash -c \
        "cd '$FLAGSCALE_DEPS/FlagGems' && $pip_cmd install \
        --root-user-action=ignore --no-build-isolation --no-deps ." || return 1
    log_success "FlagGems $FLAGGEMS_REF (source) installed over image built-in"
}

main() {
    install_pip || die "Inference pip failed"
    install_vllm || die "vLLM install failed"
    install_vllm_plugin || die "vllm-plugin-FL failed"
    validate_vllm_plugin || die "vllm-plugin-FL validation failed"
    install_flaggems || die "FlagGems source install failed"
}

main
