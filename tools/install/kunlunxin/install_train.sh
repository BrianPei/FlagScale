#!/bin/bash
# Train task (Kunlunxin): requirements/kunlunxin/train.txt + runtime verification

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
FLAGSCALE_MEGATRON_PATH="${FLAGSCALE_MEGATRON_PATH:-/opt/Megatron-LM-FL}"
REQ_FILE="$PROJECT_ROOT/requirements/kunlunxin/train.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing train requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Train requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing train pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Train pip packages installed"
    fi
}

verify_kunlunxin_training_stack() {
    [ "$DEBUG" = true ] && { log_info "Skipping runtime verification in dry-run mode"; return 0; }
    set_step "Verifying Kunlunxin training stack"
    python - <<'PY'
import flagcx
import megatron.core
import torch
import transformer_engine
import transformer_engine_torch

assert hasattr(torch, "cuda"), "Kunlunxin torch fork must expose torch.cuda API"
print("Kunlunxin training stack import OK")
PY
}

main() {
    install_pip || die "Train pip failed"
    verify_kunlunxin_training_stack || die "Kunlunxin training stack verification failed"
}

main
