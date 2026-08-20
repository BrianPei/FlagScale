#!/bin/bash
# =============================================================================
# FlagScale Kunlunxin Environment Variables
# =============================================================================

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${UV_PROJECT_ENVIRONMENT:=$FLAGSCALE_HOME/venv}"
: "${FLAGSCALE_CONDA:=/root/miniconda}"
: "${FLAGSCALE_ENV_NAME:=python310_torch29_cuda}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${MPI_HOME:=/usr/local/mpi}"
: "${KLX_HOME:=/opt/xccl_Linux_x86_64}"
: "${FLAGSCALE_MEGATRON_PATH:=$FLAGSCALE_DEPS/Megatron-LM-FL}"

# Kunlunxin XRE runtime (libxpurt/libxpuml/libxpucuda) -- required by
# libflagcx.so for the vllm_fl CommunicatorFL (TP/EP collectives over XCCL).
# Without these on the loader path, libflagcx.so fails to dlopen and vLLM
# silently falls back off the flagcx path to the NCCL path, which cannot run
# on the P800 (NCCL calls CUDA APIs that torch_xmlir does not implement ->
# "NCCL error: unhandled cuda error" at ncclCommInitRank). The xre dir is
# versioned (e.g. /opt/xre-Linux-x86_64-5.24.0.0) with no symlink/env var, so
# glob the installed one to stay robust across xre upgrades.
if [ -z "${XRE_HOME:-}" ]; then
    for _d in /opt/xre-Linux-x86_64-*; do
        [ -d "$_d/so" ] && XRE_HOME="$_d" && break
    done
fi
# flagcx (FlagOS collective lib over XCCL) + its python wrapper. Setting
# FLAGCX_PATH switches vllm_fl PlatformFL.dist_backend to "flagcx" so
# CommunicatorFL (real flagcx, full op coverage incl. reduce_scatter/all_gather)
# is used instead of CudaCommunicator (NCCL, unusable on P800). The wrapper
# ships differently per base image, so auto-detect (like XRE_HOME):
#  - Official Kunlunxin runtime image: pip editable install. An egg-link sits
#    in site-packages but its source dir is NOT on sys.path (no
#    easy-install.pth / conda site skips .pth), so `import flagcx` (re-issued
#    by the torch_xmlir import hook via __origin__import__) misses. Point
#    FLAGCX_PATH at the egg-link source dir so PYTHONPATH picks it up.
#  - flagos-dev manual base: flagcx under /opt/FlagCX (no egg-link) -> the
#    /opt/FlagCX default below covers it. (A /plugin subdir check was tried
#    first but it matched /opt/FlagCX on the runtime image too -- which has a
#    /plugin dir but not the flagcx package -- shadowing the egg-link fallback.)
# A caller-set value wins.
if [ -z "${FLAGCX_PATH:-}" ]; then
    for _el in \
        "$FLAGSCALE_CONDA"/envs/"$FLAGSCALE_ENV_NAME"/lib/python*/site-packages/flagcx.egg-link \
        /usr/lib/python*/site-packages/flagcx.egg-link; do
        [ -f "$_el" ] || continue
        _dir=$(head -1 "$_el" 2>/dev/null)
        [ -n "$_dir" ] && [ -d "$_dir" ] && FLAGCX_PATH="$_dir" && break
    done
fi
: "${FLAGCX_PATH:=/opt/FlagCX}"

: "${UV_HTTP_TIMEOUT:=500}"
: "${UV_INDEX_STRATEGY:=unsafe-best-match}"
: "${UV_LINK_MODE:=copy}"
: "${TE_FL_SKIP_CUDA:=1}"

export FLAGSCALE_HOME FLAGSCALE_CONDA FLAGSCALE_ENV_NAME FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export UV_PROJECT_ENVIRONMENT MPI_HOME KLX_HOME XRE_HOME FLAGCX_PATH FLAGSCALE_MEGATRON_PATH TE_FL_SKIP_CUDA
export UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

export PATH="$FLAGSCALE_CONDA/envs/$FLAGSCALE_ENV_NAME/bin:$UV_PROJECT_ENVIRONMENT/bin:$FLAGSCALE_CONDA/bin:$HOME/.local/bin:$MPI_HOME/bin:$KLX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${XRE_HOME:+$XRE_HOME/so:}$KLX_HOME/so:$MPI_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$FLAGSCALE_MEGATRON_PATH:$FLAGCX_PATH:${PYTHONPATH:-}"
