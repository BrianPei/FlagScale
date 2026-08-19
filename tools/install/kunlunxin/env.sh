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
# is used instead of CudaCommunicator (NCCL, unusable on P800).
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
