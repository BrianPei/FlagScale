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

# Select one of the isolated runtimes embedded in the Ascend all image.
# This file is installed as <runtime>/activate.sh and is sourced without
# arguments by the reusable CI workflows.

set -e

# Resolve the requested runtime from this activation path every time. Reusing
# FLAGSCALE_RUNTIME here would make a previous activation sticky in the same
# shell (for example train followed by inference).
RUNTIME_NAME="$(basename "$(dirname "${BASH_SOURCE[0]}")")"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"

case "$RUNTIME_NAME" in
    train)
        PYTHON_HOME=/usr/local/python3.12.13
        ASCEND_HOME=/usr/local/Ascend
        ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
        NNAL_HOME="$FLAGSCALE_HOME/runtimes/train/nnal"
        MPI_HOME=/usr/local/mpi
        ;;
    inference|serve)
        PYTHON_HOME=/usr/local/python3.11.13
        ASCEND_HOME=/usr/local/Ascend
        ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
        NNAL_HOME=/usr/local/Ascend/nnal
        MPI_HOME=/usr/local/mpi
        export VLLM_PLUGINS=fl,ascend
        export TRITON_ALL_BLOCKS_PARALLEL=1
        ;;
    *)
        echo "Unsupported FlagScale runtime: $RUNTIME_NAME" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export FLAGSCALE_HOME FLAGSCALE_RUNTIME="$RUNTIME_NAME"
export ASCEND_HOME ASCEND_TOOLKIT_HOME NNAL_HOME MPI_HOME
export PATH="$PYTHON_HOME/bin:$NNAL_HOME/atb/latest/atb/cxx_abi_1/bin:$ASCEND_TOOLKIT_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NNAL_HOME/atb/latest/atb/cxx_abi_1/lib:$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/lib64/plugin/opskernel:$ASCEND_TOOLKIT_HOME/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$MPI_HOME/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

# The all-in-one image inherits the inference CANN Python path from its base
# image.  Reset it on every activation so the train runtime cannot import
# TBE/opp modules from the other CANN release (and vice versa).
export PYTHONPATH="$ASCEND_TOOLKIT_HOME/python/site-packages:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe"
export ASCEND_AICPU_PATH="$ASCEND_TOOLKIT_HOME"
export ASCEND_OPP_PATH="$ASCEND_TOOLKIT_HOME/opp"
export TOOLCHAIN_HOME="$ASCEND_TOOLKIT_HOME/toolkit"
export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
