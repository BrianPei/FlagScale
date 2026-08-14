#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

runtime_root=${FLAGSCALE_RUNTIME_ROOT:?FLAGSCALE_RUNTIME_ROOT is required}
task=$(cat "$runtime_root/task")

case "$task" in
    train)
        python_prefix=/opt/flagscale/runtimes/train/conda
        maca_home=/opt/flagscale/runtimes/train/maca
        unset VLLM_PLUGINS VLLM_FL_PLATFORM
        export TE_FL_SKIP_CUDA=1
        export NVTE_WITH_MACA=1
        ;;
    inference|serve)
        python_prefix=/opt/conda
        maca_home=/opt/maca
        export VLLM_PLUGINS=fl
        export VLLM_FL_PLATFORM=metax
        unset TE_FL_SKIP_CUDA NVTE_WITH_MACA
        ;;
    *)
        echo "Unsupported MetaX runtime task: $task" >&2
        return 1
        ;;
esac

export FLAGSCALE_RUNTIME_TASK="$task"
export CONDA_PREFIX="$python_prefix"
export FLAGSCALE_CONDA="$python_prefix"
export MACA_HOME="$maca_home"
export CUDA_HOME="$maca_home"
# Do not inherit the other task's Conda or MACA paths from the image ENV.
export PATH="$python_prefix/bin:$maca_home/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$python_prefix/lib:$maca_home/lib:/usr/local/mpi/lib64:/usr/local/mpi/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export PYTHONNOUSERSITE=1
