#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

runtime_root=${FLAGSCALE_RUNTIME_ROOT:?FLAGSCALE_RUNTIME_ROOT is required}
task=$(cat "$runtime_root/task")
conda_root=${FLAGSCALE_CONDA:-/opt/flagscale/miniconda3}

case "$task" in
    train)
        env_name=flagscale-train
        ;;
    inference|serve)
        env_name=flagscale-inference
        ;;
    *)
        echo "Unsupported CUDA runtime task: $task" >&2
        return 1
        ;;
esac

[ -f "$conda_root/etc/profile.d/conda.sh" ] || {
    echo "CUDA Conda activation script not found: $conda_root" >&2
    return 1
}

source "$conda_root/etc/profile.d/conda.sh"
conda activate "$env_name"

export FLAGSCALE_RUNTIME_TASK="$task"
export FLAGSCALE_CONDA="$conda_root"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MPI_HOME=${MPI_HOME:-/usr/local/mpi}
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export PATH="$CONDA_PREFIX/bin:$MPI_HOME/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export PYTHONNOUSERSITE=1
