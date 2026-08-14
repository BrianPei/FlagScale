#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

runtime_root=${FLAGSCALE_RUNTIME_ROOT:?FLAGSCALE_RUNTIME_ROOT is required}
task=$(cat "$runtime_root/task")

case "$task" in
    train|inference)
        python_prefix="$runtime_root/venv"
        ;;
    serve)
        python_prefix="$(dirname "$runtime_root")/inference/venv"
        ;;
    *)
        echo "Unsupported MUSA runtime task: $task" >&2
        return 1
        ;;
esac

[ -f "$python_prefix/bin/activate" ] || {
    echo "MUSA runtime venv not found: $python_prefix" >&2
    return 1
}

source "$python_prefix/bin/activate"

export FLAGSCALE_RUNTIME_TASK="$task"
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export MPI_HOME=${MPI_HOME:-/usr/local/openmpi}
export FS_PLATFORM=musa
export PATH="$python_prefix/bin:$MUSA_HOME/bin:$MPI_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MPI_HOME/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
