#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

runtime_root=${FLAGSCALE_RUNTIME_ROOT:?FLAGSCALE_RUNTIME_ROOT is required}
task=$(cat "$runtime_root/task")

case "$task" in
    train)
        python_home=/usr/local/python3.12.13
        ascend_home=/opt/flagscale/runtimes/train/Ascend
        unset VLLM_PLUGINS VLLM_FL_PLATFORM TRITON_ALL_BLOCKS_PARALLEL
        ;;
    inference|serve)
        python_home=/usr/local/python3.11.15
        ascend_home=/usr/local/Ascend
        export VLLM_PLUGINS=fl
        export VLLM_FL_PLATFORM=ascend
        export TRITON_ALL_BLOCKS_PARALLEL=1
        ;;
    *)
        echo "Unsupported Ascend runtime task: $task" >&2
        return 1
        ;;
esac

toolkit_home="$ascend_home/cann-9.0.0"
export FLAGSCALE_RUNTIME_TASK="$task"
export ASCEND_HOME="$ascend_home"
export ASCEND_HOME_PATH="$toolkit_home"
export ASCEND_TOOLKIT_HOME="$toolkit_home"
export ASCEND_OPP_PATH="$toolkit_home/opp"
# Use deterministic paths so activating the train runtime cannot retain the
# inference image's Python/CANN libraries (or vice versa).
export PATH="$python_home/bin:$toolkit_home/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$toolkit_home/lib64:$ascend_home/nnal/atb/9.0.0/lib:$ascend_home/nnal/asdsip/9.0.0/lib:/usr/local/Ascend/driver/lib64:/usr/local/mpi/lib64:/usr/local/mpi/lib:/usr/local/lib:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu"
export PYTHONNOUSERSITE=1
