#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

runtime_root=${FLAGSCALE_RUNTIME_ROOT:?FLAGSCALE_RUNTIME_ROOT is required}
task=$(cat "$runtime_root/task")
inherited_path=${PATH:-}
inherited_ld_library_path=${LD_LIBRARY_PATH:-}

case "$task" in
    train)
        python_prefix=/opt/flagscale/runtimes/train/conda
        maca_home=/opt/flagscale/runtimes/train/maca
        inherit_vendor_paths=false
        unset VLLM_PLUGINS VLLM_FL_PLATFORM
        export TE_FL_SKIP_CUDA=1
        export NVTE_WITH_MACA=1
        ;;
    inference|serve)
        python_prefix=/opt/conda
        maca_home=/opt/maca
        inherit_vendor_paths=true
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
runtime_path="$python_prefix/bin:$maca_home/bin:$maca_home/mxgpu_llvm/bin:$maca_home/ompi/bin:$maca_home/ucx/bin:/opt/mxdriver/bin"
runtime_ld_library_path="$python_prefix/lib:$maca_home/lib:$maca_home/mxshmem/lib:$maca_home/ompi/lib:$maca_home/ucx/lib:/opt/mxdriver/lib"

if [ "$inherit_vendor_paths" = true ]; then
    # The inference base owns additional vendor runtime paths. Preserve them so
    # native extensions such as deep_ep_cpp can resolve the bundled MPI/UCX.
    export PATH="$runtime_path:$inherited_path"
    export LD_LIBRARY_PATH="$runtime_ld_library_path:$inherited_ld_library_path"
else
    # Do not inherit inference Conda or MACA paths into the relocated train
    # runtime. Every vendor path below is rooted in the train-owned prefix.
    export PATH="$runtime_path:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export LD_LIBRARY_PATH="$runtime_ld_library_path:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
fi
export PYTHONNOUSERSITE=1
