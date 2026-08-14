#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-2}"

[ "$phase" = post ] || exit 0

docker_args=(
  --privileged
  --env MTHREADS_VISIBLE_DEVICES=all
  --env MTHREADS_DRIVER_CAPABILITIES=all
)

docker run --rm "${docker_args[@]}" \
  --env EXPECTED_WORLD_SIZE="$nproc" "$candidate" \
  python -c '
import os
import torch
import torch_musa

assert torch.musa.is_available()
assert torch.musa.device_count() >= int(os.environ["EXPECTED_WORLD_SIZE"])
x = torch.tensor(range(8), dtype=torch.float32, device="musa")
assert (x * 2).cpu().tolist() == [0., 2., 4., 6., 8., 10., 12., 14.]
'

if [ "$task" != train ]; then
  docker run --rm "${docker_args[@]}" "$candidate" \
    python -c '
import torch_musa
import vllm

assert hasattr(torch_musa, "_MUSAC")
print(vllm.__version__)
'
fi

if [ "$task" != inference ]; then
  docker run --rm "${docker_args[@]}" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    --env TORCHDYNAMO_DISABLE=1 \
    --env TORCH_COMPILE_DISABLE=1 \
    --env NVTE_TORCH_COMPILE=0 \
    "$candidate" python -c '
import torch
import torch_musa
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch import Linear

for symbol in ("generic_gemm", "layernorm_fwd", "layernorm_bwd", "rmsnorm_fwd", "rmsnorm_bwd"):
    assert hasattr(tex, symbol), symbol
torch.musa.set_device(0)
layer = Linear(16, 8).to("musa")
value = torch.randn(4, 16, device="musa", requires_grad=True)
output = layer(value)
output.sum().backward()
assert bool(output.isfinite().all().item())
assert value.grad is not None and bool(value.grad.isfinite().all().item())
assert layer.weight.grad is not None and bool(layer.weight.grad.isfinite().all().item())
print("Native MUSA TransformerEngine:", transformer_engine.__version__)
'
fi

if [ "$task" != inference ]; then
  docker run --rm "${docker_args[@]}" "$candidate" python -c '
import torch_musa
import megatron.core
from megatron.plugin.platform import get_platform

assert get_platform().device_name() == "musa"
'

  docker run --rm --ipc=host "${docker_args[@]}" \
    --env EXPECTED_WORLD_SIZE="$nproc" "$candidate" \
    torchrun --standalone --nproc_per_node="$nproc" --no-python python -c '
import os
import torch
import torch_musa

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["EXPECTED_WORLD_SIZE"])
torch.musa.set_device(rank)
torch.distributed.init_process_group(backend="mccl")
value = torch.tensor([rank], dtype=torch.int64, device="musa")
torch.distributed.all_reduce(value)
assert value.item() == world * (world - 1) // 2
torch.distributed.destroy_process_group()
'
fi
