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

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"

[ "$phase" = post ] || exit 0

case "$task" in
    train)
        env_name=flagscale-train
        imports='import megatron.core; import transformer_engine'
        ;;
    inference)
        env_name=flagscale-inference
        imports='import vllm'
        ;;
    all)
        env_name=flagscale-all
        imports='import megatron.core; import transformer_engine; import vllm'
        ;;
    *)
        echo "Unsupported CUDA image task: $task" >&2
        exit 1
        ;;
esac

docker run --rm --gpus all \
    --entrypoint /root/miniconda3/bin/conda \
    "$candidate" run -n "$env_name" python -c "
import torch

assert torch.cuda.is_available()
value = torch.tensor(range(8), dtype=torch.float32, device='cuda')
assert (value * 2).cpu().tolist() == [0., 2., 4., 6., 8., 10., 12., 14.]
$imports
print('CUDA image validation passed:', '$task')
"
