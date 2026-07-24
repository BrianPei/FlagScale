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

"""Initialize the Ascend training stack before running Megatron GPT training."""

import runpy
import sys
from pathlib import Path

import torch_npu
import transformer_engine
from transformer_engine.plugin.core.backends.vendor.npu.patches import apply_patch


apply_patch()
if transformer_engine.te_device_type() != "npu":
    raise RuntimeError(
        f"TransformerEngine selected {transformer_engine.te_device_type()}, expected npu"
    )

# Megatron's NPU registration imports transfer_to_npu, so select TE's NPU
# backend first while the original torch.cuda namespace is still intact.
from megatron.plugin.platform import get_platform


platform = get_platform()
if platform.device_name() != "npu":
    raise RuntimeError(f"Megatron-LM-FL selected {platform.device_name()}, expected npu")
if not torch_npu.npu.is_available():
    raise RuntimeError("torch_npu reports that no NPU is available")

train_script = Path("flagscale/train/megatron/train_gpt.py").resolve()
sys.path.insert(0, str(train_script.parent))
runpy.run_path(str(train_script), run_name="__main__")
