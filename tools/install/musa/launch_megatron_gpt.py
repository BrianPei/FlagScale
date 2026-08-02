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

"""Select Megatron's MUSA platform before running GPT training."""

import os
import runpy
import sys
from pathlib import Path

# Hugging Face copies trusted tokenizer modules into this cache. Isolate each
# torchrun worker so concurrent imports cannot observe partially copied files.
os.environ.setdefault(
    "HF_MODULES_CACHE", f"/tmp/flagscale-hf-modules-{os.environ.get('LOCAL_RANK', '0')}"
)

import torch
import torch_musa  # noqa: F401
import flagscale
from megatron.plugin.platform import get_platform, set_platform
from megatron.plugin.platform.platform_register import PLATFORMS


if "musa" not in PLATFORMS:
    raise RuntimeError(f"Megatron-LM-FL did not register MUSA: {list(PLATFORMS)}")

# torch_musa exposes CUDA-compatible APIs, so Megatron's automatic selector can
# choose CUDA first. Select its registered MUSA implementation explicitly.
set_platform(PLATFORMS["musa"])
platform = get_platform()
if platform.device_name() != "musa":
    raise RuntimeError(f"Megatron-LM-FL selected {platform.device_name()}, expected musa")
if not torch.musa.is_available():
    raise RuntimeError("torch_musa reports that no MUSA device is available")

print("Megatron-LM-FL Platform: musa Selected")

train_script = (
    Path(flagscale.__file__).resolve().parent / "train" / "megatron" / "train_gpt.py"
)
if not train_script.is_file():
    raise RuntimeError(f"FlagScale training entrypoint not found: {train_script}")
sys.path.insert(0, str(train_script.parent))
runpy.run_path(str(train_script), run_name="__main__")
