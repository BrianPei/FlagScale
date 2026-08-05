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

"""Initialize torch_musa before vLLM plugins are imported in child processes."""

import os


if (
    os.environ.get("FS_PLATFORM") == "musa"
    and os.environ.get("FLAGSCALE_MUSA_BUILD_NO_DEVICE", "false").lower() != "true"
):
    import torch_musa  # noqa: F401
