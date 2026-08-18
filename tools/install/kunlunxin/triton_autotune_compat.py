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

"""triton.autotune compatibility shim for Kunlunxin P800.

Why this exists
---------------
``vllm_fl/utils.py`` imports ``flag_gems`` unconditionally at module load. The
Kunlunxin flag_gems backend ops decorate kernels with::

    @triton.autotune(generate_configs=..., ...)

``generate_configs`` is a newer-triton autotune keyword that the P800 runtime
triton does not accept, so importing flag_gems raises ``TypeError`` at import
time, which breaks ``vllm_fl:register`` and leaves vLLM on the
``UnspecifiedPlatform`` (``Device string must not be empty``).

This shim wraps ``triton.autotune`` so the decorator can be applied without
crashing: it first tries the real call, and on ``TypeError`` retries after
dropping the known-incompatible kwargs. With ``USE_FLAGGEMS=false`` the flag_gems
ops are never dispatched, so the dropped kwargs only matter at import time and
do not affect runtime correctness -- inference runs on vendor kernels.

The module is installed next to a ``.pth`` file by ``install_inference.sh`` so
it auto-imports before any ``import flag_gems`` for every Python process in the
Kunlunxin inference/serve/all conda env (including vLLM ``spawn`` workers).
"""

import triton

# Keyword arguments that newer triton's autotune accepts but the P800 runtime
# triton rejects. Extend this set when on-box logs reveal further gaps.
_COMPAT_DROP = ("generate_configs",)

_orig_autotune = triton.autotune


def _compat_autotune(*args, **kwargs):
    try:
        return _orig_autotune(*args, **kwargs)
    except TypeError:
        cleaned = {k: v for k, v in kwargs.items() if k not in _COMPAT_DROP}
        if cleaned == kwargs:
            # The TypeError was not caused by a known incompatible kwarg; let it
            # propagate so a real bug is surfaced rather than silently masked.
            raise
        return _orig_autotune(*args, **cleaned)


if not getattr(triton, "_flagscale_autotune_compat", False):
    triton.autotune = _compat_autotune
    triton._flagscale_autotune_compat = True
