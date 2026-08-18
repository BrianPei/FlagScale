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

"""triton compatibility shim for Kunlunxin P800 flag_gems import.

Why this exists
---------------
``vllm_fl/utils.py`` imports ``flag_gems`` unconditionally at module load, and
``DeviceInfo`` (Kunlunxin platform detection) depends on
``flag_gems.runtime.backend``. So flag_gems MUST be importable for vLLM to load
the fl plugin onto the Kunlunxin platform at all -- otherwise vLLM falls back to
``UnspecifiedPlatform`` (``Device string must not be empty``) and inference
cannot run, even on vendor kernels.

But importing flag_gems on the P800 runtime triton hits multiple newer-triton
API gaps at import time, for example:

* ``flag_gems/ops/arcsin.py``: ``_ASIN = tl_extra_shim.asin`` ->
  ``AttributeError: module 'triton.language.math' has no attribute 'asin'``.
* ``flag_gems/runtime/backend/_kunlunxin/ops/addmm.py``:
  ``@triton.autotune(generate_configs=...)`` ->
  ``TypeError: autotune() got an unexpected keyword argument 'generate_configs'``.

These are device-side intrinsics/autotune kwargs the P800 runtime triton does not
provide. With ``USE_FLAGGEMS=false`` the flag_gems ops are never dispatched, so
the missing intrinsics are never actually executed for compute -- the shim only
needs to let flag_gems be IMPORTED for platform detection; inference then runs on
vendor kernels.

What the shim does
------------------
1. ``triton.language.math``: PEP 562 ``__getattr__`` returns a dummy callable for
   any missing math intrinsic (``asin``, ``acos``, ``atan``, ...) so module-level
   attribute access in flag_gems ops does not raise. The dummy raises loudly if
   ever actually called, so a real dispatch through a missing intrinsic fails
   visibly instead of silently computing wrong values.
2. ``triton.autotune``: tolerate newer-triton kwargs (``generate_configs``, ...)
   the P800 runtime rejects, by retrying without them on ``TypeError``.

Existing attributes / successful autotune calls pass through unchanged, so this is
a no-op for code paths that work today (including the train path in the ``all``
image).

Installed next to a ``.pth`` file by ``install_inference.sh`` so it auto-imports
before any ``import flag_gems`` for every Python process in the Kunlunxin
inference/serve/all conda env (including vLLM ``spawn`` workers).
"""

import triton
import triton.language.math as _tl_math

# Keyword arguments that newer triton's autotune accepts but the P800 runtime
# triton rejects. Extend this set when on-box logs reveal further gaps.
_COMPAT_AUTOTUNE_DROP = ("generate_configs",)


def _make_missing_math_dummy(name):
    def _missing_math(*args, **kwargs):  # pragma: no cover - only on real dispatch
        raise RuntimeError(
            f"triton.language.math.{name} is not provided by the P800 runtime "
            "triton; the flag_gems op using it must not be dispatched "
            "(set USE_FLAGGEMS=false)."
        )
    _missing_math.__name__ = name
    return _missing_math


# PEP 562: attribute access on a module falls back to __getattr__ when normal
# lookup fails. We use it to synthesize dummies for math intrinsics the P800
# runtime triton lacks (asin/acos/atan/...), so flag_gems' module-level
# ``_X = tl_extra_shim.<name>`` does not raise at import.
_orig_math_getattr = getattr(_tl_math, "__getattr__", None)


def _math_getattr(name):
    if _orig_math_getattr is not None:
        try:
            return _orig_math_getattr(name)
        except AttributeError:
            pass
    return _make_missing_math_dummy(name)


if not getattr(_tl_math, "_flagscale_math_compat", False):
    _tl_math.__getattr__ = _math_getattr
    _tl_math._flagscale_math_compat = True

# --- triton.autotune kwarg tolerance ----------------------------------------

_orig_autotune = triton.autotune


def _compat_autotune(*args, **kwargs):
    try:
        return _orig_autotune(*args, **kwargs)
    except TypeError:
        cleaned = {k: v for k, v in kwargs.items() if k not in _COMPAT_AUTOTUNE_DROP}
        if cleaned == kwargs:
            # The TypeError was not caused by a known incompatible kwarg; let it
            # propagate so a real bug is surfaced rather than silently masked.
            raise
        return _orig_autotune(*args, **cleaned)


if not getattr(triton, "_flagscale_autotune_compat", False):
    triton.autotune = _compat_autotune
    triton._flagscale_autotune_compat = True
