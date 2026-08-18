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

"""triton compatibility shim for Kunlunxin P800 flag_gems import (lazy).

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

Why the patch is LAZY (do not import triton at startup)
-------------------------------------------------------
This module is auto-imported by a ``.pth`` file at Python interpreter startup,
which runs BEFORE ``import torch``. An eager top-level ``import triton`` here was
observed to perturb the CUDA runtime library load order: the vendor
``torch_xmlir/_XMLIRC.so`` then failed with
``undefined symbol: cudaHostPointerGetAttributes, version libcudart.so.12`` and
``torch.cuda.is_available()`` returned False -- the inference image build probe
died at its first ``assert torch.cuda.is_available()``, before any flag_gems /
vllm_fl code was reached.

So the module must NOT import triton at the top level. Instead it inserts a
``sys.meta_path`` finder that intercepts the FIRST ``import triton`` issued by
other code (which, in the inference path, happens inside ``import flag_gems``
during ``vllm_fl:register`` -- i.e. well after torch has already imported
cleanly). The finder lets the real triton load normally and then applies the
patch, before flag_gems' ``@triton.autotune`` decorators execute. Startup
therefore touches neither triton nor any CUDA library, and torch loads exactly as
it does without the shim.

Installed next to a ``.pth`` file by ``install_inference.sh`` so it auto-imports
for every Python process in the Kunlunxin inference/serve/all conda env (including
vLLM ``spawn`` workers).
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys

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


def _apply_patch():
    """Patch the already-imported ``triton`` package. Idempotent.

    Called from the meta_path finder's loader right after real triton has loaded,
    so triton and its submodules are guaranteed importable here without
    retriggering the finder (``sys.modules["triton"]`` is already populated).
    """
    import triton
    import triton.language.math as _tl_math

    # --- triton.language.math: synthesize dummies for missing intrinsics -----
    if not getattr(_tl_math, "_flagscale_math_compat", False):
        _orig_math_getattr = getattr(_tl_math, "__getattr__", None)

        def _math_getattr(name):
            if _orig_math_getattr is not None:
                try:
                    return _orig_math_getattr(name)
                except AttributeError:
                    pass
            return _make_missing_math_dummy(name)

        _tl_math.__getattr__ = _math_getattr
        _tl_math._flagscale_math_compat = True

    # --- triton.autotune: drop incompatible newer kwargs on TypeError --------
    if not getattr(triton, "_flagscale_autotune_compat", False):
        _orig_autotune = triton.autotune

        def _compat_autotune(*args, **kwargs):
            try:
                return _orig_autotune(*args, **kwargs)
            except TypeError:
                cleaned = {
                    k: v for k, v in kwargs.items()
                    if k not in _COMPAT_AUTOTUNE_DROP
                }
                if cleaned == kwargs:
                    # The TypeError was not caused by a known incompatible kwarg;
                    # let it propagate so a real bug is surfaced, not masked.
                    raise
                return _orig_autotune(*args, **cleaned)

        triton.autotune = _compat_autotune
        triton._flagscale_autotune_compat = True


class _TritonCompatFinder(importlib.abc.MetaPathFinder):
    """Intercept the first ``import triton`` to patch it right after it loads.

    Returns a spec whose loader delegates to the real triton loader and then
    calls ``_apply_patch``. Armed once; after triton is in ``sys.modules`` the
    import system never asks finders again, so the flag stays False permanently.
    """

    _armed = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "triton" or not self._armed:
            return None
        self._armed = False

        # Resolve the real triton spec via the standard path finder (does NOT
        # iterate sys.meta_path, so no re-entrancy into this finder).
        real_spec = importlib.machinery.PathFinder.find_spec("triton", path)
        if real_spec is None:
            return None

        real_loader = real_spec.loader
        if real_loader is None:
            return real_spec

        class _PatchingLoader(importlib.abc.Loader):
            def create_module(self, spec):
                if hasattr(real_loader, "create_module"):
                    return real_loader.create_module(spec)
                return None

            def exec_module(self, module):
                real_loader.exec_module(module)
                # triton is now fully in sys.modules; patch it before flag_gems
                # reads ``triton.autotune`` for its decorators.
                try:
                    _apply_patch()
                except Exception:
                    # Never let the compat shim turn a loadable triton into a
                    # failed import. If patching throws, the original triton is
                    # still usable; the gap will surface at flag_gems import.
                    pass

        real_spec.loader = _PatchingLoader()
        return real_spec


# Insert at the front so we see ``import triton`` before the cached/builtin
# finders. Pure registration -- no triton import happens here.
if not any(isinstance(f, _TritonCompatFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _TritonCompatFinder())
