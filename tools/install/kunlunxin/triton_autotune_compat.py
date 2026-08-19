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
1. ``triton.language.math``: PEP 562 ``__getattr__`` returns a real, existing
   intrinsic (``sin``) as a stand-in for any missing one (``asin``, ``acos``,
   ``atan``, ...), so module-level attribute access in flag_gems ops does not
   raise AND triton's AST dependency analyser (run by ``pointwise_dynamic``'s
   ``cache_key`` at import) accepts it -- a bare dummy callable makes that
   analyser raise ``Unsupported function referenced`` and kills the import. The
   stand-in is never dispatched (``USE_FLAGGEMS=false``), so its wrong semantics
   are inert.
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

A second finder, ``_FlagGemsCompatFinder``, re-runs ``_apply_patch`` before
every fresh ``import flag_gems``. This covers the ``all`` image, where
``import torch`` pulls in TE-FL which imports flag_gems (and thus triton)
while torch is still half-initialised -- the triton finder's single shot at
``_apply_patch`` then fails in that context, so the flag_gems finder defers
the flag_gems import and retries once torch is fully up.

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


def _apply_patch():
    """Patch the already-imported ``triton`` package. Idempotent.

    Called from the meta_path finder's loader right after real triton has loaded,
    so triton and its submodules are guaranteed importable here without
    retriggering the finder (``sys.modules["triton"]`` is already populated).
    """
    import triton
    import triton.language.math as _tl_math

    # --- triton.language.math: stand-in for missing intrinsics -------------
    if not getattr(_tl_math, "_flagscale_math_compat", False):
        _orig_math_getattr = getattr(_tl_math, "__getattr__", None)
        # flag_gems ops reference intrinsics the P800 runtime triton lacks
        # (e.g. asin). A bare dummy callable breaks triton's AST dependency
        # analyser (jit dependencies_finder, run by pointwise_dynamic's
        # cache_key at import): it raises "Unsupported function referenced"
        # and the whole flag_gems import dies. Return a real, existing
        # intrinsic (sin) instead -- the analyser accepts known intrinsics.
        # It is never dispatched: USE_FLAGGEMS=false keeps flag_gems ops off
        # the dispatch path, so the wrong-semantics stand-in is inert.
        _standin = _tl_math.sin

        def _math_getattr(name):
            if _orig_math_getattr is not None:
                try:
                    return _orig_math_getattr(name)
                except AttributeError:
                    pass
            return _standin

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


class _FlagGemsCompatFinder(importlib.abc.MetaPathFinder):
    """Re-run ``_apply_patch`` right before every fresh ``import flag_gems``.

    The triton finder above only catches the *first* ``import triton``. In the
    ``all`` image, ``import torch`` triggers TE-FL to ``import flag_gems`` (and
    thus triton) while torch is still half-initialised; ``_apply_patch``'s
    ``import triton.language.math`` then fails and is swallowed by the triton
    finder's ``except: pass``, leaving ``triton.language.math`` without the
    ``__getattr__`` shim. TE-FL catches the flag_gems failure, but triton is now
    cached unpatched, so the later probe ``import flag_gems`` dies on
    ``arcsin``'s ``tl_extra_shim.asin``.

    This finder re-runs ``_apply_patch`` (idempotent) immediately before
    flag_gems executes. If triton.language.math still cannot load, it defers
    the flag_gems import by raising ImportError so the caller catches it and
    the next attempt -- by which time torch is fully up -- patches cleanly.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "flag_gems":
            return None
        real_spec = importlib.machinery.PathFinder.find_spec("flag_gems", path)
        if real_spec is None or real_spec.loader is None:
            return real_spec
        real_loader = real_spec.loader

        class _PatchingLoader(importlib.abc.Loader):
            def create_module(self, spec):
                if hasattr(real_loader, "create_module"):
                    return real_loader.create_module(spec)
                return None

            def exec_module(self, module):
                try:
                    _apply_patch()
                except Exception:
                    # triton.language.math not importable yet (torch still
                    # half-initialised inside TE-FL's flag_gems import). Defer
                    # this import; the caller (TE-FL) catches it and we retry
                    # on the next flag_gems import, once torch is fully up.
                    raise ImportError(
                        "flagscale triton compat patch pending; "
                        "deferring flag_gems import"
                    ) from None
                real_loader.exec_module(module)

        real_spec.loader = _PatchingLoader()
        return real_spec


# Insert at the front so we see ``import triton`` / ``import flag_gems`` before
# the cached/builtin finders. Pure registration -- no triton import happens here.
for _finder in (_TritonCompatFinder, _FlagGemsCompatFinder):
    if not any(isinstance(f, _finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _finder())
