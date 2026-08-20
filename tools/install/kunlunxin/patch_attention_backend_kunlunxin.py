#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Patch vllm_fl platform.py (v0.1.1+vllm0.13.0) so the Kunlunxin P800 uses a
working attention backend: the FlagGems AttentionFLBackend.

vllm_fl's ``PlatformFL.get_attn_backend_cls`` ignores the ``selected_backend``
chosen by vLLM's attention selector (e.g. the ``attention_backend:
TORCH_SDPA`` set in the case yaml) and returns whatever the dispatch
``attention_backend`` op resolves to. On v0.1.1 the dispatch path is broken
for the P800 in two ways:

1. The FlagGems backend's ``attention_backend`` op returns
   ``AttentionBackendEnum.TRITON_ATTN.get_path()``. Upstream PR #34 (merged on
   main, *after* v0.1.1 was cut) is what wires this up and, crucially, calls
   ``custom_attention.register_attention()`` to *override* ``TRITON_ATTN`` to
   point at ``vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend``.
   On v0.1.1 ``register_attention()`` is **never called**, so ``TRITON_ATTN.get_path()``
   returns vllm's own ``TritonAttentionBackend`` -- whose triton kernel is not
   XPU-compatible on the P800.

2. If the FlagGems ``attention_backend`` op is filtered out (``use_flaggems_op``
   blacklist) the dispatch falls back to the reference backend, whose
   ``attention_backend`` returns ``FLASH_ATTN`` -- vllm's NVIDIA
   ``FlashAttentionBackend``. The P800 has no NVIDIA flash_attn, so
   ``is_flash_attn_varlen_func_available()`` is False, ``reshape_and_cache_flash``
   is never imported, and ``FlashAttentionBackend`` raises
   ``NameError: name 'reshape_and_cache_flash' is not defined`` at forward
   (vllm/v1/attention/backends/flash_attn.py).

Neither path runs on the P800. The working backend already exists in the
v0.1.1 source -- ``AttentionFLBackend`` (impl/attention.py), a v1
``AttentionBackend`` subclass that supports DECODER and whose forward uses
``flag_gems.flash_attn_varlen_func`` + ``flag_gems.reshape_and_cache_flash``.
FlagGems is the unified multi-chip operator library with Kunlunxin support,
and the wider FlagOS stack runs Kunlunxin on flash attention (see
TransformerEngine-FL PRs #27/#29/#30). Under ``enforce_eager=True`` vLLM calls
the selected backend's forward directly, so the backend choice is decisive.

Force ``AttentionFLBackend`` for kunlunxin -- this is the v0.1.1 patch
equivalent of upstream PR #34's "enable FlagGems attention backend" intent.

NOTE: do NOT use ``AttentionBackendEnum.TORCH_SDPA.get_path()``. In vllm
0.13.0 ``TORCH_SDPA = ""`` is a ViT-only placeholder tag with no impl class;
``get_path()`` raises ``ValueError: Backend TORCH_SDPA must be registered
before use``. TORCH_SDPA has no decoder backend class in 0.13.0.

Applied to the checked-out vllm-plugin-FL source before ``pip install``, so
the patch tracks this repo rather than a fork. Remove once v0.1.1+'s dispatch
registers/overrides a working Kunlunxin attention_backend (i.e. once the
PR #34 wiring lands in a tagged release we pin to).

Idempotent: re-applying to already-patched source skips the replacement (the
target is the *unpatched* string, absent after the first pass).
"""

import sys
from pathlib import Path

PLATFORM = "vllm_fl/platform.py"

# (description, old, new) -- `old` must occur exactly once in fresh source.
PATCHES = [
    (
        "force FlagGems AttentionFLBackend for kunlunxin (not TORCH_SDPA)",
        (
            "        backend_path = call_op(\"attention_backend\","
            " use_mla=use_mla, use_sparse=use_sparse)\n"
            "\n"
            "        logger.info_once(\n"
        ),
        (
            "        backend_path = call_op(\"attention_backend\","
            " use_mla=use_mla, use_sparse=use_sparse)\n"
            "\n"
            "        # Kunlunxin P800: on v0.1.1 the dispatch attention_backend"
            " resolves to TRITON_ATTN whose get_path() returns vllm's own\n"
            "        # TritonAttentionBackend (custom_attention.register_attention"
            " -- the TRITON_ATTN -> AttentionFLBackend override -- is never\n"
            "        # called on v0.1.1, landing only in upstream PR #34), or"
            " falls back to reference FLASH_ATTN (vllm NVIDIA\n"
            "        # flash_attn => reshape_and_cache_flash NameError in"
            " flash_attn.py). Neither runs on the P800. Force the FlagGems\n"
            "        # AttentionFLBackend (impl/attention.py), whose forward"
            " uses flag_gems.flash_attn_varlen_func +\n"
            "        # flag_gems.reshape_and_cache_flash (FlagGems: unified"
            " multi-chip lib with Kunlunxin support). v0.1.1 patch\n"
            "        # equivalent of upstream PR #34 'enable FlagGems attention"
            " backend'. NOTE: do not use TORCH_SDPA.get_path() -- in vllm\n"
            "        # 0.13.0 TORCH_SDPA is a ViT-only empty placeholder and"
            " get_path() raises ValueError. Remove once v0.1.1+ dispatch\n"
            "        # registers/overrides a working Kunlunxin attention_backend.\n"
            "        if cls.vendor_name == \"kunlunxin\":\n"
            "            backend_path = (\"vllm_fl.dispatch.backends.flaggems\"\n"
            "                             \".impl.attention.AttentionFLBackend\")\n"
            "\n"
            "        logger.info_once(\n"
        ),
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-plugin-FL checkout dir>",
              file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    platform = root / PLATFORM
    if not platform.is_file():
        print(f"error: {platform} not found (not a vllm-plugin-FL checkout?)",
              file=sys.stderr)
        return 1

    text = platform.read_text()
    for desc, old, new in PATCHES:
        count = text.count(old)
        if count == 0:
            print(f"skip: {desc} (target not found; already patched?)")
            continue
        if count > 1:
            print(f"error: {desc} matched {count} times; ambiguous, refusing",
                  file=sys.stderr)
            return 1
        text = text.replace(old, new)
        print(f"applied: {desc}")

    platform.write_text(text)
    print(f"patched {platform}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
