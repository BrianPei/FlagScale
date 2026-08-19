#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Patch vllm_fl platform.py (v0.1.1+vllm0.13.0) so the Kunlunxin P800 uses a
working attention backend.

vllm_fl's ``PlatformFL.get_attn_backend_cls`` ignores the ``selected_backend``
chosen by vLLM's attention selector (e.g. the ``attention_backend:
TORCH_SDPA`` set in the case yaml) and instead resolves the backend through
the dispatch ``attention_backend`` op. On a platform with no kunlunxin vendor
backend registered, dispatch falls back to its built-in implementations:

- flagos (FlagGems) ``attention_backend`` returns ``TRITON_ATTN`` -- the
  triton attention kernel is not XPU-compatible on the P800.
- reference (PyTorch) ``attention_backend`` returns ``FLASH_ATTN`` -- the P800
  has no NVIDIA flash_attn, so ``is_flash_attn_varlen_func_available()`` is
  False, ``reshape_and_cache_flash`` is never imported, and
  ``FlashAttentionBackend`` raises ``NameError: name 'reshape_and_cache_flash'
  is not defined`` at forward (vllm/v1/attention/backends/flash_attn.py).

Neither dispatch default runs on the P800. Under ``enforce_eager=True`` vLLM
calls the selected backend's forward directly, so the backend choice is
decisive. Force ``TORCH_SDPA`` for kunlunxin -- torch scaled_dot_product_attention
is a core PyTorch op implemented by torch_xmlir, and is the case yaml's
explicit intent.

Applied to the checked-out vllm-plugin-FL source before ``pip install``, so
the patch tracks this repo rather than a fork. Remove once vllm_fl dispatch
has a kunlunxin attention_backend implementation returning a working backend.

Idempotent: re-applying to already-patched source skips the replacement (the
target is the *unpatched* string, absent after the first pass).
"""

import sys
from pathlib import Path

PLATFORM = "vllm_fl/platform.py"

# (description, old, new) -- `old` must occur exactly once in fresh source.
PATCHES = [
    (
        "force TORCH_SDPA attention backend for kunlunxin",
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
            "        # Kunlunxin P800: dispatch resolves attention_backend to"
            " TRITON_ATTN (flagos) or FLASH_ATTN (reference), neither of which\n"
            "        # runs on the P800 -- no NVIDIA flash_attn =>"
            " is_flash_attn_varlen_func_available() is False =>\n"
            "        # reshape_and_cache_flash NameError in flash_attn.py; the"
            " triton-attn kernel is not XPU-compatible. Force\n"
            "        # torch SDPA (a core PyTorch op implemented by"
            " torch_xmlir), which is the case yaml's explicit intent via\n"
            "        # `attention_backend: TORCH_SDPA`. Remove once vllm_fl"
            " dispatch has a kunlunxin attention_backend impl.\n"
            "        if cls.vendor_name == \"kunlunxin\":\n"
            "            backend_path = AttentionBackendEnum.TORCH_SDPA.get_path()\n"
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
