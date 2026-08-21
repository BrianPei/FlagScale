#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Patch vllm-plugin-FL (v0.1.1+vllm0.13.0) for the Kunlunxin P800 image.

Upstream vllm-plugin-FL does not register the ``kunlunxin`` vendor against
vllm 0.13.0:

- ``VENDOR_DEVICE_MAP`` has no ``kunlunxin`` entry, so the ``PlatformFL``
  class body ``device_type = get_device_type("kunlunxin")`` raises
  ``ValueError`` before vLLM can resolve ``current_platform``.
- ``DeviceInfo.supported_device`` omits ``kunlunxin``.
- ``utils.py`` imports ``DeviceDetector`` from
  ``flag_gems.runtime.backend.device``, but the P800 image ships
  flag_gems 5.4.0dev which moved ``DeviceDetector`` to
  ``flag_gems.runtime.backend.device_finder`` (the ``device`` module is
  gone), so the import fails at module load.

flag_gems 5.4.0dev itself knows about kunlunxin
(``DeviceDetector().vendor_name == "kunlunxin"``, ``device_name="cuda"``);
this patch only bridges the gap in vllm-plugin-FL.

A second patch target (only present on the ``main`` ref, which carries the
kunlunxin vendor backend added by PR #268) gates the upstream
``patch_decode_attention`` in
``vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py``. That function
unconditionally replaces ``KunlunxinPagedAttention.forward_decode`` with
``xtorch_ops.prefill_attention(is_prefix_cache=True)`` to dodge a decode NaN
observed on layer 43+ of Qwen3.6-27B. Models that do not hit that NaN (e.g.
Qwen3-4B, 36 layers) must keep the native ``forward_decode`` -- forcing
``prefill_attention`` mis-maps the paged KV cache and garbles decode output.
``VLLM_FL_KLX_DISABLE_DECODE_PATCH=1`` skips the upstream patch so the
native path is exercised. Remove once upstream gates the fallback on a
runtime NaN check instead of applying it unconditionally.

Applied to the checked-out vllm-plugin-FL source before ``pip install``,
so the patch tracks this repo rather than a fork.

Idempotent: re-applying to already-patched source skips each replacement
(the target is the *unpatched* string, absent after the first pass).
"""

import sys
from pathlib import Path

UTILS = "vllm_fl/utils.py"
KUNLUNXIN_PATCH = "vllm_fl/dispatch/backends/vendor/kunlunxin/patch.py"

# (file, description, old, new) -- each `old` must occur exactly once in the
# fresh target file. utils.py entries always apply; the patch.py entry only
# applies on the main ref (kunlunxin vendor backend is absent on tag refs).
PATCHES = [
    (
        UTILS,
        "DeviceDetector device_finder fallback",
        "import flag_gems\nfrom flag_gems.runtime.backend.device import DeviceDetector\n",
        (
            "import flag_gems\n"
            "try:\n"
            "    from flag_gems.runtime.backend.device import DeviceDetector\n"
            "except (ImportError, FileNotFoundError):\n"
            "    from flag_gems.runtime.backend.device_finder import DeviceDetector\n"
        ),
    ),
    (
        UTILS,
        "kunlunxin VENDOR_DEVICE_MAP entry",
        '    "mthreads": {"device_type": "musa", "device_name": "musa"},\n}',
        (
            '    "mthreads": {"device_type": "musa", "device_name": "musa"},\n'
            '    # Registered backend: vendor/kunlunxin (P800, cuda-alike via torch_xmlir)\n'
            '    "kunlunxin": {"device_type": "cuda", "device_name": "cuda"},\n}'
        ),
    ),
    (
        UTILS,
        "kunlunxin in DeviceInfo.supported_device",
        'self.supported_device = ["nvidia", "ascend", "metax", "mthreads"]',
        'self.supported_device = ["nvidia", "ascend", "metax", "mthreads", "kunlunxin"]',
    ),
    (
        KUNLUNXIN_PATCH,
        "patch_decode_attention env gate (VLLM_FL_KLX_DISABLE_DECODE_PATCH)",
        (
            "    with is_prefix_cache=True provides correct results.\n"
            '    """\n'
            "    try:\n"
            "        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as attn_mod\n"
            "        import xtorch_ops\n"
        ),
        (
            "    with is_prefix_cache=True provides correct results.\n"
            '    """\n'
            "    # Gate (flagos-ai/FlagScale): this workaround unconditionally replaces\n"
            "    # forward_decode with xtorch_ops.prefill_attention(is_prefix_cache=True).\n"
            "    # It exists because decode_paged_attention NaNs on layer 43+ of\n"
            "    # Qwen3.6-27B. Models that do NOT hit that NaN (e.g. Qwen3-4B, 36\n"
            "    # layers) must keep the native forward_decode -- forcing\n"
            "    # prefill_attention here mis-maps the paged KV cache and garbles\n"
            "    # decode. VLLM_FL_KLX_DISABLE_DECODE_PATCH=1 skips this patch so the\n"
            "    # native path is exercised. Remove once upstream gates the fallback\n"
            "    # on a runtime NaN check.\n"
            "    import os\n"
            '    if os.environ.get("VLLM_FL_KLX_DISABLE_DECODE_PATCH") == "1":\n'
            "        logger.info(\n"
            '            "Skipped patch_decode_attention '
            '(VLLM_FL_KLX_DISABLE_DECODE_PATCH=1): "\n'
            '            "keeping native KunlunxinPagedAttention.forward_decode"\n'
            "        )\n"
            "        return\n"
            "    try:\n"
            "        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as attn_mod\n"
            "        import xtorch_ops\n"
        ),
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-plugin-FL checkout dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])

    utils = root / UTILS
    if not utils.is_file():
        print(f"error: {utils} not found (not a vllm-plugin-FL checkout?)",
              file=sys.stderr)
        return 1

    for relpath, desc, old, new in PATCHES:
        target = root / relpath
        if not target.is_file():
            # kunlunxin/patch.py only exists on the main ref (PR #268 added
            # the vendor backend); tag refs legitimately lack it.
            print(f"skip: {desc} ({relpath} not found; non-main ref?)")
            continue
        text = target.read_text()
        count = text.count(old)
        if count == 0:
            # Already patched, or upstream source changed shape.
            print(f"skip: {desc} (target not found; already patched?)")
            continue
        if count > 1:
            print(f"error: {desc} matched {count} times; ambiguous, refusing",
                  file=sys.stderr)
            return 1
        target.write_text(text.replace(old, new))
        print(f"applied: {desc} ({relpath})")

    print("patched vllm-plugin-FL for kunlunxin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
