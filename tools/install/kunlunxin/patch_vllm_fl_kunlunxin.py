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

Applied to the checked-out vllm-plugin-FL source before ``pip install``,
so the patch tracks this repo rather than a fork. Remove it once
vllm-plugin-FL upstream registers kunlunxin against vllm 0.13.0.

Idempotent: re-applying to already-patched source skips each replacement
(the target is the *unpatched* string, absent after the first pass).
"""

import sys
from pathlib import Path

UTILS = "vllm_fl/utils.py"

# (description, old, new) -- each `old` must occur exactly once in fresh source.
PATCHES = [
    (
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
        "kunlunxin VENDOR_DEVICE_MAP entry",
        '    "mthreads": {"device_type": "musa", "device_name": "musa"},\n}',
        (
            '    "mthreads": {"device_type": "musa", "device_name": "musa"},\n'
            '    # Registered backend: vendor/kunlunxin (P800, cuda-alike via torch_xmlir)\n'
            '    "kunlunxin": {"device_type": "cuda", "device_name": "cuda"},\n}'
        ),
    ),
    (
        "kunlunxin in DeviceInfo.supported_device",
        'self.supported_device = ["nvidia", "ascend", "metax", "mthreads"]',
        'self.supported_device = ["nvidia", "ascend", "metax", "mthreads", "kunlunxin"]',
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

    text = utils.read_text()
    for desc, old, new in PATCHES:
        count = text.count(old)
        if count == 0:
            # Already patched, or upstream source changed shape.
            print(f"skip: {desc} (target not found; already patched?)")
            continue
        if count > 1:
            print(f"error: {desc} matched {count} times; ambiguous, refusing",
                  file=sys.stderr)
            return 1
        text = text.replace(old, new)
        print(f"applied: {desc}")

    utils.write_text(text)
    print(f"patched {utils}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
