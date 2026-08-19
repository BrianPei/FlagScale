#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Patch vllm_fl flagcx.py (v0.1.1+vllm0.13.0) for the FlagCX wrapper shipped
in the Kunlunxin P800 base image.

vllm_fl's flagcx.py was written against an older FlagCX python wrapper API:

- ``flagcxGetUniqueId()`` was assumed to return a ctypes *pointer* whose
  ``.contents`` is the ``flagcxUniqueId``. The P800's FlagCX wrapper
  (``/opt/FlagCX/plugin/interservice/flagcx_wrapper.py``) returns the
  ``flagcxUniqueId`` object directly (a ``ctypes.Structure`` with an
  ``internal`` field), so ``.contents`` raises
  ``AttributeError: 'flagcxUniqueId' object has no attribute 'contents'``.

- ``flagcxCommInitRank`` was called as
  ``flagcxCommInitRank(world_size, ctypes.byref(unique_id), rank)``. The P800
  wrapper signature is
  ``flagcxCommInitRank(world_size, unique_id: flagcxUniqueId, rank)`` -- it
  takes the object and applies ``byref`` internally; passing ``byref`` again
  gives it a ``CArgObject`` it cannot handle.

All other call sites (``flagcxAllReduce``, ``adaptor_stream_copy``,
``flagcxGroupStart/End`` ...) already pass ``self.comm`` / ``flagcx_stream``
as objects, matching the wrapper method signatures, so they need no change.

Applied to the checked-out vllm-plugin-FL source before ``pip install``, so
the patch tracks this repo rather than a fork. Remove it once vllm_fl's
flagcx.py matches the FlagCX wrapper API shipped in the base image.

Idempotent: re-applying to already-patched source skips each replacement
(the target is the *unpatched* string, absent after the first pass).
"""

import sys
from pathlib import Path

FLAGCX = "vllm_fl/distributed/device_communicators/flagcx.py"

# (description, old, new) -- each `old` must occur exactly once in fresh source.
PATCHES = [
    (
        "flagcxGetUniqueId returns object not pointer (drop .contents)",
        "self.unique_id = self.flagcx.flagcxGetUniqueId().contents",
        "self.unique_id = self.flagcx.flagcxGetUniqueId()",
    ),
    (
        "flagcxCommInitRank takes unique_id object not byref",
        "ctypes.byref(self.unique_id), self.rank)",
        "self.unique_id, self.rank)",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-plugin-FL checkout dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    flagcx = root / FLAGCX
    if not flagcx.is_file():
        print(f"error: {flagcx} not found (not a vllm-plugin-FL checkout?)",
              file=sys.stderr)
        return 1

    text = flagcx.read_text()
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

    flagcx.write_text(text)
    print(f"patched {flagcx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
