"""Early compatibility hooks for optional accelerator dependencies."""

import os
import sys
import types


def _flash_attention_is_disabled() -> bool:
    return os.environ.get("TE_FL_SKIP_CUDA") == "1" or os.environ.get("NVTE_FLASH_ATTN") == "0"


if _flash_attention_is_disabled():
    try:
        import flash_attn_2_cuda  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # Some TE-FL releases import this optional CUDA extension while loading
        # their backend registry, before the caller can select a vendor backend.
        module = types.ModuleType("flash_attn_2_cuda")

        def unavailable(name: str):
            def fail(*args, **kwargs):
                raise RuntimeError(
                    f"flash_attn_2_cuda.{name} is unavailable because CUDA "
                    "FlashAttention is disabled for this CI runtime"
                )

            return fail

        module.__getattr__ = unavailable
        sys.modules[module.__name__] = module
