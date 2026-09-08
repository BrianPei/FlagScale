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

        # Set __file__ to a sentinel path to prevent __getattr__ from intercepting it
        # and to avoid triggering "built-in module" errors in inspect.getsourcefile()
        module.__file__ = "<flash_attn_2_cuda stub>"

        def unavailable(name: str):
            # Whitelist special attributes that should not raise errors
            if name in ("__file__", "__path__", "__spec__", "__loader__", "__package__"):
                raise AttributeError(f"module 'flash_attn_2_cuda' has no attribute '{name}'")

            def fail(*args, **kwargs):
                raise RuntimeError(
                    f"flash_attn_2_cuda.{name} is unavailable because CUDA "
                    "FlashAttention is disabled for this CI runtime"
                )

            return fail

        module.__getattr__ = unavailable
        sys.modules[module.__name__] = module
