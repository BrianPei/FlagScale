"""Early compatibility hooks for optional accelerator dependencies."""

import os
import sys
import types


def _flash_attention_is_disabled() -> bool:
    return os.environ.get("TE_FL_SKIP_CUDA") == "1" or os.environ.get("NVTE_FLASH_ATTN") == "0"


def _patch_te_fl_backends_flash_attn_import():
    """Prevent TE-FL from importing flash-attn when NVTE_FLASH_ATTN=0.

    TE-FL backends.py:108 checks fa_utils.is_installed but ignores _NVTE_FLASH_ATTN,
    causing ImportError on non-CUDA platforms even when flash-attn is explicitly disabled.

    This patch intercepts the import chain and blocks flash_attn modules from being loaded
    when the disable flag is set, before TE-FL's backend registry attempts to import them.
    """
    if not _flash_attention_is_disabled():
        return

    # Block flash_attn package family
    for module_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn_2_cuda",
        "flash_attn_3_cuda",
    ):
        if module_name not in sys.modules:
            stub = types.ModuleType(module_name)
            stub.__file__ = f"<{module_name} stub - disabled by NVTE_FLASH_ATTN=0>"
            stub.__path__ = []

            def _make_unavailable_attr(mod_name: str):
                def _unavailable_attr(name: str):
                    if name in ("__file__", "__path__", "__spec__", "__loader__", "__package__"):
                        raise AttributeError(f"module '{mod_name}' has no attribute '{name}'")

                    # Immediately raise on attribute access to block "from X import Y"
                    raise RuntimeError(
                        f"{mod_name}.{name} is unavailable because FlashAttention "
                        "is disabled for this CI runtime (NVTE_FLASH_ATTN=0)"
                    )
                return _unavailable_attr

            stub.__getattr__ = _make_unavailable_attr(module_name)
            sys.modules[module_name] = stub


_patch_te_fl_backends_flash_attn_import()
