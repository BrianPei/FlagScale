"""
Compatibility patches for platforms with outdated or missing dependencies.

This module must be imported before any other FlagScale modules to ensure
proper fallback behavior for optional dependencies.
"""
import sys
import warnings
from unittest.mock import MagicMock


def patch_flash_attn():
    """
    Mock flash_attn_2_cuda for platforms without FlashAttention support.

    TransformerEngine unconditionally imports flash_attn_2_cuda in some
    attention backends, even when FlashAttention is not available on the
    platform (e.g., Enflame, MUSA, MetaX, KunLunXin).
    """
    try:
        import flash_attn_2_cuda
    except (ImportError, ModuleNotFoundError):
        warnings.warn(
            "flash_attn_2_cuda not available - using mock implementation. "
            "FlashAttention features will be disabled.",
            RuntimeWarning,
            stacklevel=2
        )
        sys.modules['flash_attn_2_cuda'] = MagicMock()


def apply_all_patches():
    """Apply all compatibility patches at once."""
    patch_flash_attn()
