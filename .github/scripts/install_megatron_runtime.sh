#!/usr/bin/env bash

set -euo pipefail

: "${MEGATRON_INSTALL_DIR:?MEGATRON_INSTALL_DIR is required}"

python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
if [ ! -x "$python_bin" ]; then
  echo "::error::Python executable not found: $python_bin" >&2
  exit 1
fi

if [ ! -d "$MEGATRON_INSTALL_DIR" ]; then
  echo "::error::Megatron-LM-FL install directory is missing: $MEGATRON_INSTALL_DIR" >&2
  exit 1
fi

# CRITICAL: Remove any site-packages megatron-core to prevent conflicts
echo "Removing any site-packages megatron-core to ensure PYTHONPATH priority..."
"$python_bin" -m pip uninstall -y megatron-core >/dev/null 2>&1 || true

# Set PYTHONPATH with absolute priority
export PYTHONPATH="${MEGATRON_INSTALL_DIR}:${PYTHONPATH:-}"
# Disable user site-packages to prevent interference
export PYTHONNOUSERSITE=1

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "PYTHONPATH=${PYTHONPATH}" >> "$GITHUB_ENV"
  echo "PYTHONNOUSERSITE=1" >> "$GITHUB_ENV"
fi

"$python_bin" - <<'PY'
import sys
import os

megatron_path = os.environ.get("MEGATRON_INSTALL_DIR")
if megatron_path and megatron_path not in sys.path:
    sys.path.insert(0, megatron_path)

try:
    import megatron
    import megatron.core
    print(f"Megatron-LM-FL Python: {sys.executable}")
    print(f"Megatron-LM-FL import passed: {megatron.__file__}")
    print(f"Megatron-Core version: {getattr(megatron.core, '__version__', 'unknown')}")
    print(f"Megatron-Core path: {megatron.core.__file__}")

    # Verify critical modules exist
    try:
        from megatron.core.transformer.cuda_graph_config import CudaGraphConfig
        print(f"✓ cuda_graph_config module available")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"⚠ Warning: cuda_graph_config not available (expected for Megatron-Core < 0.18): {e}")
except ImportError as e:
    print(f"::error::Failed to import Megatron-LM-FL: {e}", file=sys.stderr)
    sys.exit(1)
PY
