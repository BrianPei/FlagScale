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

export PYTHONPATH="${MEGATRON_INSTALL_DIR}:${PYTHONPATH:-}"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "PYTHONPATH=${PYTHONPATH}" >> "$GITHUB_ENV"
fi

"$python_bin" - <<'PY'
import sys
import os

megatron_path = os.environ.get("MEGATRON_INSTALL_DIR")
if megatron_path and megatron_path not in sys.path:
    sys.path.insert(0, megatron_path)

try:
    import megatron
    print(f"Megatron-LM-FL Python: {sys.executable}")
    print(f"Megatron-LM-FL import passed: {megatron.__file__}")
except ImportError as e:
    print(f"::error::Failed to import Megatron-LM-FL: {e}", file=sys.stderr)
    sys.exit(1)
PY
