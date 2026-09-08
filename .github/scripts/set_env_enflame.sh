#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up Enflame GCU environment"

ci_activate_python_environment

if ! command -v efml-smi >/dev/null 2>&1; then
  echo "::warning::efml-smi not found, skipping Enflame validation"
else
  efml-smi || echo "::warning::efml-smi returned non-zero"
fi

if [ -n "${CI_NPROC_PER_NODE:-}" ]; then
  python3 - <<'PY'
import os
import sys

try:
    import torch
    import torch_gcu
    available = torch.gcu.device_count()
    required = int(os.environ.get("CI_NPROC_PER_NODE", "0"))
    print(f"Enflame GCU devices available: {available}, required: {required}")
    if available < required:
        print(f"::error::Not enough devices: available={available}, required={required}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"::warning::Could not verify device count: {e}")
PY
fi

echo "Enflame GCU environment setup complete"
