#!/usr/bin/env bash

set -euo pipefail

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CI_SETUP_DIR/set_env_common.sh"

echo "Setting up Enflame GCU environment"

ci_activate_python_environment

# Remove pre-installed packages from image to avoid import conflicts
echo "Removing pre-installed megatron-core and transformer-engine from image..."
pip uninstall -y megatron-core || echo "No pre-installed megatron-core found"
pip uninstall -y transformer-engine || echo "No pre-installed transformer-engine found"

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

# Patch TE-FL flash_attn detection for torch_gcu compatibility
python3 - <<'PY'
import sys
import os

try:
    import transformer_engine.pytorch.attention.dot_product_attention.backends as te_backends
    backends_file = te_backends.__file__

    with open(backends_file, 'r') as f:
        content = f.read()

    # Only patch if not already patched
    if 'torch_gcu' not in content and 'flash_attn_2_cuda' in content:
        # Find the flash_attn import section and add torch_gcu guard
        original = '''else:
        if torch.cuda.is_available() and get_device_compute_capability() >= (10, 0):'''

        patched = '''else:
        # Skip flash_attn in torch_gcu environment (Enflame GCU compatibility)
        try:
            import torch_gcu
            skip_flash_attn = True
        except ImportError:
            skip_flash_attn = False

        if skip_flash_attn:
            pass
        elif torch.cuda.is_available() and get_device_compute_capability() >= (10, 0):'''

        if original in content:
            content = content.replace(original, patched)
            with open(backends_file, 'w') as f:
                f.write(content)
            print(f"Patched TE-FL backends.py for torch_gcu: {backends_file}")
        else:
            print(f"::warning::TE-FL backends.py structure changed, patch skipped")
    else:
        print("TE-FL backends.py already patched or no patch needed")
except ImportError:
    print("::warning::TE-FL not installed yet, patch will be applied at runtime if needed")
except Exception as e:
    print(f"::warning::Failed to patch TE-FL backends.py: {e}")
PY

echo "Enflame GCU environment setup complete"
