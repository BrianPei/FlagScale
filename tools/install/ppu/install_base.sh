#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"

# Fail before pip can replace the PPU torch fork with a public CPU/CUDA wheel.
python -c 'import torch; print("Vendor torch:", torch.__version__, torch.__file__)'
command -v git
command -v c++
command -v make
python -m pip --version
mkdir -p /opt/flagscale/ppu
python - <<'PY' > /opt/flagscale/ppu/vendor-constraints.txt
import importlib.metadata as md
for dist in md.distributions():
    name = dist.metadata.get("Name", "")
    if name.lower().replace("_", "-").startswith(("torch", "triton", "flagcx")):
        print(f"{name}=={dist.version}")
PY
python -m pip install -c /opt/flagscale/ppu/vendor-constraints.txt \
    -r "$script_dir/../../../requirements/ppu/train.txt"
