#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Triton Ascend can compile the same precompiled launcher header from several
# vLLM workers at once. Serialize those compiler calls until upstream cache
# creation is process-safe.
compiler="${FLAGSCALE_TRITON_CXX:-/usr/bin/g++}"
lock_file="${FLAGSCALE_TRITON_CXX_LOCK:-/tmp/flagscale-triton-gxx.lock}"

if ! command -v flock >/dev/null 2>&1; then
    exec "$compiler" "$@"
fi

exec flock "$lock_file" "$compiler" "$@"
