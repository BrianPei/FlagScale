#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# FlagScale MUSA Environment Variables
# =============================================================================

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${FLAGSCALE_CONDA:=/opt/conda}"
: "${UV_PROJECT_ENVIRONMENT:=$FLAGSCALE_HOME/venv}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${MPI_HOME:=/usr/local/mpi}"
: "${MUSA_HOME:=/usr/local/musa}"

: "${UV_HTTP_TIMEOUT:=500}"
: "${UV_INDEX_STRATEGY:=unsafe-best-match}"
: "${UV_LINK_MODE:=copy}"

export FLAGSCALE_HOME FLAGSCALE_CONDA FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export UV_PROJECT_ENVIRONMENT MPI_HOME MUSA_HOME
export UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

export PATH="$UV_PROJECT_ENVIRONMENT/bin:$FLAGSCALE_CONDA/bin:$HOME/.local/bin:$MPI_HOME/bin:$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MUSA_HOME/lib64:$MPI_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:$LD_LIBRARY_PATH"
