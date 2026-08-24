#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${VLLM_PLUGINS:=fl}"
: "${VLLM_FL_PLATFORM:=enflame}"

export FLAGSCALE_HOME FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export VLLM_PLUGINS VLLM_FL_PLATFORM
