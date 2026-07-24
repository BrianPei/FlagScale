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

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).parents[4] / "flagscale" / "train"))
pytest.importorskip("megatron.core")

training_utils = importlib.import_module("megatron.training.utils")


def test_mock_data_does_not_parse_real_data_paths(monkeypatch):
    args = SimpleNamespace(mock_data=True, data_path=[])

    def fail_if_called(_):
        pytest.fail("mock data must not parse a real-data blend")

    monkeypatch.setattr(training_utils, "get_blend_from_list", fail_if_called)

    assert training_utils.get_blend_and_blend_per_split(args) == (None, None)
