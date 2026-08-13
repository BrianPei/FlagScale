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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from flagscale.train.megatron.training.tokenizer.tokenizer import _QwenTokenizerFS


@pytest.mark.parametrize(
    ("disable_fast", "expected_use_fast"),
    [(False, True), (True, False)],
)
def test_qwen_tokenizer_respects_hf_fast_setting(disable_fast, expected_use_fast):
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [0]
    args = SimpleNamespace(tokenizer_hf_no_use_fast=disable_fast)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
        _QwenTokenizerFS("/tmp/qwen-tokenizer", args)

    load.assert_called_once_with(
        "/tmp/qwen-tokenizer",
        trust_remote_code=True,
        use_fast=expected_use_fast,
    )


def test_qwen_tokenizer_defaults_to_fast_when_argument_is_missing():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [0]

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
        _QwenTokenizerFS("/tmp/qwen-tokenizer", SimpleNamespace())

    assert load.call_args.kwargs["use_fast"] is True
