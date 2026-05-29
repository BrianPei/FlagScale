from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner import utils as auto_utils


def _strategy(**overrides):
    strategy = {
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
    }
    strategy.update(overrides)
    return strategy


def test_divisible_true_false_and_zero_divisor():
    assert auto_utils.divisible(16, 4) is True
    assert auto_utils.divisible(16, 5) is False
    with pytest.raises(ZeroDivisionError):
        auto_utils.divisible(1, 0)


def test_beside_returns_items_matching_all_non_ignored_dims():
    current = _strategy(micro_batch_size=8)
    matching_a = _strategy(micro_batch_size=1)
    matching_b = _strategy(micro_batch_size=2)
    different_tp = _strategy(micro_batch_size=4, tensor_model_parallel_size=2)

    assert auto_utils.beside(
        ["micro_batch_size"], current, [matching_a, matching_b, different_tp]
    ) == [
        matching_a,
        matching_b,
    ]


def test_beside_ignores_non_strategy_dimensions():
    current = _strategy(micro_batch_size=8)
    history = [_strategy(micro_batch_size=4, arbitrary_runtime_field="ignored")]

    assert auto_utils.beside(["micro_batch_size"], current, history) == history


def test_sort_by_memory_prefers_memory_saving_dimensions():
    no_recompute = _strategy(use_recompute=False)
    recompute = _strategy(use_recompute=True)
    lower_tp = _strategy(tensor_model_parallel_size=1)
    higher_tp = _strategy(tensor_model_parallel_size=4)

    assert auto_utils.sort_by_memory(recompute) < auto_utils.sort_by_memory(
        no_recompute
    )
    assert auto_utils.sort_by_memory(higher_tp) < auto_utils.sort_by_memory(lower_tp)


def test_sort_by_memory_handles_optional_booleans():
    strategy = _strategy(sequence_parallel=None, use_distributed_optimizer=None)

    key = auto_utils.sort_by_memory(strategy)

    assert key[2] == -float("inf")
    assert key[-1] == -float("inf")


def test_sort_by_memory_model_uses_calibrated_value():
    assert auto_utils.sort_by_memory_model({"memory_model": 12.5}) == 12.5


def test_sort_by_performance_prefers_larger_data_parallel_and_smaller_recompute_layers():
    low_dp = _strategy(data_parallel_size=1, recompute_num_layers=4)
    high_dp = _strategy(data_parallel_size=4, recompute_num_layers=1)

    assert auto_utils.sort_by_performance(high_dp) < auto_utils.sort_by_performance(
        low_dp
    )


@pytest.mark.parametrize(
    ("strategy1", "strategy2", "expected"),
    [
        (_strategy(use_recompute=False), _strategy(use_recompute=True), True),
        (
            _strategy(
                use_recompute=True, recompute_method="block", recompute_num_layers=2
            ),
            _strategy(
                use_recompute=True, recompute_method="block", recompute_num_layers=4
            ),
            True,
        ),
        (
            _strategy(
                use_recompute=True, recompute_method="block", recompute_num_layers=5
            ),
            _strategy(
                use_recompute=True, recompute_method="block", recompute_num_layers=4
            ),
            False,
        ),
        (
            _strategy(
                use_recompute=True,
                recompute_method="uniform",
                recompute_granularity="selective",
            ),
            _strategy(
                use_recompute=True,
                recompute_method="uniform",
                recompute_granularity="full",
            ),
            True,
        ),
        (
            _strategy(
                use_recompute=True,
                recompute_method="uniform",
                recompute_granularity="full",
                recompute_num_layers=2,
            ),
            _strategy(
                use_recompute=True,
                recompute_method="uniform",
                recompute_granularity="full",
                recompute_num_layers=2,
            ),
            True,
        ),
    ],
)
def test_compare_by_recompute_branches(strategy1, strategy2, expected):
    assert auto_utils.compare_by_recompute(strategy1, strategy2) is expected


def test_convert_config_to_megatron_args_derives_standard_fields(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "megatron.core.tokenizers.utils.build_tokenizer",
        SimpleNamespace(
            vocab_size_with_padding=lambda vocab_size, args: vocab_size + 7
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "megatron.training.arguments",
        SimpleNamespace(moe_freq_type=lambda value: value),
    )
    cfg = OmegaConf.create(
        {
            "train": {
                "model": {
                    "hidden_size": 16,
                    "num_attention_heads": 4,
                    "num_layers": 8,
                    "seq_length": 32,
                    "swiglu": True,
                    "multiple_of": 8,
                    "hidden_dim_multiplier": 1.0,
                },
                "system": {"use_flash_attn": True},
                "data": {"tokenizer": {"vocab_size": 100}},
            }
        }
    )
    strategy = _strategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=4,
        expert_model_parallel_size=1,
        use_distributed_optimizer=True,
        micro_batch_size=2,
        sequence_parallel=True,
        context_parallel_size=1,
        num_layers_per_virtual_pipeline_stage=2,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )

    args = auto_utils.convert_config_to_megatron_args(cfg, strategy)

    assert args.kv_channels == 4
    assert args.num_query_groups == 4
    assert args.ffn_hidden_size == 48
    assert args.padded_vocab_size == 107
    assert args.world_size == 16
    assert args.virtual_pipeline_model_parallel_size == 2
