import copy

import pytest

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner.prune import history as history_prune


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
        "acc_step": 4,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "performance": None,
        "max_mem": None,
        "pruned": False,
    }
    strategy.update(overrides)
    return strategy


def test_prune_by_micro_batch_size_uses_larger_success_for_performance():
    current = _strategy(micro_batch_size=2, performance=None, max_mem=None)
    hist = [_strategy(micro_batch_size=4, performance=123.4, max_mem=2048)]

    assert history_prune.prune_by_micro_batch_size(None, current, hist) is True
    assert current["pruned"] is True
    assert current["performance"] == 123.4
    assert current["max_mem"] == 2048


def test_prune_by_micro_batch_size_uses_smaller_oom_for_memory():
    current = _strategy(micro_batch_size=4)
    hist = [_strategy(micro_batch_size=2, max_mem="OOM", performance=None)]

    assert history_prune.prune_by_micro_batch_size(None, current, hist) is True
    assert current["pruned"] is True
    assert current["max_mem"] == "OOM"
    assert current["performance"] is None


def test_prune_by_micro_batch_size_no_matching_history_keeps_strategy():
    current = _strategy(micro_batch_size=4, tensor_model_parallel_size=2)
    hist = [_strategy(micro_batch_size=2, max_mem="OOM", tensor_model_parallel_size=1)]

    assert history_prune.prune_by_micro_batch_size(None, current, hist) is False
    assert current["pruned"] is False


def test_prune_by_recompute_prunes_recompute_when_no_recompute_succeeded():
    current = _strategy(
        use_recompute=True,
        recompute_method="uniform",
        recompute_granularity="full",
        recompute_num_layers=1,
    )
    hist = [_strategy(performance=88.0, max_mem=1000)]

    assert history_prune.prune_by_recompute(None, current, hist) is True
    assert current["performance"] == 88.0
    assert current["max_mem"] == 1000


def test_prune_by_recompute_prunes_no_recompute_when_recompute_oomed():
    current = _strategy(use_recompute=False)
    hist = [
        _strategy(
            use_recompute=True,
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=1,
            max_mem="OOM",
        )
    ]

    assert history_prune.prune_by_recompute(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_prune_by_recompute_uniform_larger_num_layers_oom():
    current = _strategy(
        use_recompute=True,
        recompute_method="uniform",
        recompute_granularity="full",
        recompute_num_layers=4,
    )
    hist = [
        _strategy(
            use_recompute=True,
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=2,
            max_mem="OOM",
        )
    ]

    assert history_prune.prune_by_recompute(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_prune_by_sequence_parallel_reuses_sp_success_for_no_sp_performance():
    current = _strategy(sequence_parallel=False, tensor_model_parallel_size=2)
    hist = [
        _strategy(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            performance=50.0,
            max_mem=1200,
        )
    ]

    assert history_prune.prune_by_sequence_parallel(None, current, hist) is True
    assert current["performance"] == 50.0
    assert current["max_mem"] == 1200


def test_prune_by_sequence_parallel_reuses_sp_oom_for_no_sp_memory():
    current = _strategy(sequence_parallel=False, tensor_model_parallel_size=2)
    hist = [
        _strategy(sequence_parallel=True, tensor_model_parallel_size=2, max_mem="OOM")
    ]

    assert history_prune.prune_by_sequence_parallel(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_prune_by_mbs_recompute_sp_combines_all_memory_rules():
    current = _strategy(
        micro_batch_size=4,
        sequence_parallel=False,
        use_recompute=False,
    )
    hist = [
        _strategy(
            micro_batch_size=2,
            sequence_parallel=True,
            use_recompute=True,
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=1,
            max_mem="OOM",
        )
    ]

    assert history_prune.prune_by_mbs_recompute_sp(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_prune_by_tp_pp_prunes_same_parallel_product_lower_tp_after_oom():
    current = _strategy(tensor_model_parallel_size=1, pipeline_model_parallel_size=4)
    hist = [
        _strategy(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2, max_mem="OOM"
        )
    ]

    assert history_prune.prune_by_tp_pp(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_prune_registry_contains_wrapped_rules():
    registered_names = {
        func.__name__ for func in history_prune._HISTORY_BASED_PRUNE_FUNC
    }

    assert registered_names == {"wrapper"}
    assert len(history_prune._HISTORY_BASED_PRUNE_FUNC) >= 3
    assert all(callable(func) for func in history_prune._HISTORY_BASED_PRUNE_FUNC)


def test_registered_wrapper_delegates_to_function_logic():
    current = _strategy(micro_batch_size=4)
    hist = [_strategy(micro_batch_size=2, max_mem="OOM")]
    wrapper = history_prune._HISTORY_BASED_PRUNE_FUNC[0]

    assert wrapper(None, current, hist) is True
    assert current["max_mem"] == "OOM"


def test_history_rules_mutate_only_current_strategy_not_history():
    current = _strategy(micro_batch_size=4)
    hist_item = _strategy(micro_batch_size=2, max_mem="OOM")
    before = copy.deepcopy(hist_item)

    assert history_prune.prune_by_micro_batch_size(None, current, [hist_item]) is True
    assert hist_item == before
