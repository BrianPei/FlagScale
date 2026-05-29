import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.search.searcher import (
    Searcher,
    get_first_last_num_layers_for_pp,
)


def _config(space=None, priority=None, memory_model=None):
    auto_tuner = {
        "cards": 4,
        "nproc_per_node": 4,
        "platform": {},
        "space": space
        or {
            "data_parallel_size": [1, 2],
            "use_distributed_optimizer": [True, False],
            "tensor_model_parallel_size": [1, 2],
            "sequence_parallel": [True, False],
            "pipeline_model_parallel_size": [1, 2],
            "num_layers_per_virtual_pipeline_stage": [0, 1, 2],
            "use_recompute": [False, True],
            "recompute_method": ["uniform", "block"],
            "recompute_granularity": ["full", "selective"],
            "recompute_num_layers": [1, 2, 4],
            "micro_batch_size": [1, 2, 3, 4],
            "context_parallel_size": [1],
            "expert_model_parallel_size": [1],
        },
        "algo": {"name": "grid", "priority": priority},
    }
    if memory_model is not None:
        auto_tuner["memory_model"] = memory_model
    return OmegaConf.create(
        {
            "experiment": {"auto_tuner": auto_tuner},
            "train": {
                "model": {
                    "num_layers": 4,
                    "global_batch_size": 4,
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "seq_length": 16,
                }
            },
        }
    )


def test_get_first_last_num_layers_for_two_way_pipeline():
    assert get_first_last_num_layers_for_pp(7, 2) == (3, 4)
    assert get_first_last_num_layers_for_pp(8, 2) == (4, 4)


def test_get_first_last_num_layers_for_multi_way_pipeline_keeps_edges_non_empty():
    first, last = get_first_last_num_layers_for_pp(10, 4)

    assert first > 0
    assert last > 0
    assert first + last < 10


def test_sort_memory_priority_orders_memory_saving_values_first():
    searcher = Searcher.__new__(Searcher)
    dim = [1, 4, 2]
    searcher._sort("data_parallel_size", dim, "memory")
    assert dim == [1, 2, 4]

    dim = [False, True]
    searcher._sort("use_recompute", dim, "memory")
    assert dim == [True, False]


def test_sort_performance_priority_orders_fast_values_first():
    searcher = Searcher.__new__(Searcher)
    dim = [1, 4, 2]
    searcher._sort("micro_batch_size", dim, "performance")
    assert dim == [4, 2, 1]

    dim = [4, 1, 2]
    searcher._sort("tensor_model_parallel_size", dim, "performance")
    assert dim == [1, 2, 4]


def test_build_space_uses_auto_defaults_and_user_overrides():
    cfg = _config(
        {
            "data_parallel_size": [2],
            "use_distributed_optimizer": [False],
            "tensor_model_parallel_size": "auto",
            "sequence_parallel": [False],
            "pipeline_model_parallel_size": [1],
            "num_layers_per_virtual_pipeline_stage": [0],
            "use_recompute": [False],
            "recompute_method": ["uniform"],
            "recompute_granularity": ["full"],
            "recompute_num_layers": [1],
            "micro_batch_size": [1, 2],
            "context_parallel_size": [1],
        }
    )
    searcher = Searcher.__new__(Searcher)
    searcher.config = cfg

    space = Searcher.build_space(searcher, cfg)

    assert space["data_parallel_size"] == [2]
    assert space["tensor_model_parallel_size"] == [1, 2, 3, 4]
    assert space["expert_model_parallel_size"] == [1]


def test_searcher_filters_invalid_parallel_and_micro_batch_configs():
    cfg = _config(
        {
            "data_parallel_size": [1, 2, 3],
            "use_distributed_optimizer": [True, False],
            "tensor_model_parallel_size": [1, 2, 3],
            "sequence_parallel": [True, False],
            "pipeline_model_parallel_size": [1, 2, 3],
            "num_layers_per_virtual_pipeline_stage": [0, 1, 2],
            "use_recompute": [False],
            "recompute_method": ["uniform"],
            "recompute_granularity": ["full"],
            "recompute_num_layers": [1],
            "micro_batch_size": [1, 2, 3, 4],
            "context_parallel_size": [1, 2],
        }
    )

    searcher = Searcher(cfg)

    assert searcher.strategies
    for strategy in searcher.strategies:
        assert (
            strategy["data_parallel_size"]
            * strategy["tensor_model_parallel_size"]
            * strategy["pipeline_model_parallel_size"]
            * strategy["context_parallel_size"]
            == 4
        )
        assert 4 % (strategy["data_parallel_size"] * strategy["micro_batch_size"]) == 0
        assert 8 % strategy["tensor_model_parallel_size"] == 0
        assert 2 % strategy["tensor_model_parallel_size"] == 0
        assert strategy["micro_batch_size"] != 3


def test_searcher_disables_dist_optimizer_for_dp_one_and_sp_for_tp_one():
    cfg = _config(
        {
            "data_parallel_size": [1],
            "use_distributed_optimizer": [True],
            "tensor_model_parallel_size": [1],
            "sequence_parallel": [True],
            "pipeline_model_parallel_size": [4],
            "num_layers_per_virtual_pipeline_stage": [0],
            "use_recompute": [False],
            "recompute_method": ["uniform"],
            "recompute_granularity": ["full"],
            "recompute_num_layers": [1],
            "micro_batch_size": [1],
            "context_parallel_size": [1],
        }
    )

    searcher = Searcher(cfg)

    assert searcher.strategies
    assert all(
        strategy["use_distributed_optimizer"] is False
        for strategy in searcher.strategies
    )
    assert all(
        strategy["sequence_parallel"] is False for strategy in searcher.strategies
    )


def test_searcher_recompute_rules_normalize_disabled_and_selective_configs():
    cfg = _config(
        {
            "data_parallel_size": [1],
            "use_distributed_optimizer": [False],
            "tensor_model_parallel_size": [1],
            "sequence_parallel": [False],
            "pipeline_model_parallel_size": [4],
            "num_layers_per_virtual_pipeline_stage": [0],
            "use_recompute": [False, True],
            "recompute_method": ["uniform"],
            "recompute_granularity": ["full", "selective"],
            "recompute_num_layers": [1, 2, 4],
            "micro_batch_size": [1],
            "context_parallel_size": [1],
        }
    )

    searcher = Searcher(cfg)
    disabled = [s for s in searcher.strategies if not s["use_recompute"]]
    selective = [
        s for s in searcher.strategies if s["recompute_granularity"] == "selective"
    ]
    uniform = [s for s in searcher.strategies if s["recompute_method"] == "uniform"]

    assert disabled
    assert all(s["recompute_method"] is None for s in disabled)
    assert all(s["recompute_num_layers"] is None for s in disabled)
    assert selective
    assert all(s["recompute_method"] is None for s in selective)
    assert all(s["recompute_num_layers"] is None for s in selective)
    assert all(s["recompute_num_layers"] == 1 for s in uniform)


def test_searcher_rejects_unknown_algo():
    cfg = _config()
    cfg.experiment.auto_tuner.algo.name = "random"
    searcher = Searcher.__new__(Searcher)
    searcher.config = cfg

    with pytest.raises(NotImplementedError):
        Searcher.build_algo(searcher, [], cfg)


def test_grid_algo_iterates_and_reports_done():
    algo = GridAlgo([{"idx": 1}, {"idx": 2}], _config())

    assert algo.has_done() is False
    assert algo.search() == {"idx": 1}
    assert algo.search() == {"idx": 2}
    assert algo.search() is None
    assert algo.has_done() is True


def test_grid_algo_checkout_memory_model_sorts_descending_on_init():
    cfg = _config(memory_model={"model_name": "default"})
    strategies = [{"idx": 1, "memory_model": 100}, {"idx": 2, "memory_model": 300}]

    algo = GridAlgo(strategies, cfg)

    assert [strategy["idx"] for strategy in algo.strategies] == [2, 1]
