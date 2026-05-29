import csv

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner.hetero.hetero_pruner import HeteroPruner
from flagscale.runner.auto_tuner.hetero.hetero_searcher import (
    HeteroSearcher,
    _generate_all_partitions_with_max_diff,
)


def _hetero_config(tmp_path=None, extra_auto_tuner=None, split="auto"):
    exp_dir = str(tmp_path) if tmp_path is not None else "/tmp/flagscale-test"
    auto_tuner = {
        "algo": {"name": "grid", "priority": None},
        "space": {
            "hetero_process_meshes": [1, 1, 1, "auto", 1, 1, 1, 1, "auto", 1],
            "hetero_pipeline_layer_split": split,
            "hetero_inter_mesh_max_layer_diff": 4,
            "hetero_intra_mesh_max_layer_diff": 4,
            "micro_batch_size": [1, 2],
            "use_distributed_optimizer": [False, True],
            "use_recompute": [False, True],
            "sequence_parallel": [True, False],
            "recompute_granularity_per_stage_micro_batch": "auto",
            "recompute_method_per_stage_micro_batch": "auto",
            "recompute_num_layers_per_stage_micro_batch": "auto",
        },
    }
    if extra_auto_tuner:
        auto_tuner.update(extra_auto_tuner)
    return OmegaConf.create(
        {
            "experiment": {"exp_dir": exp_dir, "auto_tuner": auto_tuner},
            "train": {
                "model": {
                    "num_layers": 4,
                    "global_batch_size": 4,
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "seq_length": 16,
                    "untie_embeddings_and_output_weights": False,
                },
                "system": {"hetero": {"hetero_device_types": ["a100", "h100"]}},
            },
        }
    )


def _resources():
    return {
        "node-a": {"type": "a100", "slots": 2},
        "node-b": {"type": "h100", "slots": 2},
    }


def _valid_strategy(**overrides):
    strategy = {
        "idx": 1,
        "pipeline_model_parallel_size": 2,
        "hetero_pipeline_layer_split": [2, 2],
        "hetero_process_meshes": [[1, 1, 1, 2, 1], [1, 1, 1, 2, 1]],
        "hetero_device_types": ["a100", "h100"],
        "micro_batch_size": 1,
        "sequence_parallel": False,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "performance": None,
        "max_mem": None,
    }
    strategy.update(overrides)
    return strategy


def test_generate_all_partitions_with_max_diff_balances_values():
    partitions = list(_generate_all_partitions_with_max_diff(6, 3, 1))

    assert [2, 2, 2] in partitions
    assert all(sum(partition) == 6 for partition in partitions)
    assert all(partition[0] - partition[-1] <= 1 for partition in partitions)


def test_generate_all_partitions_edge_cases():
    assert list(_generate_all_partitions_with_max_diff(0, 0, 0)) == [[]]
    assert list(_generate_all_partitions_with_max_diff(5, 1, 0)) == [[5]]
    assert list(_generate_all_partitions_with_max_diff(0, 2, 0)) == []


def test_hetero_build_space_parses_mesh_templates_and_device_types():
    cfg = _hetero_config()
    searcher = HeteroSearcher.__new__(HeteroSearcher)
    searcher.config = cfg
    searcher.resources = _resources()
    searcher.recompute_search_space = {}

    space = HeteroSearcher.build_space(searcher, cfg)

    assert space["mesh_templates"] == [[1, 1, 1, "auto", 1], [1, 1, 1, "auto", 1]]
    assert space["device_types"] == ["a100", "h100"]
    assert searcher.recompute_search_space["use_recompute"] == [False, True]


def test_hetero_build_space_rejects_bad_mesh_template_length():
    cfg = _hetero_config()
    cfg.experiment.auto_tuner.space.hetero_process_meshes = [1, 1, 1]
    searcher = HeteroSearcher.__new__(HeteroSearcher)
    searcher.config = cfg
    searcher.recompute_search_space = {}

    try:
        HeteroSearcher.build_space(searcher, cfg)
    except ValueError as exc:
        assert "divisible by 5" in str(exc)
    else:
        raise AssertionError("invalid hetero_process_meshes should raise")


def test_get_valid_mesh_configs_filters_by_hidden_size_and_batch_divisibility():
    cfg = _hetero_config()
    searcher = HeteroSearcher.__new__(HeteroSearcher)
    nodes = [{"slots": 4, "type": "a100"}]

    valid = HeteroSearcher._get_valid_mesh_configs(
        searcher, ["auto", 1, 1, "auto", 1], nodes, cfg, [1, 2]
    )

    assert [1, 1, 1, 4, 1] in valid
    assert [2, 1, 1, 2, 1] in valid
    assert all(mesh[0] in {1, 2, 4} for mesh in valid)


def test_hetero_searcher_generates_small_strategy_space():
    cfg = _hetero_config(split=[2, 2])

    searcher = HeteroSearcher(cfg, _resources())

    assert searcher.strategies
    for strategy in searcher.strategies:
        assert strategy["hetero_pipeline_layer_split"] == [2, 2]
        assert strategy["pipeline_model_parallel_size"] == 2
        assert strategy["micro_batch_size"] in [1, 2]
        assert strategy["hetero_device_types"] == ["a100", "h100"]


def test_hetero_recompute_auto_templates_render_all_placeholder():
    searcher = HeteroSearcher.__new__(HeteroSearcher)
    searcher.recompute_search_space = {
        "use_recompute": [True],
        "granularity": "auto",
        "method": "auto",
        "num_layers": "auto",
    }

    configs = HeteroSearcher._generate_recompute_configs(
        searcher, pp_size=2, num_micro_batches=3
    )

    assert configs
    assert all(
        cfg["recompute_granularity_per_stage_micro_batch"][0][1] == 3 for cfg in configs
    )
    assert all(
        cfg["recompute_num_layers_per_stage_micro_batch"][0][2] == 1 for cfg in configs
    )


def test_hetero_pruner_architecture_checks_missing_and_bad_split(tmp_path):
    pruner = HeteroPruner(_hetero_config(tmp_path))

    assert pruner._check_architectural_validity({})[0] is True
    invalid, reason = pruner._check_architectural_validity(
        _valid_strategy(hetero_pipeline_layer_split=[1, 1, 1])
    )
    assert invalid is True
    assert "Layer split length" in reason

    invalid, reason = pruner._check_architectural_validity(
        _valid_strategy(hetero_pipeline_layer_split=[1, 1])
    )
    assert invalid is True
    assert "Layer split sum" in reason


def test_hetero_pruner_architecture_checks_tied_embeddings_and_sp_rules(tmp_path):
    pruner = HeteroPruner(_hetero_config(tmp_path))

    invalid, reason = pruner._check_architectural_validity(
        _valid_strategy(hetero_process_meshes=[[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]])
    )
    assert invalid is True
    assert "Tied embeddings" in reason

    invalid, reason = pruner._check_architectural_validity(
        _valid_strategy(
            hetero_process_meshes=[[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
            sequence_parallel=False,
        )
    )
    assert invalid is True
    assert "Tied embeddings" in reason

    invalid, reason = pruner._check_architectural_validity(
        _valid_strategy(sequence_parallel=True)
    )
    assert invalid is True
    assert "TP=1" in reason


def test_hetero_history_oom_prunes_same_mesh_and_batch(tmp_path):
    pruner = HeteroPruner(_hetero_config(tmp_path))
    current = _valid_strategy(use_recompute=False)
    history = [
        _valid_strategy(
            idx=7,
            use_recompute=True,
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=1,
            max_mem="OOM",
        )
    ]

    assert pruner._check_hetero_history_oom(current, history) is True
    assert "History task 7" in current["prune_reason"]


def test_hetero_memory_model_utilization_handles_inf_and_threshold(tmp_path):
    cfg = _hetero_config(
        tmp_path,
        extra_auto_tuner={
            "hetero_memory_model": {
                "gpu_memory": {"a100": 1000, "h100": 2000},
                "gpu_utilization": [0.2, 0.8],
            }
        },
    )
    pruner = HeteroPruner(cfg)

    is_oom, reason = pruner._check_memory_model_utilization(
        _valid_strategy(hetero_memory_model=float("inf"))
    )
    assert is_oom is True
    assert "infinity" in reason

    is_oom, reason = pruner._check_memory_model_utilization(
        _valid_strategy(hetero_memory_model=[900, 1000])
    )
    assert is_oom is True
    assert "a100" in reason

    is_oom, reason = pruner._check_memory_model_utilization(
        _valid_strategy(hetero_memory_model=[700, 1000])
    )
    assert is_oom is False
    assert reason == ""


def test_hetero_prune_applies_side_effects_and_saves_csv(tmp_path):
    pruner = HeteroPruner(_hetero_config(tmp_path))
    strategy = _valid_strategy(hetero_pipeline_layer_split=[1, 1])

    assert pruner.prune(strategy, []) is True
    assert strategy["pruned"] is True
    assert strategy["pruned_idx"] == 1
    assert pruner.pruned_count == 1

    pruner.save_pruned_history()
    with open(pruner.pruned_history_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert rows[0]["pruned_idx"] == "1"
    assert "Layer split sum" in rows[0]["prune_reason"]


def test_hetero_prune_appends_valid_strategy_to_history(tmp_path):
    pruner = HeteroPruner(_hetero_config(tmp_path))
    strategy = _valid_strategy()
    history = []

    assert pruner.prune(strategy, history) is False
    assert history == [strategy]
    assert pruner.pruned_count == 0
