from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner.hetero import hetero_theoretical_memory as htm


def _args(**overrides):
    args = {
        "num_layers": 4,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "kv_channels": 4,
        "num_query_groups": 4,
        "seq_length": 8,
        "micro_batch_size": 2,
        "ffn_hidden_size": 64,
        "swiglu": False,
        "vocab_size": 128,
        "padded_vocab_size": 128,
        "num_experts": None,
        "moe_layer_freq": 1,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
        "moe_router_topk": 1,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "sequence_parallel": False,
        "virtual_pipeline_model_parallel_size": None,
        "multi_latent_attention": False,
        "qk_layernorm": False,
        "untie_embeddings_and_output_weights": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _strategy(**overrides):
    strategy = {
        "hetero_process_meshes": [[1, 1, 1, 1, 2]],
        "hetero_pipeline_layer_split": [2, 2],
        "pipeline_model_parallel_size": 2,
        "micro_batch_size": 2,
        "sequence_parallel": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
    }
    strategy.update(overrides)
    return strategy


def _config(global_batch_size=8):
    return OmegaConf.create(
        {"train": {"model": {"global_batch_size": global_batch_size}}}
    )


def test_get_global_moe_pattern_dense_and_integer_frequency():
    assert htm._get_global_moe_pattern(_args(num_experts=None, num_layers=3)) == [
        0,
        0,
        0,
    ]
    assert htm._get_global_moe_pattern(
        _args(num_experts=8, moe_layer_freq=2, num_layers=5)
    ) == [
        1,
        0,
        1,
        0,
        1,
    ]


def test_get_global_moe_pattern_repeats_list_frequency():
    assert htm._get_global_moe_pattern(
        _args(num_experts=8, moe_layer_freq=[1, 0, 0], num_layers=5)
    ) == [
        1,
        0,
        0,
        1,
        0,
    ]


def test_get_mesh_params_for_stage_maps_offsets_and_skips_invalid_mesh():
    meshes = [[1, 1, 1, 2, 2], [2, 1, 1, 1, 1], ["bad"]]

    assert htm._get_mesh_params_for_stage(0, meshes) == {
        "tp": 1,
        "cp": 1,
        "ep": 1,
        "dp": 2,
        "mesh_idx": 0,
    }
    assert htm._get_mesh_params_for_stage(2, meshes) == {
        "tp": 2,
        "cp": 1,
        "ep": 1,
        "dp": 1,
        "mesh_idx": 1,
    }
    assert htm._get_mesh_params_for_stage(9, meshes) is None


def test_activation_component_helpers_cover_standard_mla_moe_and_vocab_fallbacks():
    standard = htm._calculate_attn_activation_components(_args())
    mla = htm._calculate_attn_activation_components(
        _args(
            multi_latent_attention=True,
            qk_head_dim=8,
            qk_pos_emb_head_dim=4,
            v_head_dim=8,
        )
    )
    dense_mlp = htm._calculate_mlp_activation_components(_args(), is_expert=False)
    expert_mlp = htm._calculate_mlp_activation_components(
        _args(num_experts=4, moe_ffn_hidden_size=32, moe_router_topk=2, swiglu=True),
        is_expert=True,
    )
    gate = htm._calculate_moe_gate_activation(
        _args(num_experts=4, moe_router_topk=None)
    )
    embedding = htm._calculate_embedding_activation(_args(padded_vocab_size=None))
    output = htm._calculate_output_layer_activation(_args(padded_vocab_size=None))

    assert standard["tp_scaled"] > 0
    assert mla["tp_scaled"] > 0
    assert dense_mlp["tp_scaled"] > 0
    assert expert_mlp["tp_scaled"] > dense_mlp["tp_scaled"]
    assert gate > 0
    assert embedding > 0
    assert output > embedding


def test_weight_optimizer_memory_handles_dense_moe_invalid_mesh_and_optimizer_multiplier():
    strategy = _strategy(
        hetero_process_meshes=[[1, 1, 1, 2, 1], [2, 1, 2, 1, 1], ["invalid"]],
        hetero_pipeline_layer_split=[2, 2],
    )
    dense_args = _args(use_distributed_optimizer=False)
    do_args = _args(use_distributed_optimizer=True)
    moe_args = _args(num_experts=4, moe_layer_freq=[1, 0], moe_ffn_hidden_size=32)

    dense_mem = htm.hetero_compute_weight_and_optimizer_memory(
        dense_args, strategy, _config()
    )
    do_mem = htm.hetero_compute_weight_and_optimizer_memory(
        do_args, strategy, _config()
    )
    moe_mem = htm.hetero_compute_weight_and_optimizer_memory(
        moe_args, strategy, _config()
    )

    assert set(dense_mem) == {0, 1}
    assert dense_mem[0] > do_mem[0]
    assert moe_mem[0] > dense_mem[0]


def test_activation_memory_handles_invalid_stage_uniform_recompute_and_sequence_parallel(
    monkeypatch,
):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    base_args = _args(num_experts=4, moe_layer_freq=[1, 0])
    invalid_stage = _strategy(
        hetero_process_meshes=[[1, 1, 1, 1, 1]], hetero_pipeline_layer_split=[1, 1]
    )
    no_recompute = _strategy()
    uniform = _strategy(
        recompute_method="uniform", recompute_granularity="full", recompute_num_layers=1
    )
    sequence_parallel = _strategy(
        sequence_parallel=True, hetero_process_meshes=[[2, 1, 1, 1, 2]]
    )

    invalid_result = htm.hetero_compute_activation_memory(
        base_args, invalid_stage, _config()
    )
    no_recompute_result = htm.hetero_compute_activation_memory(
        base_args, no_recompute, _config()
    )
    uniform_result = htm.hetero_compute_activation_memory(base_args, uniform, _config())
    sp_result = htm.hetero_compute_activation_memory(
        base_args, sequence_parallel, _config()
    )

    assert invalid_result[1] == float("inf")
    assert len(no_recompute_result) == 2
    assert uniform_result[0] < no_recompute_result[0]
    assert sp_result[0] < no_recompute_result[0]


def test_report_theoretical_memory_aggregates_static_and_activation(monkeypatch):
    monkeypatch.setattr(
        htm,
        "hetero_compute_weight_and_optimizer_memory",
        lambda *_: {0: 2 * htm.NUM_BYTES_IN_MEGABYTE},
    )
    monkeypatch.setattr(
        htm,
        "hetero_compute_activation_memory",
        lambda *_: [3 * htm.NUM_BYTES_IN_MEGABYTE],
    )

    result = htm.hetero_report_theoretical_memory(
        _strategy(
            hetero_process_meshes=[[1, 1, 1, 1, 1]], hetero_pipeline_layer_split=[1]
        ),
        _config(),
        _args(),
    )

    assert result == [5]


def test_report_theoretical_memory_returns_inf_per_mesh_on_failure(monkeypatch):
    def raise_error(*_):
        raise RuntimeError("boom")

    monkeypatch.setattr(htm, "hetero_compute_weight_and_optimizer_memory", raise_error)

    assert htm.hetero_report_theoretical_memory(
        _strategy(hetero_process_meshes=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        _config(),
        _args(),
    ) == [float("inf"), float("inf")]
