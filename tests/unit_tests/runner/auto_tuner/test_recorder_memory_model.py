import os
import sys
import types
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner import memory_model
from flagscale.runner.auto_tuner.record.recorder import Recorder, ServeRecorder


def _config(tmp_path, order="ascend", metric_name=None):
    performance = {"order": order}
    if metric_name is not None:
        performance["name"] = metric_name
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "auto_tuner": {"performance": performance, "platform": {}},
            },
            "train": {
                "model": {
                    "global_batch_size": 8,
                    "seq_length": 16,
                    "train_samples": 1024,
                }
            },
        }
    )


def _make_logs(tmp_path, stdout_lines=None, host_lines=None):
    stdout_lines = stdout_lines or []
    host_lines = host_lines or []
    logs = tmp_path / "logs"
    attempt = logs / "details" / "host_0" / "output_0" / "sub_0" / "attempt_0" / "0"
    attempt.mkdir(parents=True)
    (attempt / "stdout.log").write_text("\n".join(stdout_lines), encoding="utf-8")
    logs.mkdir(exist_ok=True)
    (logs / "host_0.output").write_text("\n".join(host_lines), encoding="utf-8")
    return SimpleNamespace(experiment=SimpleNamespace(exp_dir=str(tmp_path)))


def test_get_all_performance_paths_finds_rank_stdout_logs(tmp_path):
    task = _make_logs(tmp_path, stdout_lines=["elapsed time per iteration (ms): 10"])
    recorder = Recorder(_config(tmp_path))

    performance_paths, host_path = recorder.get_all_performance_and_host_paths(task)

    assert len(performance_paths) == 1
    assert performance_paths[0].endswith("stdout.log")
    assert host_path == os.path.join(str(tmp_path), "logs")


def test_get_all_performance_paths_missing_details_raises(tmp_path):
    recorder = Recorder(_config(tmp_path))
    task = SimpleNamespace(experiment=SimpleNamespace(exp_dir=str(tmp_path)))

    with pytest.raises(ValueError, match="detail folder"):
        recorder.get_all_performance_and_host_paths(task)


def test_grep_performance_averages_after_warmup(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text(
        "elapsed time per iteration (ms): 100\n"
        "elapsed time per iteration (ms): 20\n"
        "elapsed time per iteration (ms): 40\n",
        encoding="utf-8",
    )
    recorder = Recorder(_config(tmp_path))
    recorder.cur_strategy = {"idx": 3}

    assert recorder.grep_performance([str(path)]) == 30.0


def test_grep_performance_returns_none_for_missing_metric(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("no metric here\n", encoding="utf-8")
    recorder = Recorder(_config(tmp_path))
    recorder.cur_strategy = {"idx": 3}

    assert recorder.grep_performance([str(path)]) is None


def test_grep_max_memory_accepts_prefix_and_suffix_patterns(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "host_0.output").write_text(
        "max reserved: 123.5\n" "512 max reserved\n" "not utf8 follows: \xff\n",
        encoding="latin-1",
    )
    recorder = Recorder(_config(tmp_path))
    recorder.cur_strategy = {"idx": 4}

    assert recorder.grep_max_memory(str(logs)) == 512.0


def test_grep_error_detects_oom_and_non_oom_errors(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "host_0.output").write_text(
        "Error: CUDA out of memory on rank 0\n" "Error: logical failure\n",
        encoding="utf-8",
    )
    recorder = Recorder(_config(tmp_path))
    recorder.cur_strategy = {"idx": 5}

    errors = recorder.grep_error(str(logs))

    assert "OOM" in errors
    assert any("logical failure" in error for error in errors)


def test_record_marks_oom_when_errors_contain_oom(tmp_path):
    task = _make_logs(
        tmp_path,
        stdout_lines=["elapsed time per iteration (ms): 99"],
        host_lines=["Error: CUDA out of memory", "max reserved: 900"],
    )
    recorder = Recorder(_config(tmp_path))
    strategy = {"idx": 1}

    recorder.record(task, strategy)

    assert strategy["performance"] is None
    assert strategy["max_mem"] == "OOM"
    assert "OOM" in strategy["error"]


def test_record_collects_performance_and_memory_for_success(tmp_path):
    task = _make_logs(
        tmp_path,
        stdout_lines=[
            "elapsed time per iteration (ms): 100",
            "elapsed time per iteration (ms): 30",
            "elapsed time per iteration (ms): 50",
        ],
        host_lines=["max reserved: 4096"],
    )
    recorder = Recorder(_config(tmp_path))
    strategy = {"idx": 2}

    recorder.record(task, strategy)

    assert strategy["performance"] == 40.0
    assert strategy["max_mem"] == 4096.0
    assert strategy["error"] is None


def test_record_stopped_by_tuner_keeps_partial_performance(tmp_path):
    task = _make_logs(
        tmp_path,
        stdout_lines=["elapsed time per iteration (ms): 77"],
        host_lines=["Error: timeout", "max reserved: 1000"],
    )
    recorder = Recorder(_config(tmp_path))
    strategy = {"idx": 3, "stopped_by_tuner": True}

    recorder.record(task, strategy)

    assert strategy["performance"] == 77.0
    assert strategy["max_mem"] == 1000.0
    assert strategy["error"] is None


def test_recorder_sort_filters_pruned_and_supports_orders(tmp_path):
    history = [
        {"idx": 1, "performance": 30.0},
        {"idx": 2, "performance": None},
        {"idx": 3, "performance": 10.0},
        {"idx": 4, "performance": 1.0, "pruned": True},
    ]

    ascend = Recorder(_config(tmp_path, order="ascend")).sort(history)
    descend = Recorder(_config(tmp_path, order="descend")).sort(history)

    assert [item["idx"] for item in ascend] == [3, 1, 2]
    assert [item["idx"] for item in descend] == [1, 3, 2]


def test_recorder_sort_rejects_unknown_order(tmp_path):
    recorder = Recorder(_config(tmp_path, order="sideways"))

    with pytest.raises(ValueError, match="not supported"):
        recorder.sort([{"idx": 1, "performance": 1.0}])


def test_save_and_read_history_roundtrip_types(tmp_path):
    recorder = Recorder(_config(tmp_path))
    os.makedirs(os.path.dirname(recorder.path), exist_ok=True)
    history = [
        {
            "idx": 2,
            "performance": 1.25,
            "max_mem": None,
            "sequence_parallel": True,
            "dims": [1, 2],
            "meta": {"a": 1},
            "stopped_by_tuner": True,
        }
    ]

    recorder.save(history)
    rows = recorder.read()

    assert rows == [
        {
            "idx": 2,
            "performance": 1.25,
            "max_mem": None,
            "sequence_parallel": True,
            "dims": [1, 2],
            "meta": {"a": 1},
            "stopped_by_tuner": None,
        }
    ]


def test_read_missing_history_returns_empty_list(tmp_path):
    recorder = Recorder(_config(tmp_path))

    assert recorder.read() == []


def test_parse_value_handles_scalars_and_json(tmp_path):
    recorder = Recorder(_config(tmp_path))

    assert recorder.parse_value("") is None
    assert recorder.parse_value("true") is True
    assert recorder.parse_value("False") is False
    assert recorder.parse_value("7") == 7
    assert recorder.parse_value("1.5") == 1.5
    assert recorder.parse_value('["a", 1]') == ["a", 1]
    assert recorder.parse_value('{"x": 2}') == {"x": 2}
    assert recorder.parse_value("plain") == "plain"


def test_serve_recorder_records_and_sorts_metrics(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "auto_tuner": {
                    "performance": {"metric": "request_throughput", "order": "descend"}
                },
            }
        }
    )
    recorder = ServeRecorder(cfg)
    strategy = {"idx": 1}

    recorder.record(
        strategy,
        {
            "mean_e2el_ms": 1.234,
            "request_throughput": 9.876,
            "total_token_throughput": 100.123,
            "mean_ttft_ms": 2.345,
            "mean_itl_ms": 3.456,
            "mean_tpot_ms": 4.567,
        },
    )
    sorted_history = recorder.sort(
        [
            strategy,
            {"idx": 2, "request_throughput": 1.0},
            {"idx": 3, "request_throughput": None},
        ]
    )

    assert strategy["e2e_latency"] == 1.23
    assert strategy["request_throughput"] == 9.88
    assert [item["idx"] for item in sorted_history] == [1, 2, 3]


def test_default_memory_model_uses_report_theoretical_memory(monkeypatch, tmp_path):
    fs_module = types.ModuleType("megatron.training.fs_theoretical_memory_usage")
    captured = {}

    def fake_report(args, num_microbatches):
        captured["args"] = args
        captured["num_microbatches"] = num_microbatches
        return 12345

    fs_module.report_theoretical_memory = fake_report
    megatron_module = types.ModuleType("megatron")
    megatron_module.__path__ = []
    training_module = types.ModuleType("megatron.training")
    training_module.__path__ = []
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.training", training_module)
    monkeypatch.setitem(
        sys.modules, "megatron.training.fs_theoretical_memory_usage", fs_module
    )
    monkeypatch.setattr(
        memory_model,
        "convert_config_to_megatron_args",
        lambda config, strategy: SimpleNamespace(hidden_size=8),
    )
    cfg = OmegaConf.create({"train": {"model": {"global_batch_size": 8}}})
    strategy = {"data_parallel_size": 2, "micro_batch_size": 2}

    assert memory_model.default_model(strategy, cfg) == 12345
    assert captured["num_microbatches"] == 2
    assert captured["args"].hidden_size == 8


def test_calculate_hetero_memory_delegates_with_global_batch_size(monkeypatch):
    captured = {}

    def fake_report(strategy, config, base_args):
        captured["strategy"] = strategy
        captured["config"] = config
        captured["base_args"] = base_args
        return [100, 200]

    monkeypatch.setattr(
        memory_model,
        "convert_config_to_megatron_args",
        lambda config, strategy: SimpleNamespace(),
    )
    monkeypatch.setattr(memory_model, "hetero_report_theoretical_memory", fake_report)
    cfg = OmegaConf.create({"train": {"model": {"global_batch_size": 16}}})
    strategy = {"hetero_process_meshes": [[1, 1, 1, 2, 1]]}

    assert memory_model.calculate_hetero_memory(strategy, cfg) == [100, 200]
    assert captured["base_args"].global_batch_size == 16
    assert captured["strategy"] is strategy


def test_to_str_helpers_handle_nan_and_complex_values(tmp_path):
    recorder = Recorder(_config(tmp_path))

    assert recorder.to_str(None) == ""
    assert recorder.to_str(float("nan")) == ""
    assert recorder.to_str({"x": [1]}) == '{"x": [1]}'
