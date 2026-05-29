import csv
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner.hetero import hetero_recorder as hetero_recorder_module
from flagscale.runner.auto_tuner.hetero.hetero_recorder import HeteroRecorder


def _config(tmp_path, hostfile=None, patterns=None, order="ascend"):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"hostfile": str(hostfile) if hostfile is not None else None},
                "auto_tuner": {
                    "performance": {"order": order},
                    "platform": {},
                    "hetero_memory_model": {"memory_grep_patterns": patterns or {}},
                },
            },
            "train": {"model": {"global_batch_size": 8}},
        }
    )


def test_init_loads_memory_patterns_and_host_map(tmp_path, monkeypatch):
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("node-a slots=8 type=a100\n", encoding="utf-8")
    monkeypatch.setattr(
        hetero_recorder_module,
        "parse_hostfile",
        lambda path: {"node-a": {"type": "a100"}, "node-b": {}},
    )

    recorder = HeteroRecorder(
        _config(
            tmp_path,
            hostfile=hostfile,
            patterns={"a100": "reserved", "default": "max reserved"},
        )
    )

    assert recorder.memory_patterns == {"a100": "reserved", "default": "max reserved"}
    assert recorder.host_to_type_map == {"node-a": "a100", "node-b": "default"}


def test_build_host_map_missing_hostfile_keeps_empty_map(tmp_path):
    recorder = HeteroRecorder(_config(tmp_path, hostfile=tmp_path / "missing.txt"))

    assert recorder.host_to_type_map == {}


def test_grep_max_memory_aggregates_by_device_type(tmp_path, monkeypatch):
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        hetero_recorder_module,
        "parse_hostfile",
        lambda path: {"node-a": {"type": "a100"}, "node-b": {"type": "h100"}},
    )
    logs = tmp_path / "logs" / "details"
    for host, lines in {
        "node-a": ["reserved: 10", "12 reserved"],
        "node-b": ["peak: 40.5", "20 peak"],
    }.items():
        leaf = logs / f"host_0_{host}" / "rank0"
        leaf.mkdir(parents=True)
        (leaf / "stdout.log").write_text("\n".join(lines), encoding="utf-8")

    recorder = HeteroRecorder(
        _config(
            tmp_path, hostfile=hostfile, patterns={"a100": "reserved", "h100": "peak"}
        )
    )

    assert recorder.grep_max_memory(str(tmp_path / "logs")) == {
        "a100": 12.0,
        "h100": 40.5,
    }


def test_grep_max_memory_returns_per_host_when_no_host_map(tmp_path):
    logs = tmp_path / "logs" / "details" / "host_0_node-a" / "rank0"
    logs.mkdir(parents=True)
    (logs / "stdout.log").write_text("max reserved: 33\n", encoding="utf-8")
    recorder = HeteroRecorder(_config(tmp_path, hostfile=None))

    assert recorder.grep_max_memory(str(tmp_path / "logs")) == {"node-a": 33.0}


def test_grep_max_memory_missing_details_returns_empty(tmp_path):
    recorder = HeteroRecorder(_config(tmp_path, hostfile=None))

    assert recorder.grep_max_memory(str(tmp_path / "missing")) == {}


def test_to_str_handles_scalars_omegaconf_and_json_fallback():
    assert HeteroRecorder._to_str(None) == ""
    assert HeteroRecorder._to_str(2.0) == "2"
    assert HeteroRecorder._to_str(2.5) == "2.5"
    assert HeteroRecorder._to_str(True) == "True"
    assert HeteroRecorder._to_str(float("inf")) == "inf"
    assert HeteroRecorder._to_str(float("nan")) == "nan"
    assert HeteroRecorder._to_str(OmegaConf.create({"a": 1})) == '{"a": 1}'
    assert HeteroRecorder._to_str([1, 2]) == "[1, 2]"


class _Unserializable:
    def __str__(self):
        return "fallback-object"


def test_to_str_falls_back_to_str_for_unserializable_object():
    assert HeteroRecorder._to_str(_Unserializable()) == "fallback-object"


def test_save_filters_pruned_rows_drops_runtime_columns_and_quotes_csv(tmp_path):
    recorder = HeteroRecorder(_config(tmp_path))
    history = [
        {
            "idx": 2,
            "performance": 5.0,
            "max_mem": 100,
            "max_mem_per_device": {"a100": 12},
            "pruned": False,
            "stopped_by_tuner": True,
            "hetero_memory_model_calibrated": [1, 2],
        },
        {"idx": 1, "performance": 1.0, "pruned": True, "prune_reason": "history"},
    ]

    recorder.save(history)

    with open(recorder.path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["idx"] == "2"
    assert rows[0]["performance"] == "5"
    assert rows[0]["max_mem_per_device"] == '{"a100": 12}'
    assert "max_mem" not in rows[0]
    assert "stopped_by_tuner" not in rows[0]


def test_save_empty_or_all_pruned_history_writes_empty_csv(tmp_path):
    recorder = HeteroRecorder(_config(tmp_path))

    recorder.save([])
    assert recorder.path
    assert open(recorder.path).read() == "\n"

    recorder.save([{"idx": 1, "performance": 1.0, "pruned": True}])
    assert open(recorder.path).read() == "\n"


def test_save_handles_dataframe_creation_failure(tmp_path, monkeypatch):
    recorder = HeteroRecorder(_config(tmp_path))

    def raise_dataframe(*args, **kwargs):
        raise RuntimeError("bad dataframe")

    monkeypatch.setattr(hetero_recorder_module.pd, "DataFrame", raise_dataframe)

    recorder.save([{"idx": 1, "performance": 1.0}])
    assert not Path(recorder.path).exists()
