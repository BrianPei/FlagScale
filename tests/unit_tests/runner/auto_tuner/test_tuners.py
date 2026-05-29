from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("pandas")

from flagscale.runner.auto_tuner import tuner_serve, tuner_train
from flagscale.runner.utils import JobStatus


class _DummyAlgo:
    def __init__(self):
        self.idx = 0
        self.checkout_mode = None

    def checkout(self, mode):
        self.checkout_mode = mode


class _DummySearcher:
    def __init__(self, *args, **kwargs):
        self.algo = _DummyAlgo()


class _DummyPruner:
    def __init__(self, *args, **kwargs):
        self.pruned_count = 0


class _DummyRecorder:
    metric = "performance"

    def __init__(self, *args, **kwargs):
        self.cur_strategy = None
        self.record_calls = []
        self.saved = None

    def read(self):
        return [{"idx": 1, "performance": 10.0}]

    def sort(self, history):
        return sorted(history, key=lambda item: item.get("performance") or float("inf"))

    def record(self, *args):
        self.record_calls.append(args)
        self.cur_strategy = args[-1] if args else None

    def save(self, history):
        self.saved = list(history)


class _DummyGenerator:
    def __init__(self, *args, **kwargs):
        self.args = args

    def gen_best_task(self, strategy, config):
        return OmegaConf.create({"experiment": {"runner": {}}, "action": "idle"})


class _DummyRunner:
    def __init__(self, task):
        self.task = task
        self.ran = False
        self.stopped = False
        self.launcher = self
        self.statuses = []
        self.sub_processes = []
        self.serve_alive_values = []

    def run(self, *args, **kwargs):
        self.ran = True
        self.run_args = args
        self.run_kwargs = kwargs

    def stop(self):
        self.stopped = True

    def _query_status(self):
        return self.statuses.pop(0) if self.statuses else JobStatus.COMPLETED_OR_IDLE

    def _query_sub_process_status(self):
        return self.sub_processes.pop(0) if self.sub_processes else False

    def _serve_alive(self):
        return self.serve_alive_values.pop(0) if self.serve_alive_values else True

    def _profile_serve(self):
        return {"latency": 1.5}


def _train_config(tmp_path, hetero=False):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {
                    "nnodes": 2,
                    "nproc_per_node": 4,
                    "hostfile": str(tmp_path / "hosts"),
                },
                "auto_tuner": {"control": {"interval": 0, "max_time_per_task": 5}},
            },
            "train": {"system": {"hetero": {"enable_hetero": hetero}}},
        }
    )


def _serve_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {
                    "nnodes": 1,
                    "nproc_per_node": 2,
                    "deploy": {"use_fs_serve": False, "port": 9000},
                },
                "auto_tuner": {"control": {"interval": 0, "max_time_per_task": 5}},
            },
            "serve": [{"serve_id": "svc", "engine_args": {}}],
        }
    )


def test_train_auto_tuner_init_homogeneous_wires_components(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner_train, "Searcher", _DummySearcher)
    monkeypatch.setattr(tuner_train, "Pruner", _DummyPruner)
    monkeypatch.setattr(tuner_train, "Generator", _DummyGenerator)
    monkeypatch.setattr(tuner_train, "Recorder", _DummyRecorder)

    tuner = tuner_train.TrainAutoTuner(_train_config(tmp_path, hetero=False))

    assert isinstance(tuner.searcher, _DummySearcher)
    assert isinstance(tuner.pruner, _DummyPruner)
    assert isinstance(tuner.generator, _DummyGenerator)
    assert isinstance(tuner.recorder, _DummyRecorder)
    assert tuner.config.experiment.auto_tuner.cards == 8
    assert tuner.history == [{"idx": 1, "performance": 10.0}]


def test_train_auto_tuner_init_hetero_uses_hostfile_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tuner_train, "parse_hostfile", lambda path: {"node-a": {"slots": 3}}
    )
    monkeypatch.setattr(tuner_train, "HeteroSearcher", _DummySearcher)
    monkeypatch.setattr(tuner_train, "HeteroPruner", _DummyPruner)
    monkeypatch.setattr(tuner_train, "HeteroGenerator", _DummyGenerator)
    monkeypatch.setattr(tuner_train, "HeteroRecorder", _DummyRecorder)

    tuner = tuner_train.TrainAutoTuner(_train_config(tmp_path, hetero=True))

    assert isinstance(tuner.searcher, _DummySearcher)
    assert tuner.config.experiment.auto_tuner.cards == 3


def test_train_auto_tuner_init_hetero_requires_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner_train, "parse_hostfile", lambda path: {})

    with pytest.raises(ValueError, match="valid hostfile"):
        tuner_train.TrainAutoTuner(_train_config(tmp_path, hetero=True))


def test_train_auto_tuner_log_helpers_and_clear_log(tmp_path):
    tuner = tuner_train.TrainAutoTuner.__new__(tuner_train.TrainAutoTuner)
    tuner.logger = SimpleNamespace(
        error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None
    )
    log = tmp_path / "tuner.log"
    log.write_text("Searching 3 / 10\nPruned 2 strategy\n", encoding="utf-8")
    doomed = tmp_path / "task_1"
    doomed.mkdir()
    (doomed / "x.txt").write_text("x", encoding="utf-8")

    assert tuner.find_search_num_value(str(log)) == "3"
    assert tuner.find_pruned_num_value(str(log)) == "2"
    assert tuner.find_search_num_value(str(tmp_path / "missing.log")) == "0"
    tuner.clear_log(str(doomed))
    assert not doomed.exists()


def test_train_auto_tuner_run_record_get_best_and_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner_train, "SSHTrainRunner", _DummyRunner)
    tuner = tuner_train.TrainAutoTuner.__new__(tuner_train.TrainAutoTuner)
    tuner.cur_task = OmegaConf.create(
        {"experiment": {"runner": {"enable_monitoring": True}}}
    )
    tuner.cur_strategy = {"idx": 1}
    tuner.history = [{"idx": 1, "performance": None}, {"idx": 2, "performance": 5.0}]
    tuner.searcher = _DummySearcher()
    tuner.recorder = _DummyRecorder()
    tuner.has_checkout = False

    tuner.run()
    tuner.record()
    tuner.checkout("memory")

    assert tuner.runner.ran is True
    assert tuner.runner.run_kwargs == {"enable_monitoring": True}
    assert tuner.recorder.saved == tuner.history
    assert tuner.get_best() == {"idx": 2, "performance": 5.0}
    assert tuner.has_checkout is True
    assert tuner.searcher.algo.checkout_mode == "memory"


def test_serve_auto_tuner_init_sets_port_and_components(tmp_path, monkeypatch):
    monkeypatch.setattr(tuner_serve, "ServeSearcher", _DummySearcher)
    monkeypatch.setattr(tuner_serve, "ServeGenerator", _DummyGenerator)
    monkeypatch.setattr(tuner_serve, "ServeRecorder", _DummyRecorder)

    cfg = _serve_config(tmp_path)
    tuner = tuner_serve.ServeAutoTuner(cfg)

    assert cfg.serve[0].engine_args.port == 9000
    assert isinstance(tuner.searcher, _DummySearcher)
    assert tuner.pruner is None
    assert isinstance(tuner.generator, _DummyGenerator)
    assert isinstance(tuner.recorder, _DummyRecorder)
    assert tuner.config.experiment.auto_tuner.cards == 2


def test_serve_auto_tuner_run_monitor_and_record(monkeypatch):
    monkeypatch.setattr(tuner_serve, "SSHServeRunner", _DummyRunner)
    monkeypatch.setattr(tuner_serve.time, "sleep", lambda *_: None)
    times = iter([100.0, 101.0, 102.0, 102.5])
    monkeypatch.setattr(tuner_serve.time, "time", lambda: next(times))
    tuner = tuner_serve.ServeAutoTuner.__new__(tuner_serve.ServeAutoTuner)
    tuner.cur_task = OmegaConf.create({"experiment": {"runner": {}}})
    tuner.cur_strategy = {"idx": 1}
    tuner.idx = 1
    tuner.interval = 0
    tuner.max_time_per_task = 10
    tuner.recorder = _DummyRecorder()
    tuner.history = []
    tuner.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    tuner.run()
    tuner.runner.statuses = [JobStatus.RUNNING]
    tuner.runner.serve_alive_values = [True]
    tuner.monitor()
    tuner.record()

    assert tuner.cur_result == {"latency": 1.5}
    assert tuner.cur_strategy["elapsed_time"] == 2.5
    assert tuner.history == [tuner.recorder.cur_strategy]
