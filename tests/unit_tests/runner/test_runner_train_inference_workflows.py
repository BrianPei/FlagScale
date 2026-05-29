import collections
import subprocess

import pytest
from omegaconf import OmegaConf

from flagscale.runner import runner_inference, runner_train
from flagscale.runner.runner_base_legacy import JobStatus


def _train_config(tmp_path, runner_extra=None):
    runner = {
        "type": "ssh",
        "backend": "torchrun",
        "hostfile": None,
        "ssh_port": 2222,
        "nproc_per_node": 2,
        "rdzv_id": "run-id",
        "tee": "1",
    }
    if runner_extra:
        runner.update(runner_extra)
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "exp"),
                "task": {
                    "type": "train",
                    "backend": "megatron",
                    "entrypoint": "train.py",
                },
                "runner": runner,
                "envs": {"CUDA_VISIBLE_DEVICES": "0,1", "ENV": "value"},
                "cmds": {"before_start": "echo before", "after_stop": "echo after"},
            },
            "train": {
                "system": {
                    "checkpoint": {
                        "save": str(tmp_path / "ckpt-save"),
                        "load": str(tmp_path / "ckpt-load"),
                    },
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "scripts_dir": str(tmp_path / "logs" / "scripts"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                        "details_dir": str(tmp_path / "logs" / "details"),
                        "tensorboard_dir": str(tmp_path / "tensorboard"),
                        "wandb_save_dir": str(tmp_path / "wandb"),
                    },
                },
                "model": {"num_layers": 2},
                "data": {},
            },
        }
    )


def _inference_config(tmp_path, runner_extra=None):
    runner = {"type": "ssh", "hostfile": None, "ssh_port": 2222, "nproc_per_node": 2}
    if runner_extra:
        runner.update(runner_extra)
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "infer-exp"),
                "task": {
                    "type": "inference",
                    "backend": "vllm",
                    "entrypoint": "serve.py",
                },
                "runner": runner,
                "envs": {"CUDA_VISIBLE_DEVICES": "0,1"},
                "cmds": {
                    "before_start": "echo infer-before",
                    "after_stop": "echo infer-after",
                },
            },
            "inference": {
                "logging": {
                    "log_dir": str(tmp_path / "infer-logs"),
                    "scripts_dir": str(tmp_path / "infer-logs" / "scripts"),
                    "pids_dir": str(tmp_path / "infer-logs" / "pids"),
                },
                "model": "qwen",
            },
        }
    )


def test_generate_run_script_train_foreground_monitoring_and_no_shared_fs(tmp_path):
    config = _train_config(
        tmp_path, runner_extra={"no_shared_fs": True, "ssh_port": 2022}
    )
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()

    script = runner_train._generate_run_script_train(
        config,
        "worker0",
        3,
        "torchrun train.py",
        background=False,
        pkg_dir=str(pkg_dir),
        enable_monitoring=True,
    )

    content = open(script, encoding="utf-8").read()
    assert "echo before" in content
    assert f"mkdir -p {config.train.system.checkpoint.load}" in content
    assert f"cd {pkg_dir}" in content
    assert "monitor_launcher.py" in content
    assert "--node-rank 3" in content
    assert "--no-shared-fs" in content
    assert "--ssh-port 2022" in content
    assert "set -o pipefail" in content
    assert "tee -a" in content
    assert "host.output" in content


def test_generate_stop_and_query_scripts_train_include_fallbacks(tmp_path):
    config = _train_config(tmp_path)
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config

    stop_script = runner_train._generate_stop_script_train(config, "localhost", 0)
    query_script = runner._generate_query_script("localhost", 0)
    query_children_script = runner._generate_query_sub_process_script("localhost", 0)

    assert "pkill -P $pid" in open(stop_script, encoding="utf-8").read()
    assert "pkill -f 'torchrun'" in open(stop_script, encoding="utf-8").read()
    assert "echo after" in open(stop_script, encoding="utf-8").read()
    assert "ps -p $pid -o state" in open(query_script, encoding="utf-8").read()
    assert "ps -eo pid,ppid" in open(query_children_script, encoding="utf-8").read()


def test_run_node_derives_nproc_and_current_envs(mocker):
    func = mocker.Mock()
    runner_config = OmegaConf.create(
        {"nproc_per_node": 8, "master_addr": "master", "master_port": 1234}
    )
    user_envs = {
        "BASE": "1",
        "device_type_specific": {"A100": {"CUDA_VISIBLE_DEVICES": "0,1"}},
    }

    runner_train.run_node(
        func,
        1,
        "worker1",
        {"slots": 4, "type": "A100"},
        user_envs,
        runner_config,
        2,
        "10.0.0.1",
        29500,
        True,
        False,
    )

    func.assert_called_once()
    kwargs = func.call_args.kwargs
    assert kwargs["device_type"] == "A100"
    assert kwargs["background"] is True
    assert kwargs["cur_envs"] == {"BASE": "1", "CUDA_VISIBLE_DEVICES": "0,1"}
    assert func.call_args.args[:6] == ("worker1", "master", 1234, 2, 1, 2)


def test_ssh_train_run_each_local_updates_device_and_node_specific(tmp_path, mocker):
    config = _train_config(tmp_path)
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config
    runner.user_envs = config.experiment.envs
    runner.user_script = "train.py"
    runner.user_args = ["--use-cache", "false"]
    runner.device_type_specific = {"A100": {"new_flag": "true", "use_cache": "true"}}
    runner.node_specific = {"localhost": {"another_flag": "true"}}
    run_local = mocker.patch("flagscale.runner.runner_train.run_local_command")

    runner._run_each(
        "localhost",
        "127.0.0.1",
        29500,
        1,
        0,
        2,
        device_type="A100",
        background=False,
        dryrun=True,
        cur_envs={"CUDA_VISIBLE_DEVICES": "0,1"},
        enable_monitoring=False,
    )

    script_path = run_local.call_args.args[0].removeprefix("bash ")
    content = open(script_path, encoding="utf-8").read()
    assert "CUDA_VISIBLE_DEVICES=0,1" in content
    assert "--hetero-current-device-type A100" in content
    assert "--use-cache" in content
    assert "--new-flag" in content
    assert "--another-flag" in content
    assert run_local.call_args.args[:2] == (f"bash {script_path}", True)
    assert run_local.call_args.kwargs == {"stream_output": True}


def test_ssh_train_run_each_remote_uses_ssh_and_scp_for_no_shared_fs(tmp_path, mocker):
    config = _train_config(tmp_path, runner_extra={"no_shared_fs": True})
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config
    runner.user_script = "train.py"
    runner.user_args = []
    runner.device_type_specific = None
    runner.node_specific = None
    run_ssh = mocker.patch("flagscale.runner.runner_train.run_ssh_command")
    run_scp = mocker.patch("flagscale.runner.runner_train.run_scp_command")

    runner._run_each(
        "worker0",
        "10.0.0.1",
        29500,
        1,
        0,
        2,
        background=True,
        dryrun=True,
        cur_envs={"ENV": "value"},
    )

    assert run_ssh.call_args_list[0].args == (
        "worker0",
        f"mkdir -p {config.train.system.logging.scripts_dir}",
        2222,
        True,
    )
    run_scp.assert_called_once()
    assert run_ssh.call_args_list[1].args[:4] == (
        "worker0",
        f"bash {run_scp.call_args.args[1]}",
        2222,
        True,
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["R"], JobStatus.RUNNING),
        ([""], JobStatus.COMPLETED_OR_IDLE),
        (["R", ""], JobStatus.TRANSITIONAL),
    ],
)
def test_ssh_train_query_status_classifies_local_and_multinode(
    statuses, expected, mocker
):
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.resources = None if len(statuses) == 1 else {"h0": {}, "h1": {}}
    mocker.patch.object(runner, "_query_each", side_effect=statuses)

    assert runner._query_status() == expected


def test_ssh_train_query_sub_process_status_requires_all_nodes(mocker):
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.resources = {"h0": {}, "h1": {}}
    mocker.patch.object(runner, "_query_each_sub_process", side_effect=["123", ""])

    assert runner._query_sub_process_status() is False


def test_ssh_train_query_each_local_and_remote_paths(tmp_path, mocker):
    config = _train_config(tmp_path, runner_extra={"no_shared_fs": True})
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config
    mocker.patch.object(runner, "_generate_query_script", return_value="/tmp/query.sh")
    local = mocker.patch(
        "flagscale.runner.runner_train.run_local_command",
        return_value=subprocess.CompletedProcess("cmd", 0, stdout="R\n", stderr=""),
    )
    ssh = mocker.patch(
        "flagscale.runner.runner_train.run_ssh_command",
        return_value=subprocess.CompletedProcess("ssh", 0, stdout="S\n", stderr=""),
    )
    scp = mocker.patch("flagscale.runner.runner_train.run_scp_command")

    assert runner._query_each("localhost", 0) == "R"
    assert runner._query_each("worker0", 1) == "S"
    local.assert_called_once_with("bash /tmp/query.sh", query=True)
    scp.assert_called_once_with(
        "worker0", "/tmp/query.sh", config.train.system.logging.scripts_dir, 2222
    )
    assert ssh.call_args_list[-1].args == ("worker0", "bash /tmp/query.sh", 2222)
    assert ssh.call_args_list[-1].kwargs == {"query": True}


def test_ssh_train_run_local_without_hostfile_starts_and_stops_tail(tmp_path, mocker):
    config = _train_config(tmp_path, runner_extra={"master_port": 2345})
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config
    runner.resources = None
    runner.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1"}
    stop = mocker.Mock()
    mocker.patch(
        "flagscale.runner.runner_train.start_tail_log", return_value=(object(), stop)
    )
    run_each = mocker.patch.object(runner, "_run_each")

    assert runner.run(background=True, dryrun=False, enable_monitoring=True) is None

    run_each.assert_called_once_with(
        "localhost",
        "localhost",
        2345,
        1,
        0,
        2,
        background=True,
        dryrun=False,
        cur_envs=runner.user_envs,
        enable_monitoring=True,
    )
    stop.set.assert_called_once()


def test_ssh_train_run_multinode_builds_pool_tasks(tmp_path, mocker):
    config = _train_config(tmp_path, runner_extra={"nnodes": 2})
    runner = object.__new__(runner_train.SSHTrainRunner)
    runner.config = config
    runner.resources = collections.OrderedDict(
        [
            ("worker0", {"slots": 8, "type": "A100"}),
            ("worker1", {"slots": 4, "type": "A100"}),
        ]
    )
    runner.user_envs = {"ENV": "1"}
    mocker.patch("flagscale.runner.runner_train.get_free_port", return_value=29501)

    class FakePool:
        instances = []

        def __init__(self, processes):
            self.processes = processes
            self.calls = []
            FakePool.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, tasks):
            self.calls.append((func, tasks))

    mocker.patch("flagscale.runner.runner_train._MAX_CPU_COUNT", 8)
    mocker.patch("flagscale.runner.runner_train.multiprocessing.Pool", FakePool)

    runner.run(background=True, dryrun=True, enable_monitoring=False)

    pool = FakePool.instances[-1]
    assert pool.processes == 2
    func, tasks = pool.calls[-1]
    assert func is runner_train.run_node
    assert len(tasks) == 2
    assert tasks[0][0] == runner._run_each
    assert tasks[0][2] == "worker0"
    assert tasks[0][6] == 2
    assert tasks[0][7] == "worker0"
    assert tasks[0][8] == 29501


def test_generate_run_and_stop_script_inference(tmp_path, mocker):
    config = _inference_config(tmp_path, runner_extra={"no_shared_fs": True})
    mocker.patch(
        "flagscale.runner.runner_inference.get_pkg_dir", return_value=str(tmp_path)
    )

    run_script = runner_inference._generate_run_script_inference(
        config, "worker0", 0, "python serve.py", background=False
    )
    stop_script = runner_inference._generate_stop_script(config, "worker0", 0)

    run_content = open(run_script, encoding="utf-8").read()
    assert "echo infer-before" in run_content
    assert f"cd {tmp_path}" in run_content
    assert "export PYTHONPATH=" in run_content
    assert "set -o pipefail" in run_content
    assert "host.output" in run_content
    stop_content = open(stop_script, encoding="utf-8").read()
    assert "pkill -f 'python'" in stop_content
    assert "echo infer-after" in stop_content


def test_ssh_inference_run_each_remote_and_local(tmp_path, mocker):
    config = _inference_config(tmp_path, runner_extra={"no_shared_fs": True})
    mocker.patch(
        "flagscale.runner.runner_inference.get_pkg_dir", return_value=str(tmp_path)
    )
    runner = object.__new__(runner_inference.SSHInferenceRunner)
    runner.config = config
    runner.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1"}
    runner.user_script = "serve.py"
    runner.user_args = ["--config-path=/tmp/inference.yaml"]
    run_local = mocker.patch("flagscale.runner.runner_inference.run_local_command")
    run_ssh = mocker.patch("flagscale.runner.runner_inference.run_ssh_command")
    run_scp = mocker.patch("flagscale.runner.runner_inference.run_scp_command")

    runner._run_each(
        "localhost", "localhost", 29500, 1, 0, 2, background=False, dryrun=True
    )
    runner._run_each("worker0", "worker0", 29500, 1, 0, 2, background=True, dryrun=True)

    local_script = run_local.call_args.args[0].removeprefix("bash ")
    assert "CUDA_VISIBLE_DEVICES=0,1" in open(local_script, encoding="utf-8").read()
    assert run_local.call_args.kwargs == {"stream_output": True}
    assert run_ssh.call_args_list[0].args == (
        "worker0",
        f"mkdir -p {config.inference.logging.scripts_dir}",
        2222,
        True,
    )
    run_scp.assert_called_once()


def test_ssh_inference_run_hostfile_and_stop_paths(tmp_path, mocker):
    config = _inference_config(tmp_path, runner_extra={"nnodes": 1})
    runner = object.__new__(runner_inference.SSHInferenceRunner)
    runner.config = config
    runner.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1,2"}
    runner.resources = collections.OrderedDict(
        [
            ("worker0", {"slots": 8, "type": None}),
            ("worker1", {"slots": 8, "type": None}),
        ]
    )
    stop = mocker.Mock()
    mocker.patch(
        "flagscale.runner.runner_inference.start_tail_log",
        return_value=(object(), stop),
    )
    mocker.patch("flagscale.runner.runner_inference.get_free_port", return_value=29502)
    run_each = mocker.patch.object(runner, "_run_each")

    runner.run(background=True, dryrun=False)

    run_each.assert_called_once_with(
        "worker0", "worker0", 29502, 1, 0, 2, background=True, dryrun=False
    )
    stop.set.assert_called_once()

    runner.resources = None
    stop_each = mocker.patch.object(runner, "_stop_each")
    runner.stop()
    stop_each.assert_called_once_with("localhost", 0)
