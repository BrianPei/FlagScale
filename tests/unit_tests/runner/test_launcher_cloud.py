import os
import subprocess

from omegaconf import OmegaConf

from flagscale.runner.launcher.launcher_cloud import CloudLauncher
from flagscale.runner.utils import JobStatus


class _Backend:
    def __init__(self):
        self.user_args = ["--config", "conf.yaml"]
        self.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1", "ENV": "value"}
        self.user_script = "train.py"
        self.generated = []

    def generate_run_script(self, config, host, node_rank, cmd, background=False):
        self.generated.append((config, host, node_rank, cmd, background))
        return f"/tmp/host_{node_rank}_{host}_run.sh"

    def generate_stop_script(self, config, host, node_rank):
        self.generated.append(("stop", config, host, node_rank))
        return f"/tmp/host_{node_rank}_{host}_stop.sh"


def _config(tmp_path, runner_extra=None):
    runner = {"nproc_per_node": 8, "master_addr": "1.2.3.4", "master_port": 23456}
    if runner_extra:
        runner.update(runner_extra)
    return OmegaConf.create(
        {
            "experiment": {"runner": runner},
            "logging": {
                "log_dir": str(tmp_path / "logs"),
                "scripts_dir": str(tmp_path / "logs" / "scripts"),
                "pids_dir": str(tmp_path / "logs" / "pids"),
            },
        }
    )


def test_run_each_generates_local_python_command_and_runs_script(tmp_path, mocker):
    config = _config(tmp_path)
    backend = _Backend()
    launcher = CloudLauncher(config, backend)
    run_local = mocker.patch(
        "flagscale.runner.launcher.launcher_cloud.run_local_command"
    )

    launcher._run_each(
        "localhost", "1.2.3.4", 23456, 1, 0, 2, background=False, dryrun=True
    )

    generated = backend.generated[-1]
    assert generated[:3] == (config, "localhost", 0)
    assert generated[4] is False
    cmd = generated[3]
    assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
    assert "ENV=value" in cmd
    assert "python train.py --config conf.yaml" in cmd
    run_local.assert_called_once_with("bash /tmp/host_0_localhost_run.sh", True)


def test_run_uses_visible_devices_and_records_host(tmp_path, mocker):
    config = _config(tmp_path, runner_extra={"nproc_per_node": 8})
    backend = _Backend()
    launcher = CloudLauncher(config, backend)
    run_each = mocker.patch.object(launcher, "_run_each")
    free_port = mocker.patch(
        "flagscale.runner.launcher.launcher_cloud.get_free_port", return_value=34567
    )

    assert launcher.run(background=True, dryrun=False) is None

    free_port.assert_called_once()
    run_each.assert_called_once_with(
        "localhost", "1.2.3.4", 23456, 1, 0, 2, background=True, dryrun=False
    )
    assert launcher.host == "1.2.3.4"


def test_run_defaults_to_free_port_and_single_process(tmp_path, mocker):
    config = _config(tmp_path, runner_extra={"nproc_per_node": None})
    del config.experiment.runner.master_addr
    del config.experiment.runner.master_port
    backend = _Backend()
    backend.user_envs = {}
    launcher = CloudLauncher(config, backend)
    run_each = mocker.patch.object(launcher, "_run_each")
    mocker.patch(
        "flagscale.runner.launcher.launcher_cloud.get_free_port", return_value=34567
    )

    launcher.run(background=False, dryrun=True)

    run_each.assert_called_once_with(
        "localhost", "localhost", 34567, 1, 0, 1, background=False, dryrun=True
    )
    assert launcher.host == "localhost"


def test_stop_generates_and_runs_local_stop_script(tmp_path, mocker):
    backend = _Backend()
    launcher = CloudLauncher(_config(tmp_path), backend)
    run = mocker.patch(
        "flagscale.runner.launcher.launcher_cloud.subprocess.run",
        return_value=subprocess.CompletedProcess("cmd", 0),
    )

    assert launcher.stop() is None

    assert backend.generated[-1][0] == "stop"
    assert run.call_args.args[0] == "bash /tmp/host_0_localhost_stop.sh"
    assert run.call_args.kwargs["shell"] is True


def test_generate_query_script_contains_pid_and_fallback(tmp_path):
    config = _config(tmp_path)
    launcher = CloudLauncher(config, _Backend())

    script = launcher._generate_query_script("localhost", 0)

    content = open(script, encoding="utf-8").read()
    assert "ps -p $pid -o state --no-headers" in content
    assert "run_fs_serve_vllm|run_inference_engine" in content
    assert script.endswith("host_0_localhost_query.sh")
    assert os.access(script, os.X_OK)


def test_query_each_returns_stdout_and_handles_errors(tmp_path, mocker):
    launcher = CloudLauncher(_config(tmp_path), _Backend())
    mocker.patch.object(
        launcher, "_generate_query_script", return_value="/tmp/query.sh"
    )
    run_local = mocker.patch(
        "flagscale.runner.launcher.launcher_cloud.run_local_command",
        side_effect=[
            subprocess.CompletedProcess("cmd", 0, stdout="R\n", stderr=""),
            RuntimeError("bad query"),
        ],
    )

    assert launcher._query_each("localhost", 0) == "R"
    assert launcher._query_each("localhost", 0) == ""
    assert run_local.call_args_list[0].args == ("bash /tmp/query.sh",)
    assert run_local.call_args_list[0].kwargs == {"query": True}


def test_query_status_classifies_cloud_statuses(tmp_path, mocker):
    launcher = CloudLauncher(_config(tmp_path), _Backend())

    mocker.patch.object(launcher, "_query_each", return_value="S")
    assert launcher.query() == JobStatus.RUNNING

    launcher._query_each.return_value = "Z"
    assert launcher.query() == JobStatus.COMPLETED_OR_IDLE

    launcher._query_each.return_value = ""
    assert launcher.query() == JobStatus.COMPLETED_OR_IDLE
