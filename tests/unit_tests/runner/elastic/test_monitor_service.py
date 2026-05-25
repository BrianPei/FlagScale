import os
import subprocess

from omegaconf import OmegaConf

from flagscale.runner.elastic.monitor_service import MonitorService
from flagscale.runner.utils import JobStatus


class FakeRunner:
    def __init__(self, resources=None, statuses=None):
        self.resources = resources
        self.statuses = list(statuses or [JobStatus.RUNNING])

    def _query_status(self):
        return self.statuses.pop(0) if self.statuses else JobStatus.COMPLETED_OR_IDLE


def make_monitor_config(tmp_path, no_shared_fs=False, timeout=60):
    return OmegaConf.create(
        {
            "experiment": {
                "runner": {
                    "no_shared_fs": no_shared_fs,
                    "ssh_port": 2222,
                    "hang_detection_timeout": timeout,
                }
            },
            "train": {
                "system": {
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                    }
                }
            },
        }
    )


def test_start_monitoring_is_idempotent_and_stop_joins_thread(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    runner = FakeRunner()
    service = MonitorService(config, runner, interval=1)

    class FakeThread:
        def __init__(self, target=None, daemon=False):
            nonlocal target_ref, daemon_ref
            target_ref = target
            daemon_ref = daemon
            self.started = False
            self.joined = False

        def start(self):
            self.started = True

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.joined = timeout

    target_ref = None
    daemon_ref = None
    thread_cls = mocker.patch(
        "flagscale.runner.elastic.monitor_service.threading.Thread", FakeThread
    )
    warning = mocker.patch("flagscale.runner.elastic.monitor_service.logger.warning")

    service.start_monitoring()
    first_thread = service.monitor_thread
    service.start_monitoring()
    service.stop()

    assert thread_cls is FakeThread
    assert target_ref == service._monitor_loop
    assert daemon_ref is True
    assert first_thread.started is True
    assert first_thread.joined == 5
    warning.assert_called_once_with("Monitor service is already running")
    assert service.is_running is False


def test_log_status_writes_status_file(tmp_path):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)

    service._log_status(JobStatus.RUNNING)

    status_log = os.path.join(service.monitor_log_dir, "status.log")
    assert os.path.exists(status_log)
    assert "Status: RUNNING" in open(status_log, encoding="utf-8").read()


def test_check_for_manual_kill_writes_diagnostic_on_fast_termination(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    service.last_job_status = JobStatus.RUNNING
    service.process_start_time = 100
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=120)
    write = mocker.patch.object(service, "_write_manual_kill_diagnostic")

    service._check_for_manual_kill(JobStatus.COMPLETED_OR_IDLE)

    write.assert_called_once()
    assert service.last_job_status == JobStatus.COMPLETED_OR_IDLE


def test_check_pid_file_anomaly_detects_dead_process(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    service = MonitorService(config, FakeRunner(), interval=1)
    pid_dir = tmp_path / "logs" / "pids"
    pid_dir.mkdir(parents=True)
    (pid_dir / "host_0_localhost.pid").write_text("12345")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.subprocess.run",
        return_value=subprocess.CompletedProcess(["ps"], 1),
    )

    assert service._check_pid_file_anomaly("localhost", 0) is True


def test_collect_logs_and_diagnostics_route_to_each_resource(tmp_path, mocker):
    resources = {"worker0": {}, "worker1": {}}
    service = MonitorService(
        make_monitor_config(tmp_path), FakeRunner(resources), interval=1
    )
    collect = mocker.patch.object(service, "_collect_logs_for_host")
    diagnostic = mocker.patch.object(service, "_generate_diagnostic_for_host")

    service._collect_logs()
    service._generate_diagnostics()

    assert collect.call_args_list[0].args == ("worker0", 0)
    assert collect.call_args_list[1].args == ("worker1", 1)
    assert diagnostic.call_args_list[0].args == ("worker0", 0)
    assert diagnostic.call_args_list[1].args == ("worker1", 1)


def test_check_log_hang_detects_stale_local_log(tmp_path, mocker):
    config = make_monitor_config(tmp_path, timeout=30)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "host_0_localhost.output"
    log_file.write_text("old log")
    service = MonitorService(config, FakeRunner(), interval=1)
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.os.path.getmtime", return_value=100
    )
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=131)

    assert service._check_log_hang("localhost", 0) is True


def test_check_log_hang_no_shared_fs_uses_remote_mtime(tmp_path, mocker):
    config = make_monitor_config(tmp_path, no_shared_fs=True, timeout=30)
    service = MonitorService(config, FakeRunner(), interval=1)
    remote_mtime = mocker.patch(
        "flagscale.runner.elastic.monitor_service.get_remote_file_mtime",
        return_value=100,
    )
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=120)

    assert service._check_log_hang("worker0", 0) is False
    remote_mtime.assert_called_once_with(
        "worker0",
        os.path.join(config.train.system.logging.log_dir, "host.output"),
        2222,
    )


def test_check_and_report_hang_generates_diagnostic_for_hanging_nodes(tmp_path, mocker):
    service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner({"worker0": {}, "worker1": {}}),
        interval=1,
    )
    check = mocker.patch.object(service, "_check_log_hang", side_effect=[True, False])
    generate = mocker.patch.object(service, "_generate_hang_diagnostic")

    service._check_and_report_hang()

    assert check.call_args_list[0].args == ("worker0", 0)
    assert check.call_args_list[1].args == ("worker1", 1)
    generate.assert_called_once_with("worker0", 0)


def test_monitor_loop_runs_collection_diagnostic_and_stops_on_completed(
    tmp_path, mocker
):
    service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner(statuses=[JobStatus.RUNNING, JobStatus.COMPLETED_OR_IDLE]),
        interval=1,
    )
    service.is_running = True
    log_status = mocker.patch.object(service, "_log_status")
    collect = mocker.patch.object(service, "_collect_logs")
    diagnostic = mocker.patch.object(service, "_generate_diagnostics")
    hang = mocker.patch.object(service, "_check_and_report_hang")
    mocker.patch("flagscale.runner.elastic.monitor_service.time.sleep")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.time.time", side_effect=[0, 0, 1, 1]
    )

    service._monitor_loop()

    assert [call.args[0] for call in log_status.call_args_list] == [
        JobStatus.RUNNING,
        JobStatus.COMPLETED_OR_IDLE,
    ]
    collect.assert_called_once()
    diagnostic.assert_called_once()
    hang.assert_called_once()
    assert service.is_running is False
