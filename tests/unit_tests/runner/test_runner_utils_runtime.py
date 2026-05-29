import os
import subprocess

import pytest
from omegaconf import OmegaConf

from flagscale.runner import utils


def test_resolve_path_warns_or_raises_for_missing_path(tmp_path, mocker):
    warn = mocker.patch("flagscale.runner.utils.logger.warning")
    missing = tmp_path / "missing"

    assert utils.resolve_path(str(missing), "data.path", check_exists=True) == str(
        missing.resolve()
    )
    warn.assert_called_once()

    with pytest.raises(FileNotFoundError, match="data.path"):
        utils.resolve_path(str(missing), "data.path", raise_missing=True)


def test_setup_exp_and_logging_dirs_populates_paths(tmp_path):
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path / "exp")}})
    exp_dir = utils.setup_exp_dir(config)
    logging = OmegaConf.create({})

    log_dir = utils.setup_logging_dirs(logging, exp_dir, log_subdir="custom_logs")

    assert os.path.isdir(exp_dir)
    assert log_dir == os.path.join(exp_dir, "custom_logs")
    assert logging.scripts_dir == os.path.join(log_dir, "scripts")
    assert logging.pids_dir == os.path.join(log_dir, "pids")


def test_validate_serve_config_accepts_list_and_reports_bad_shapes():
    utils.validate_serve_config(OmegaConf.create({"serve": [{"serve_id": "svc"}]}))

    with pytest.raises(ValueError, match="serve"):
        utils.validate_serve_config(OmegaConf.create({}))
    with pytest.raises(TypeError, match="ListConfig"):
        utils.validate_serve_config(OmegaConf.create({"serve": {"serve_id": "svc"}}))
    with pytest.raises(TypeError, match="index 0"):
        utils.validate_serve_config(OmegaConf.create({"serve": ["bad"]}))
    with pytest.raises(ValueError, match="serve_id"):
        utils.validate_serve_config(OmegaConf.create({"serve": [{"name": "bad"}]}))


class _FakeSocket:
    def __init__(self, connect_result=0, exc=None):
        self.connect_result = connect_result
        self.exc = exc
        self.timeout = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect_ex(self, target):
        if self.exc:
            raise self.exc
        return self.connect_result

    def bind(self, target):
        self.bound = target

    def getsockname(self):
        return ("127.0.0.1", 34567)

    def connect(self, target):
        self.connected = target

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("connect_result", "exc", "expected_status", "message"),
    [
        (0, None, True, "ready"),
        (111, None, False, "waiting for master"),
        (0, TimeoutError(), False, "timeout"),
        (0, RuntimeError("boom"), False, "boom"),
    ],
)
def test_is_ray_master_running_socket_outcomes(
    mocker, connect_result, exc, expected_status, message
):
    mocker.patch(
        "flagscale.runner.utils.socket.socket",
        return_value=_FakeSocket(connect_result=connect_result, exc=exc),
    )

    status, msg = utils.is_ray_master_running("127.0.0.1", port=6379, timeout=1)

    assert status is expected_status
    assert message in msg


def test_wait_for_ray_master_retries_until_ready(mocker):
    check = mocker.patch(
        "flagscale.runner.utils.is_ray_master_running",
        side_effect=[(False, "wait"), (True, "ready")],
    )
    sleep = mocker.patch("flagscale.runner.utils.time.sleep")

    assert utils.wait_for_ray_master("master", max_attempts=3, interval=5) is True
    assert check.call_count == 2
    sleep.assert_called_once_with(5)


def test_get_free_port_uses_ephemeral_socket(mocker):
    fake_socket = _FakeSocket()
    mocker.patch("flagscale.runner.utils.socket.socket", return_value=fake_socket)

    assert utils.get_free_port() == 34567
    assert fake_socket.bound == ("", 0)


def test_get_host_name_or_ip_falls_back_to_udp_probe(mocker):
    fake_socket = _FakeSocket()
    fake_socket.getsockname = lambda: ("10.0.0.5", 12345)
    mocker.patch("flagscale.runner.utils.socket.gethostname", return_value="")
    mocker.patch("flagscale.runner.utils.socket.socket", return_value=fake_socket)

    assert utils.get_host_name_or_ip() == "10.0.0.5"


def test_get_ip_addr_returns_loopback_on_resolution_error(mocker):
    mocker.patch("flagscale.runner.utils.socket.gethostname", side_effect=OSError)

    assert utils.get_ip_addr() == "127.0.0.1"


def test_is_master_uses_air_hostfile_and_ip_matching(monkeypatch, mocker):
    config = OmegaConf.create({"experiment": {"runner": {"nnodes": 2}}})
    monkeypatch.setenv("AIRS_SWITCH", "1")
    monkeypatch.setenv("AIRS_HOSTFILE_PATH", "/tmp/hosts")
    parse = mocker.patch(
        "flagscale.runner.utils.parse_hostfile",
        return_value={"10.0.0.1": {"slots": 8, "type": None}},
    )
    mocker.patch("flagscale.runner.utils.get_ip_addr", return_value="10.0.0.1")

    assert utils.is_master(config) is True
    parse.assert_called_once_with("/tmp/hosts")


def test_is_master_rejects_multinode_without_resources():
    config = OmegaConf.create({"experiment": {"runner": {"nnodes": 2}}})

    with pytest.raises(ValueError, match="multi-node"):
        utils.is_master(config, resources=None)


def test_is_master_compares_hostname_for_named_resource(mocker):
    config = OmegaConf.create({"experiment": {"runner": {"nnodes": 1}}})
    mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        return_value=subprocess.CompletedProcess(
            "hostname", 0, stdout="worker0\n", stderr=""
        ),
    )

    assert utils.is_master(config, resources={"worker0": {"slots": 1}}) is True


def test_run_scp_command_builds_port_and_honors_dryrun(mocker):
    run = mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        return_value=subprocess.CompletedProcess("scp", 0, stdout="", stderr=""),
    )

    assert utils.run_scp_command("host", "src", "dst", port=2222, dryrun=True) is None
    run.assert_not_called()

    utils.run_scp_command("host", "src", "dst", port=2222)
    assert run.call_args.args[0] == "scp -P 2222 -r src host:dst "


def test_run_scp_command_exits_on_nonzero_return(mocker):
    mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        return_value=subprocess.CompletedProcess("scp", 3, stdout="out", stderr="err"),
    )

    with pytest.raises(SystemExit) as exc_info:
        utils.run_scp_command("host", "src", "dst")

    assert exc_info.value.code == 3


def test_start_tail_log_starts_daemon_thread(mocker):
    event = object()
    event_cls = mocker.patch(
        "flagscale.runner.utils.threading.Event", return_value=event
    )
    thread = mocker.Mock()
    thread_cls = mocker.patch(
        "flagscale.runner.utils.threading.Thread", return_value=thread
    )

    assert utils.start_tail_log("/tmp/stdout.log") == (thread, event)
    event_cls.assert_called_once()
    assert thread_cls.call_args.kwargs["target"] is utils.tail_log_to_console
    assert thread_cls.call_args.kwargs["args"] == ("/tmp/stdout.log", event)
    assert thread_cls.call_args.kwargs["daemon"] is True
    thread.start.assert_called_once()


def test_tail_log_to_console_prints_existing_lines(tmp_path, capsys):
    log = tmp_path / "stdout.log"
    log.write_text("line1\n", encoding="utf-8")

    class StopAfterFirstWait:
        def __init__(self):
            self.waits = 0

        def is_set(self):
            return self.waits > 0

        def wait(self, interval):
            self.waits += 1

    utils.tail_log_to_console(str(log), StopAfterFirstWait(), poll_interval=0)

    assert "line1" in capsys.readouterr().out
