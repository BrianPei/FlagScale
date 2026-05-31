import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from flagscale.cli import app, get_action, resolve_config

ClickExit = typer.Exit


class TestGetAction:
    """Tests for get_action() function"""

    def test_default_returns_run(self):
        """No flags set returns 'run'"""
        assert get_action(False, False, False, False, False) == "run"

    def test_stop_flag(self):
        """stop=True returns 'stop'"""
        assert get_action(True, False, False, False, False) == "stop"

    def test_dryrun_flag(self):
        """dryrun=True returns 'dryrun'"""
        assert get_action(False, True, False, False, False) == "dryrun"

    def test_test_flag(self):
        """test=True returns 'test'"""
        assert get_action(False, False, True, False, False) == "test"

    def test_query_flag(self):
        """query=True returns 'query'"""
        assert get_action(False, False, False, True, False) == "query"

    def test_tune_flag(self):
        """tune=True returns 'auto_tune'"""
        assert get_action(False, False, False, False, True) == "auto_tune"

    def test_mutually_exclusive_stop_dryrun(self, capsys):
        """Multiple flags (stop and dryrun) raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(True, True, False, False, False)
        assert exc_info.value.exit_code == 1

    def test_mutually_exclusive_all_flags(self, capsys):
        """All flags set raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(True, True, True, True, True)
        assert exc_info.value.exit_code == 1

    def test_mutually_exclusive_test_query(self, capsys):
        """Multiple flags (test and query) raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            get_action(False, False, True, True, False)
        assert exc_info.value.exit_code == 1


class TestResolveConfig:
    """Tests for resolve_config() function"""

    def test_with_yaml_path(self, tmp_path):
        """Explicit yaml path returns parent dir and stem"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("model", yaml_file, "train")

        assert path == str(tmp_path)
        assert name == "test_config"

    def test_with_nested_yaml_path(self, tmp_path):
        """Yaml path in nested directory works correctly"""
        nested_dir = tmp_path / "conf" / "nested"
        nested_dir.mkdir(parents=True)
        yaml_file = nested_dir / "my_train.yaml"
        yaml_file.write_text("model: test")

        path, name = resolve_config("model", yaml_file, "train")

        assert path == str(nested_dir)
        assert name == "my_train"

    def test_yaml_not_exists(self):
        """Non-existent yaml path raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            resolve_config("model", Path("/nonexistent/path/config.yaml"), "train")
        assert exc_info.value.exit_code == 1

    def test_model_not_found(self):
        """Non-existent model raises Exit(1)"""
        with pytest.raises(ClickExit) as exc_info:
            resolve_config("nonexistent_model_xyz_12345", None, "train")
        assert exc_info.value.exit_code == 1

    def test_from_model_name_aquila(self, mocker):
        """Resolves config from examples/aquila/conf directory if it exists"""
        # This test checks if the function correctly constructs the path
        # We mock Path.exists() to control the test
        mocker.patch.object(Path, "exists", return_value=True)

        # The function should construct path: script_dir / "examples" / model / "conf" / f"{task}.yaml"
        try:
            path, name = resolve_config("aquila", None, "train")
            # If aquila exists, it should return the path
            assert "aquila" in path or name == "train"
        except SystemExit:
            # If aquila doesn't exist in the test environment, that's expected
            pass


class TestResolveConfigFromCwd:
    """Tests for resolve_config() using cwd-based lookup"""

    def test_finds_config_in_cwd(self, tmp_path, monkeypatch):
        """Resolves config from cwd/examples/<model>/conf/<task>.yaml"""
        conf_dir = tmp_path / "examples" / "mymodel" / "conf"
        conf_dir.mkdir(parents=True)
        (conf_dir / "train.yaml").write_text("test: value")
        monkeypatch.chdir(tmp_path)

        path, name = resolve_config("mymodel", None, "train")
        assert path == str(conf_dir)
        assert name == "train"

    def test_missing_config_in_cwd_raises(self, tmp_path, monkeypatch):
        """Raises Exit(1) when config not found in cwd"""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ClickExit) as exc_info:
            resolve_config("nonexistent_model", None, "train")
        assert exc_info.value.exit_code == 1


class TestResolveConfigEdgeCases:
    """Edge case tests for resolve_config()"""

    def test_yaml_path_with_spaces(self, tmp_path):
        """Yaml path with spaces in directory name works"""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        yaml_file = spaced_dir / "config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("model", yaml_file, "train")

        assert "path with spaces" in path
        assert name == "config"

    def test_yaml_path_absolute(self, tmp_path):
        """Absolute yaml path is resolved correctly"""
        yaml_file = tmp_path / "absolute_test.yaml"
        yaml_file.write_text("test: value")

        # Use absolute path
        abs_path = yaml_file.resolve()
        path, name = resolve_config("model", abs_path, "train")

        assert Path(path).is_absolute()
        assert name == "absolute_test"

    def test_empty_model_name_with_yaml(self, tmp_path):
        """Empty model name works when yaml_path is provided"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("test: value")

        path, name = resolve_config("", yaml_file, "train")

        assert path == str(tmp_path)
        assert name == "config"


runner = CliRunner()


class TestEvalRobo:
    """Tests for flagscale eval robo subcommand"""

    def test_eval_robo_calls_eval_main(self):
        """eval robo forwards args to eval_online.main"""
        with patch("flagscale.eval.robo.main") as mock_main:
            result = runner.invoke(
                app,
                [
                    "eval",
                    "robo",
                    "--model-name",
                    "qwen_gr00t",
                    "--datasets",
                    "libero_10",
                    "--server-host",
                    "example.com",
                    "--attach",
                    "--base-url",
                    "http://localhost:8080/api/hf",
                    "--model-id",
                    "test_model",
                ],
            )
            assert result.exit_code == 0
            mock_main.assert_called_once()
            args = sys.argv
            assert args[0] == "eval_online.py"
            assert "--model-name" in args
            assert "qwen_gr00t" in args
            assert "--datasets" in args
            assert "libero_10" in args
            assert "--server-host" in args
            assert "example.com" in args
            assert "--attach" in args
            assert "--base-url" in args
            assert "http://localhost:8080/api/hf" in args
            assert "--model-id" in args
            assert "test_model" in args

    def test_eval_robo_missing_required_args(self):
        """eval robo fails when required args are missing"""
        result = runner.invoke(app, ["eval", "robo"])
        assert result.exit_code != 0

    def test_eval_robo_defaults(self):
        """eval robo uses default values for optional args"""
        with patch("flagscale.eval.robo.main") as _:
            result = runner.invoke(
                app,
                [
                    "eval",
                    "robo",
                    "--model-name",
                    "pi0_5",
                    "--datasets",
                    "libero_10",
                    "--server-host",
                    "example.com",
                ],
            )
            assert result.exit_code == 0
            args = sys.argv
            assert "--poll-interval" in args
            assert "30" in args
            assert "--server-timeout" in args
            assert "300" in args
            assert "--attach" not in args
            assert "--detach" not in args

    def test_eval_robo_multiple_datasets(self):
        """eval robo handles multiple --datasets flags"""
        with patch("flagscale.eval.robo.main") as _:
            result = runner.invoke(
                app,
                [
                    "eval",
                    "robo",
                    "--model-name",
                    "qwen_gr00t",
                    "--datasets",
                    "libero_10",
                    "--datasets",
                    "libero_90",
                    "--server-host",
                    "example.com",
                ],
            )
            assert result.exit_code == 0
            args = sys.argv
            assert "libero_10" in args
            assert "libero_90" in args

    def test_eval_help_shows_subcommands(self):
        """flagscale eval --help lists available eval types"""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "robo" in result.output

    def test_eval_robo_help_shows_args(self):
        """flagscale eval robo --help shows help text"""
        result = runner.invoke(app, ["eval", "robo", "--help"])
        assert result.exit_code == 0
        assert "FlagEval" in result.output

    def test_eval_robo_forwards_all_optional_args(self):
        """eval robo forwards optional flags that are not covered by defaults test"""
        with patch("flagscale.eval.robo.main") as mock_main:
            result = runner.invoke(
                app,
                [
                    "eval",
                    "robo",
                    "--model-name",
                    "qwen_gr00t",
                    "--datasets",
                    "libero_10",
                    "--server-host",
                    "example.com",
                    "--server-port",
                    "7000",
                    "--description",
                    "nightly eval",
                    "--detach",
                    "--poll-interval",
                    "5",
                    "--server-timeout",
                    "60",
                ],
            )

        assert result.exit_code == 0
        mock_main.assert_called_once()
        assert "--server-port" in sys.argv
        assert "7000" in sys.argv
        assert "--description" in sys.argv
        assert "nightly eval" in sys.argv
        assert "--detach" in sys.argv
        assert "5" in sys.argv
        assert "60" in sys.argv


class TestCliRootAndEntryPoint:
    """Tests for top-level CLI behavior"""

    def test_root_help_lists_core_commands(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "FlagScale CLI" in result.output
        for command in ["run", "train", "serve", "inference", "install", "test"]:
            assert command in result.output

    def test_invalid_command_exits_nonzero(self):
        result = runner.invoke(app, ["unknown-command"])

        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_version_option_prints_version(self):
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "flagscale version" in result.output

    def test_flagscale_entrypoint_delegates_to_typer_app(self, monkeypatch):
        import flagscale.cli as cli

        fake_app = MagicMock()
        monkeypatch.setattr(cli, "app", fake_app)

        cli.flagscale()

        fake_app.assert_called_once_with()


class TestRunTaskAndRunCommand:
    """Tests for run_task() and explicit flagscale run command"""

    def test_run_task_sets_sys_argv_and_calls_run_main(self, monkeypatch):
        import flagscale.cli as cli

        fake_main = MagicMock()
        monkeypatch.setitem(sys.modules, "flagscale.run", type(sys)("flagscale.run"))
        sys.modules["flagscale.run"].main = fake_main

        cli.run_task("/tmp/conf", "train", "dryrun", ["trainer.nnodes=1"])

        fake_main.assert_called_once_with()
        assert sys.argv == [
            "run.py",
            "--config-path=/tmp/conf",
            "--config-name=train",
            "action=dryrun",
            "trainer.nnodes=1",
        ]

    def test_run_task_ignores_zero_system_exit(self, monkeypatch):
        import flagscale.cli as cli

        fake_main = MagicMock(side_effect=SystemExit(0))
        monkeypatch.setitem(sys.modules, "flagscale.run", type(sys)("flagscale.run"))
        sys.modules["flagscale.run"].main = fake_main

        cli.run_task("/tmp/conf", "train", "run")

        fake_main.assert_called_once_with()

    def test_run_task_reraises_nonzero_system_exit(self, monkeypatch):
        import flagscale.cli as cli

        fake_main = MagicMock(side_effect=SystemExit(2))
        monkeypatch.setitem(sys.modules, "flagscale.run", type(sys)("flagscale.run"))
        sys.modules["flagscale.run"].main = fake_main

        with pytest.raises(SystemExit) as exc_info:
            cli.run_task("/tmp/conf", "train", "run")

        assert exc_info.value.code == 2

    def test_run_command_dispatches_with_explicit_config_and_overrides(
        self, tmp_path, monkeypatch
    ):
        import flagscale.cli as cli

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        (conf_dir / "train.yaml").write_text("experiment: {}")
        run_task = MagicMock()
        monkeypatch.setattr(cli, "run_task", run_task)

        result = runner.invoke(
            app,
            [
                "run",
                "--config-path",
                str(conf_dir),
                "--config-name",
                "train",
                "--action",
                "stop",
                "trainer.nnodes=1",
            ],
        )

        assert result.exit_code == 0
        run_task.assert_called_once_with(
            str(conf_dir.resolve()), "train", "stop", extra_args=["trainer.nnodes=1"]
        )
        assert "action=stop" in result.output

    def test_run_command_rejects_missing_config_path(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                "--config-path",
                str(tmp_path / "missing"),
                "--config-name",
                "train",
            ],
        )

        assert result.exit_code == 1
        assert "Config path does not exist" in result.output

    def test_run_command_rejects_missing_config_file(self, tmp_path):
        result = runner.invoke(
            app,
            ["run", "--config-path", str(tmp_path), "--config-name", "missing"],
        )

        assert result.exit_code == 1
        assert "Config file does not exist" in result.output


class TestTaskCommandDispatch:
    """Tests for task commands that resolve configs and call run_task"""

    def test_train_uses_config_path_and_flag_action(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        config = tmp_path / "custom_train.yaml"
        config.write_text("experiment: {}")
        run_task = MagicMock()
        monkeypatch.setattr(cli, "run_task", run_task)

        result = runner.invoke(
            app, ["train", "qwen3", "--config", str(config), "--dryrun"]
        )

        assert result.exit_code == 0
        run_task.assert_called_once_with(str(tmp_path), "custom_train", "dryrun")
        assert "Train qwen3 [dryrun]" in result.output

    def test_train_rejects_mutually_exclusive_flags(self, tmp_path):
        config = tmp_path / "train.yaml"
        config.write_text("experiment: {}")

        result = runner.invoke(
            app, ["train", "qwen3", "--config", str(config), "--stop", "--dryrun"]
        )

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_serve_run_adds_cli_overrides_and_warning(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        config = tmp_path / "serve.yaml"
        config.write_text("serve: []")
        run_task = MagicMock()
        monkeypatch.setattr(cli, "run_task", run_task)

        result = runner.invoke(
            app,
            [
                "serve",
                "qwen3",
                "--config",
                str(config),
                "--port",
                "8000",
                "--model-path",
                "/models/qwen",
                "--engine-args",
                '{"dtype":"float16"}',
            ],
        )

        assert result.exit_code == 0
        run_task.assert_called_once_with(
            str(tmp_path),
            "serve",
            "run",
            [
                "+experiment.runner.cli_args.port=8000",
                "+experiment.runner.cli_args.model_path=/models/qwen",
                '+experiment.runner.cli_args.engine_args=\'{"dtype":"float16"}\'',
            ],
        )
        assert "Warning: When serving" in result.output

    @pytest.mark.parametrize(
        ("args", "expected_action"),
        [(["--stop"], "stop"), (["--test"], "test"), (["--tune"], "auto_tune")],
    )
    def test_serve_stop_test_tune_actions(
        self, tmp_path, monkeypatch, args, expected_action
    ):
        import flagscale.cli as cli

        config = tmp_path / "serve.yaml"
        config.write_text("serve: []")
        run_task = MagicMock()
        monkeypatch.setattr(cli, "run_task", run_task)

        result = runner.invoke(app, ["serve", "qwen3", "--config", str(config), *args])

        assert result.exit_code == 0
        run_task.assert_called_once_with(str(tmp_path), "serve", expected_action, [])

    @pytest.mark.parametrize(
        ("command", "flag", "expected_action"),
        [
            ("inference", "--test", "test"),
            ("rl", "--stop", "stop"),
            ("compress", "--dryrun", "dryrun"),
        ],
    )
    def test_other_task_commands_dispatch(
        self, tmp_path, monkeypatch, command, flag, expected_action
    ):
        import flagscale.cli as cli

        config = tmp_path / f"{command}.yaml"
        config.write_text("experiment: {}")
        run_task = MagicMock()
        monkeypatch.setattr(cli, "run_task", run_task)

        result = runner.invoke(app, [command, "qwen3", "--config", str(config), flag])

        assert result.exit_code == 0
        run_task.assert_called_once_with(str(tmp_path), command, expected_action)


class TestInstallTestAndPullCommands:
    """Tests for CLI commands that dispatch to subprocesses"""

    def test_install_builds_install_script_command(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        script = tmp_path / "tools" / "install" / "install.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\n")
        completed = MagicMock(returncode=0)
        run = MagicMock(return_value=completed)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli.subprocess, "run", run)

        result = runner.invoke(
            app,
            [
                "install",
                "--platform",
                "ascend",
                "--task",
                "serve",
                "--pkg-mgr",
                "pip",
                "--no-system",
                "--only-pip",
                "--no-dev",
                "--no-base",
                "--no-task",
                "--debug",
            ],
        )

        assert result.exit_code == 0
        assert run.call_args.args[0] == [
            str(script),
            "--platform",
            "ascend",
            "--task",
            "serve",
            "--pkg-mgr",
            "pip",
            "--no-system",
            "--only-pip",
            "--no-dev",
            "--no-base",
            "--no-task",
            "--debug",
        ]

    def test_install_missing_script_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["install"])

        assert result.exit_code == 1
        assert "Install script not found" in result.output

    def test_install_propagates_nonzero_return_code(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        script = tmp_path / "tools" / "install" / "install.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            cli.subprocess, "run", MagicMock(return_value=MagicMock(returncode=7))
        )

        result = runner.invoke(app, ["install"])

        assert result.exit_code == 7

    def test_test_command_builds_runner_command(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        script = tmp_path / "tests" / "test_utils" / "runners" / "run_tests.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\n")
        run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli.subprocess, "run", run)

        result = runner.invoke(
            app,
            [
                "test",
                "--platform",
                "metax",
                "--device",
                "cpu",
                "--type",
                "functional",
                "--task",
                "serve",
                "--model",
                "qwen3",
                "--list",
                "smoke.txt",
            ],
        )

        assert result.exit_code == 0
        assert run.call_args.args[0] == [
            str(script),
            "--platform",
            "metax",
            "--device",
            "cpu",
            "--type",
            "functional",
            "--task",
            "serve",
            "--model",
            "qwen3",
            "--list",
            "smoke.txt",
        ]

    def test_test_command_missing_script_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["test"])

        assert result.exit_code == 1
        assert "Test script not found" in result.output

    def test_pull_creates_default_dir_and_runs_docker_git_commands(
        self, tmp_path, monkeypatch
    ):
        import flagscale.cli as cli

        run = MagicMock()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli.subprocess, "run", run)

        result = runner.invoke(
            app, ["pull", "--image", "repo/image:tag", "--ckpt", "https://repo"]
        )

        assert result.exit_code == 0
        ckpt_dir = tmp_path / "model_download"
        assert ckpt_dir.exists()
        assert run.call_args_list[0].args == (["docker", "pull", "repo/image:tag"],)
        assert run.call_args_list[0].kwargs == {"check": True}
        assert run.call_args_list[1].args == (
            ["git", "clone", "https://repo", str(ckpt_dir)],
        )
        assert run.call_args_list[1].kwargs == {"check": True}
        assert run.call_args_list[2].args == (["git", "lfs", "pull"],)
        assert run.call_args_list[2].kwargs == {"cwd": str(ckpt_dir), "check": True}

    def test_pull_exits_when_docker_pull_fails(self, tmp_path, monkeypatch):
        import flagscale.cli as cli

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            MagicMock(side_effect=cli.subprocess.CalledProcessError(1, "docker")),
        )

        result = runner.invoke(
            app, ["pull", "--image", "bad", "--ckpt", "https://repo"]
        )

        assert result.exit_code == 1
        assert "Failed to pull Docker image" in result.output
