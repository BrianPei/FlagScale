import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from flagscale.runner.backend import backend_native_compress, backend_verl
from flagscale.runner.backend.backend_megatron import MegatronBackend
from flagscale.runner.backend.backend_native_compress import NativeCompressBackend
from flagscale.runner.backend.backend_verl import VerlBackend


def _megatron_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "exp"),
                "task": {
                    "type": "train",
                    "backend": "megatron",
                    "entrypoint": "train.py",
                },
                "runner": {"hostfile": None, "no_shared_fs": False, "ssh_port": 2222},
                "envs": {"CUDA_VISIBLE_DEVICES": "0,1"},
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
                "model": {},
                "data": {},
            },
        }
    )


def _compress_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "compress-exp"),
                "task": {
                    "type": "compress",
                    "backend": "native_compress",
                    "entrypoint": "compress.py",
                },
                "runner": {"hostfile": None, "no_shared_fs": False},
                "envs": {"ENV": "1"},
                "cmds": {
                    "before_start": "echo compress-before",
                    "after_stop": "echo compress-after",
                },
            },
            "compress": {
                "system": {
                    "save_dir": str(tmp_path / "save"),
                    "logging": {
                        "log_dir": str(tmp_path / "compress-logs"),
                        "scripts_dir": str(tmp_path / "compress-logs" / "scripts"),
                        "pids_dir": str(tmp_path / "compress-logs" / "pids"),
                        "tensorboard_dir": str(tmp_path / "tensorboard"),
                        "wandb_save_dir": str(tmp_path / "wandb"),
                    },
                }
            },
        }
    )


def _verl_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "rl-exp"),
                "task": {"type": "rl", "backend": "verl", "entrypoint": "verl.py"},
                "runner": {
                    "hostfile": None,
                    "no_shared_fs": False,
                    "ray_port": 6380,
                    "ray_dashboard_port": 8266,
                },
                "envs": {"ENV": "1"},
                "cmds": {
                    "before_start": "echo rl-before",
                    "after_stop": "echo rl-after",
                },
            },
            "rl": {
                "config-path": "conf",
                "config-name": "ppo",
                "trainer": {"n_gpus_per_node": 8},
                "append_kargs": {"data.train_files": ["a", "b"]},
            },
            "system": {
                "logging": {
                    "log_dir": str(tmp_path / "rl-logs"),
                    "scripts_dir": str(tmp_path / "rl-logs" / "scripts"),
                    "pids_dir": str(tmp_path / "rl-logs" / "pids"),
                }
            },
        }
    )


def test_megatron_backend_prepare_sets_runtime_fields(tmp_path, mocker):
    config = _megatron_config(tmp_path)
    update = mocker.patch(
        "flagscale.runner.backend.backend_megatron._update_config_train"
    )
    get_args = mocker.patch(
        "flagscale.runner.backend.backend_megatron._get_args_megatron",
        return_value=["--num-layers", "2"],
    )
    parse = mocker.patch(
        "flagscale.runner.backend.backend_megatron.parse_hostfile",
        return_value={"worker0": {"slots": 8, "type": "A100"}},
    )

    backend = MegatronBackend(config)

    assert backend.task_type == "train"
    assert backend.user_args == ["--num-layers", "2"]
    assert backend.user_envs == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert backend.user_script == "train.py"
    assert backend.resources == {"worker0": {"slots": 8, "type": "A100"}}
    assert backend.rdzv_id
    update.assert_called_once_with(config)
    get_args.assert_called_once_with(config)
    parse.assert_called_once_with(None)


def test_megatron_backend_rejects_wrong_task_type(tmp_path):
    config = _megatron_config(tmp_path)
    config.experiment.task.type = "compress"

    with pytest.raises(AssertionError, match="Unsupported task type"):
        MegatronBackend(config)


def test_megatron_backend_generate_run_and_stop_scripts(tmp_path, mocker):
    config = _megatron_config(tmp_path)
    backend = object.__new__(MegatronBackend)
    backend.config = config
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()

    script = MegatronBackend.generate_run_script(
        backend,
        config,
        "worker0",
        0,
        "torchrun train.py",
        background=True,
        pkg_dir=str(pkg_dir),
        enable_monitoring=True,
    )
    stop_script = backend.generate_stop_script("worker0", 0)

    content = open(script, encoding="utf-8").read()
    assert "echo before" in content
    assert f"cd {pkg_dir}" in content
    assert "flagscale/train" in content
    assert "monitor_launcher.py" in content
    assert "nohup" in content
    assert "host_0_worker0.output" in content
    stop_content = open(stop_script, encoding="utf-8").read()
    assert "pkill -f 'torchrun'" in stop_content
    assert "echo after" in stop_content


def test_native_compress_helpers_update_config_and_args(tmp_path, mocker):
    config = _compress_config(tmp_path)
    hydra_output = tmp_path / "hydra" / ".hydra"
    hydra_output.mkdir(parents=True)
    config_file = hydra_output / "config.yaml"
    config_file.write_text("x: 1", encoding="utf-8")
    mocker.patch(
        "flagscale.runner.backend.backend_native_compress.HydraConfig.get",
        return_value=SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(tmp_path / "hydra")),
            output_subdir=".hydra",
        ),
    )

    assert backend_native_compress._get_args_llmcompressor(config) == [
        f"--config-path={config_file}"
    ]

    config.compress.system.logging = {}
    backend_native_compress._update_config_compress(config)
    assert config.compress.system.logging.log_dir.endswith("compress_logs")
    assert config.compress.system.logging.tensorboard_dir.endswith("tensorboard")
    assert config.compress.system.logging.wandb_save_dir.endswith("wandb")


def test_native_compress_backend_prepare_and_scripts(tmp_path, mocker):
    config = _compress_config(tmp_path)
    mocker.patch(
        "flagscale.runner.backend.backend_native_compress._update_config_compress"
    )
    mocker.patch(
        "flagscale.runner.backend.backend_native_compress._get_args_llmcompressor",
        return_value=["--config-path=/tmp/config.yaml"],
    )
    mocker.patch(
        "flagscale.runner.backend.backend_native_compress.parse_hostfile",
        return_value=None,
    )
    mocker.patch(
        "flagscale.runner.backend.backend_native_compress.get_pkg_dir",
        return_value=str(tmp_path),
    )

    backend = NativeCompressBackend(config)
    assert backend.task_type == "compress"
    assert backend.user_args == ["--config-path=/tmp/config.yaml"]
    assert backend.cur_envs is None

    run_script = backend.generate_run_script(
        config, "localhost", 0, "python compress.py"
    )
    stop_script = backend.generate_stop_script(config, "localhost", 0)
    run_content = open(run_script, encoding="utf-8").read()
    assert "echo compress-before" in run_content
    assert f"mkdir -p {config.compress.system.save_dir}" in run_content
    assert "flagscale/compress" in run_content
    assert "set -o pipefail" in run_content
    assert "pkill -f 'python'" in open(stop_script, encoding="utf-8").read()


def test_verl_args_and_update_config(tmp_path):
    config = _verl_config(tmp_path)

    args = backend_verl._get_args_verl(config)

    assert "--config-path=conf" in args
    assert "--config-name=ppo" in args
    assert "trainer.n_gpus_per_node=8" in args
    assert '+data.train_files=["a", "b"]' in args

    config.system = {}
    backend_verl._update_config_rl(config)
    assert config.system.logging.log_dir.endswith("logs")
    assert config.system.logging.scripts_dir.endswith(os.path.join("logs", "scripts"))


def test_verl_backend_prepare_and_ray_scripts(tmp_path, mocker):
    config = _verl_config(tmp_path)
    mocker.patch("flagscale.runner.backend.backend_verl._update_config_rl")
    mocker.patch(
        "flagscale.runner.backend.backend_verl._get_args_verl",
        return_value=["trainer.x=1"],
    )
    mocker.patch(
        "flagscale.runner.backend.backend_verl.parse_hostfile",
        return_value={"worker0": {"slots": 8}},
    )
    mocker.patch(
        "flagscale.runner.backend.backend_verl.get_pkg_dir", return_value=str(tmp_path)
    )

    backend = VerlBackend(config)
    assert backend.task_type == "rl"
    assert backend.user_args == ["trainer.x=1"]
    assert backend.resources == {"worker0": {"slots": 8}}

    resources = {"worker0": {"slots": 8}, "worker1": {"slots": 4}}
    script = backend.generate_run_script(
        config, "worker0", 0, "python verl.py", background=True, resources=resources
    )
    stop_script = backend.generate_stop_script(config, "worker0", 0)
    content = open(script, encoding="utf-8").read()
    assert "ray start --head --port=6380" in content
    assert "--dashboard-port=8266" in content
    assert "ssh -f -n worker1" in content
    assert "--address=worker0:6380" in content
    assert "nohup" in content
    assert "pkill -f 'torchrun'" in open(stop_script, encoding="utf-8").read()
