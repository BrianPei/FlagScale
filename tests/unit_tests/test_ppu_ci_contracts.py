# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import ast
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "phase,overrides,required,absent",
    [
        ("base", {"FLAGSCALE_INSTALL_BASE": "false"}, [], ["pip install"]),
        (
            "base",
            {"FLAGSCALE_INSTALL_BASE": "false", "FLAGSCALE_PIP_DEPS": "hydra-core"},
            ["install --root-user-action=ignore hydra-core"],
            ["base.txt", "git"],
        ),
        ("train", {"FLAGSCALE_ONLY_PIP": "true"}, ["train.txt"], ["git", "--no-deps"]),
        ("train", {"FLAGSCALE_INSTALL_TASK": "false"}, [], ["pip install", "git"]),
        (
            "train",
            {"FLAGSCALE_INSTALL_TASK": "false", "FLAGSCALE_SRC_DEPS": "megatron-lm"},
            ["Megatron-LM-FL.git", "--no-deps"],
            ["TransformerEngine-FL.git", "train.txt"],
        ),
        (
            "train",
            {"FLAGSCALE_INSTALL_TASK": "false", "FLAGSCALE_SRC_DEPS": "transformer-engine"},
            ["TransformerEngine-FL.git", "--no-deps"],
            ["Megatron-LM-FL.git", "train.txt"],
        ),
        (
            "train",
            {"FLAGSCALE_INSTALL_TASK": "false", "FLAGSCALE_PIP_DEPS": "sentencepiece"},
            ["install --root-user-action=ignore sentencepiece"],
            ["train.txt", "git"],
        ),
    ],
)
def test_ppu_installer_respects_common_phase_controls(tmp_path, phase, overrides, required, absent):
    root = Path(__file__).parents[2]
    result = subprocess.run(
        ["bash", str(root / f"tools/install/ppu/install_{phase}.sh"), "--debug"],
        env={
            **os.environ,
            "FLAGSCALE_PKG_MGR": "pip",
            "FLAGSCALE_DEPS": str(tmp_path / "deps"),
            "FLAGSCALE_MEGATRON_REF": "a" * 40
            if overrides.get("FLAGSCALE_SRC_DEPS") == "megatron-lm"
            else "",
            "FLAGSCALE_TE_REF": "b" * 40
            if overrides.get("FLAGSCALE_SRC_DEPS") == "transformer-engine"
            else "",
            **overrides,
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    for text in required:
        assert text in result.stderr
    for text in absent:
        assert text not in result.stderr
    assert not (tmp_path / "deps").exists()


def test_ppu_runner_selection_and_training_config():
    from hydra import compose, initialize_config_dir

    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "parse_ppu_config", root / "tests/test_utils/runners/parse_config.py"
    )
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    assert parser.get_device_types("ppu") == ["ppu"]
    cases = parser.get_functional_tests("ppu", task="train")["train"]["qwen3"]
    unit = parser.get_unit_tests_config("ppu")
    platform = yaml.safe_load((root / ".github/configs/ppu.yml").read_text())
    assert unit["nproc_per_node"] == platform["test_matrix"]["unit"]["nproc_per_node"]
    assert "tests/unit_tests/train/megatron/test_qwen35_tokenizer.py" not in unit["exclude"]
    runtime = platform["dependencies"]["te_fl"]["runtime"]
    assert "megatron-energon[av_decode]~=7.0" in runtime["pip_packages"]
    with initialize_config_dir(
        config_dir=str(root / "tests/functional_tests/train/qwen3/conf"), version_base=None
    ):
        for case in cases:
            config = compose(config_name=case)
            assert (
                config.train.model.te_fl_prefer
                == runtime["environment"]["TE_FL_PREFER"]
                == "reference"
            )
            assert config.train.data.data_cache_path.startswith("/tmp/")
            assert config.train.data.data_path.endswith("pile_wikipedia_demo/pile_wikipedia_demo")
            assert config.experiment.exp_dir.endswith(case)
            assert config.train.model.attention_backend == "unfused"
            assert config.train.system.distributed_backend == "nccl"
            assert "TE_FL_PREFER" not in config.experiment.envs


@pytest.mark.parametrize("phase", ["base", "train"])
def test_ppu_install_phase_dry_run(tmp_path, phase):
    root = Path(__file__).parents[2]
    deps = tmp_path / "deps"
    result = subprocess.run(
        ["bash", str(root / f"tools/install/ppu/install_{phase}.sh"), "--debug"],
        env={
            **os.environ,
            "FLAGSCALE_PKG_MGR": "pip",
            "FLAGSCALE_DEPS": str(deps),
            "FLAGSCALE_MEGATRON_REF": "a" * 40,
            "FLAGSCALE_TE_REF": "b" * 40,
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert f"requirements/ppu/{phase}.txt" in result.stderr
    assert not deps.exists()
    if phase == "base":
        assert "requirements/ppu/train.txt" not in result.stderr
    else:
        assert "fetch --depth 1" in result.stderr
        assert "--no-build-isolation --no-deps" in result.stderr


def test_ppu_dockerfile_uses_common_installer():
    root = Path(__file__).parents[2]
    dockerfile = (root / "docker/ppu/Dockerfile.train").read_text()
    assert "install.sh --platform ppu --task train" in dockerfile
    assert "bash /opt/flagscale/tools/install/ppu/install_" not in dockerfile
    assert "COPY flagscale" not in dockerfile


def test_ppu_uses_shared_prepared_runtime():
    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / ".github/configs/ppu.yml").read_text())
    jobs = yaml.safe_load((root / ".github/workflows/all_tests.yml").read_text())["jobs"]
    assert (root / config["setup_script"]).is_file()
    assert config["test_matrix"]["unit"]["nproc_per_node"] == 2
    assert config["dependencies"]["megatron_lm_fl"]["enabled"] is True
    assert config["dependencies"]["te_fl"]["enabled"] is True
    assert config["dependencies"]["te_fl"]["build_mode"] == "python"
    assert jobs["ppu_prepare"]["uses"] == "./.github/workflows/prepare_dependencies.yml"
    assert jobs["ppu_tests"]["needs"] == "ppu_prepare"
    for field in (
        "megatron_cache_key",
        "megatron_artifact_available",
        "te_fl_cache_key",
        "te_fl_artifact_available",
    ):
        assert jobs["ppu_tests"]["with"][field] == f"${{{{ needs.ppu_prepare.outputs.{field} }}}}"
    for name in ("probe.py", "run_container.sh", "image_pipeline.sh", "accept_train.sh"):
        assert not (root / "tools/install/ppu" / name).exists()


@pytest.mark.parametrize(
    "phase,nproc,devices,probe_status,expected_status",
    [
        ("pre", "2", "16", 0, 0),
        ("post", "4", "16", 0, 0),
        ("post", "2", "16", 7, 7),
        ("post", "invalid", "16", 0, 2),
        ("post", "1", "16", 0, 2),
        ("post", "4", "2", 0, 2),
        ("post", "2", "invalid", 0, 2),
        ("unknown", "2", "16", 0, 2),
    ],
)
def test_ppu_image_validation_contract(
    tmp_path, phase, nproc, devices, probe_status, expected_status
):
    root = Path(__file__).parents[2]
    script = tmp_path / "validate_image_build.sh"
    shutil.copyfile(root / "tools/install/ppu/validate_image_build.sh", script)
    # Stub the external processes, leaving the actual hook's dispatch and validation intact.
    for name, body in {
        "docker": (
            'printf "%s\\n" "$1" >> "$DOCKER_CALLS"\n'
            'printf "docker:%s\\n" "$*"\n'
            'if [ "$1" = run ]; then cat > "$SMOKE_FILE"; exit "$PROBE_STATUS"; fi\n'
        ),
        "timeout": 'shift\nexec "$@"\n',
        "yq": 'printf "%s\\n" "--device=/dev/test-ppu"\n',
    }.items():
        stub = tmp_path / name
        stub.write_text("#!/bin/bash\n" + body)
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "IMAGE_BUILD_PHASE": phase,
            "IMAGE_BUILD_TASK": "train",
            "IMAGE_BUILD_BASE_IMAGE": "base:test",
            "IMAGE_BUILD_CANDIDATE_IMAGE": "candidate:test",
            "IMAGE_BUILD_RUNTIME_SMOKE_NPROC": nproc,
            "IMAGE_BUILD_RUNTIME_DEVICE_COUNT": devices,
            "PROBE_STATUS": str(probe_status),
            "YQ_BIN": str(tmp_path / "yq"),
            "SMOKE_FILE": str(tmp_path / "smoke.py"),
            "DOCKER_CALLS": str(tmp_path / "docker-calls"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_status, result.stderr
    if expected_status == 2:
        assert "docker:" not in result.stdout
    else:
        image, mode = ("base:test", "device") if phase == "pre" else ("candidate:test", "train")
        assert f"--entrypoint bash {image}" in result.stdout
        assert f"--env EXPECTED_DEVICE_COUNT={devices}" in result.stdout
        assert f"--env SMOKE_NPROC={nproc} --env SMOKE_MODE={mode}" in result.stdout
        assert "--device=/dev/test-ppu" in result.stdout
        assert ("docker:pull base:test" in result.stdout) == (phase == "pre")
        assert (tmp_path / "docker-calls").read_text().splitlines()[-1] == "rm"
        ast.parse((tmp_path / "smoke.py").read_text())
