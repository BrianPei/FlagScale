# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import os
import shutil
import subprocess
from pathlib import Path

import pytest


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
        "docker": 'printf "docker:%s\\n" "$*"\n',
        "timeout": 'shift\nexec "$@"\n',
        "run_container.sh": 'printf "container:%s\\n" "$*"\nexit "$PROBE_STATUS"\n',
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
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_status, result.stderr
    if expected_status == 2:
        assert "container:" not in result.stdout
        assert "docker:" not in result.stdout
    else:
        image, mode = ("base:test", "device") if phase == "pre" else ("candidate:test", "train")
        assert f"container:{image} env EXPECTED_DEVICE_COUNT={devices}" in result.stdout
        assert f"--nproc-per-node={nproc} tools/install/ppu/probe.py {mode}" in result.stdout
        assert ("docker:pull base:test" in result.stdout) == (phase == "pre")
