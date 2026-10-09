# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import base64
import csv
import hashlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/validate_prepared_runtime.py"
ACTION = ROOT / ".github/actions/restore-cuda-runtime/action.yml"
spec = importlib.util.spec_from_file_location("runtime_validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def record_for(files):
    stream = io.StringIO()
    writer = csv.writer(stream)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", len(data)])
    return stream.getvalue().encode()


def make_runtime(root, kind):
    root.mkdir(parents=True, exist_ok=True)
    if kind == "te":
        files = {
            "transformer_engine/__init__.py": b"# package\n",
            "transformer_engine-1.0.dist-info/METADATA": b"Name: transformer-engine\n",
            "transformer_engine-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        }
        files["transformer_engine-1.0.dist-info/RECORD"] = record_for(files)
        with zipfile.ZipFile(root / "transformer_engine-1.0-py3-none-any.whl", "w") as wheel:
            for name, data in files.items():
                wheel.writestr(name, data)
    else:
        files = {
            f"megatron/core/{part}__init__.py": b"# module\n"
            for part in ("", "tensor_parallel/", "transformer/", "pipeline_parallel/")
        }
        files["megatron_core-1.0.dist-info/RECORD"] = record_for(files)
        for name, data in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


@pytest.mark.parametrize("kind", ["te", "megatron"])
def test_accept_complete_runtime(tmp_path, kind):
    make_runtime(tmp_path, kind)
    validator.validate(kind, tmp_path)


def test_reject_truncated_wheel(tmp_path):
    make_runtime(tmp_path, "te")
    wheel = next(tmp_path.glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes()[:100])
    with pytest.raises(zipfile.BadZipFile):
        validator.validate("te", tmp_path)


@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_reject_incomplete_megatron_with_core_init_present(tmp_path, change):
    make_runtime(tmp_path, "megatron")
    module = tmp_path / "megatron/core/tensor_parallel/__init__.py"
    if change == "missing":
        module.unlink()
    else:
        module.write_text("# changed\n")
    with pytest.raises((OSError, ValueError)):
        validator.validate("megatron", tmp_path)


def test_reject_wheel_with_valid_zip_but_wrong_record(tmp_path):
    make_runtime(tmp_path, "te")
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["transformer_engine/__init__.py"] = b"# changed\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="RECORD mismatch"):
        validator.validate("te", tmp_path)


def test_probe_reports_invalid_and_strict_mode_fails(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "te", str(tmp_path), "--probe", "--output", str(output)],
        capture_output=True,
    )
    assert result.returncode == 0
    assert output.read_text() == "valid=false\n"
    result = subprocess.run([sys.executable, str(SCRIPT), "te", str(tmp_path)], capture_output=True)
    assert result.returncode == 1


@pytest.mark.parametrize(
    "cached,available,success_at,expected_downloads,passed",
    [
        (True, True, None, 0, True),
        (False, True, 1, 1, True),
        (False, True, 2, 2, True),
        (False, True, 3, 3, True),
        (False, True, None, 3, False),
        (False, False, None, 0, False),
    ],
)
@pytest.mark.parametrize("kind", ["te", "megatron"])
def test_action_recovery(tmp_path, cached, available, success_at, expected_downloads, passed, kind):
    """Execute composite shell steps; inject partial/successful artifact downloads."""
    runtime = tmp_path / ("te-fl-wheel" if kind == "te" else "megatron-lm-fl-install")
    runtime.mkdir()
    if cached:
        make_runtime(runtime, kind)
    else:
        (runtime / "partial").write_text("left by failed cache restore")
    context = {
        "inputs.kind": kind,
        "inputs.artifact-name": "test",
        "inputs.artifact-available": str(available).lower(),
    }
    downloads = 0
    final_status = 0
    action = yaml.safe_load(ACTION.read_text())
    for index, step in enumerate(action["runs"]["steps"]):
        if "if" in step:
            clauses = step["if"].split(" && ")

            def matches(clause):
                key, operator, value = clause.split()
                equal = context.get(key, "") == value.strip("'")
                return equal if operator == "==" else not equal

            if not all(matches(clause) for clause in clauses):
                continue
        if "uses" in step:
            downloads += 1
            assert list(runtime.iterdir()) == [], "Every download needs a clean destination"
            if downloads == success_at:
                make_runtime(runtime, kind)
            else:
                (runtime / "partial").write_text("incomplete download")
            continue

        def resolve(value):
            return re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: context[match.group(1)], value)

        output = tmp_path / f"output-{index}"
        env = {
            **os.environ,
            "GITHUB_WORKSPACE": str(tmp_path),
            "GITHUB_ACTION_PATH": str(ACTION.parent),
            "GITHUB_OUTPUT": str(output),
        }
        env.update({key: resolve(value) for key, value in step.get("env", {}).items()})
        # Use the test interpreter for the action's dependency-free validator.
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            final_status = result.returncode
            break
        if "id" in step and output.exists():
            for line in output.read_text().splitlines():
                key, value = line.split("=", 1)
                context[f"steps.{step['id']}.outputs.{key}"] = value
    assert downloads == expected_downloads
    assert (final_status == 0) == passed


def test_installed_bytecode_without_hash_is_allowed(tmp_path):
    make_runtime(tmp_path, "megatron")
    record = next(tmp_path.glob("*.dist-info/RECORD"))
    with record.open("a") as output:
        output.write("megatron/core/__pycache__/__init__.cpython-312.pyc,,\n")
    validator.validate("megatron", tmp_path)
