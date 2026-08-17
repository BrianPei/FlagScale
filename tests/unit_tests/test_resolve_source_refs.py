# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

SCRIPT = Path(__file__).parents[2] / "tools" / "install" / "utils" / "resolve_source_refs.py"
SPEC = importlib.util.spec_from_file_location("resolve_source_refs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_validate_catalog_accepts_supported_policies():
    catalog = {
        "megatron_lm_fl": {
            "repository": "https://github.com/flagos-ai/Megatron-LM-FL.git",
            "policy": "branch",
            "branch": "main",
        },
        "flaggems": {
            "repository": "https://github.com/flagos-ai/FlagGems.git",
            "policy": "latest_release",
        },
    }

    assert MODULE.validate_catalog(catalog) == catalog


def test_repository_source_policy_matches_image_contract():
    root = Path(__file__).parents[2]
    catalog = yaml.safe_load((root / ".github/configs/image_sources.yml").read_text())["sources"]

    assert catalog["flaggems"]["policy"] == "latest_release"
    assert "branch" not in catalog["flaggems"]
    for source_name, source in catalog.items():
        if source_name == "flaggems":
            continue
        assert source["policy"] == "branch"
        assert source["branch"] == "main"


def test_common_image_build_workflow_is_platform_agnostic():
    root = Path(__file__).parents[2]
    workflow = (root / ".github/workflows/build_image_common.yml").read_text().lower()

    for platform in ("cuda", "musa", "ascend", "metax"):
        assert platform not in workflow


@pytest.mark.parametrize("platform", ["cuda", "musa", "ascend", "metax"])
def test_platform_source_refs_use_catalog(platform):
    root = Path(__file__).parents[2]
    catalog = yaml.safe_load((root / ".github/configs/image_sources.yml").read_text())["sources"]
    config = yaml.safe_load((root / f".github/configs/{platform}.yml").read_text())

    assert config["image_build"]["enabled"] is True
    validation_script = config["image_build"].get("validation_script")
    if validation_script:
        validation_path = root / validation_script
        assert validation_path.is_file()
    for task_name, task in config["image_build"]["tasks"].items():
        assert re.fullmatch(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", task["image"])
        if task_name != "all":
            assert (root / f"requirements/{platform}/{task_name}.txt").is_file()
            assert (root / f"tools/install/{platform}/install_{task_name}.sh").is_file()
        build_args = task.get("build_args", {})
        source_refs = task.get("source_refs", {})
        assert build_args.keys().isdisjoint(source_refs)
        assert set(source_refs.values()) <= set(catalog)
        dockerfile = (root / task["dockerfile"]).read_text()
        for build_arg in source_refs:
            assert re.search(rf"^ARG {build_arg}(?:=|$)", dockerfile, re.MULTILINE)


def test_dev_image_name_matches_harbor_governance_contract():
    root = Path(__file__).parents[2]
    workflow = (root / ".github/workflows/build_image_common.yml").read_text()

    assert (
        'candidate="${registry}/flagos-dev/${image_name}:'
        '${short_sha}-${PLATFORM}-dev"' in workflow
    )


@pytest.mark.parametrize("platform", ["cuda", "musa", "ascend", "metax"])
def test_split_runtime_all_images_have_explicit_task_environments(platform):
    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / f".github/configs/{platform}.yml").read_text())
    task = config["image_build"]["tasks"]["all"]

    assert task["test_roles"] == ["train", "inference"]
    assert set(task["test_environments"]) == {"train", "inference", "serve"}
    for role, environment in task["test_environments"].items():
        assert environment["pkg_mgr"] == "runtime"
        assert environment["env_path"] == f"/opt/flagscale/runtimes/{role}"

    dockerfile = (root / task["dockerfile"]).read_text()
    assert "io.flagscale.runtime.layout=task-isolated" in dockerfile
    normalized_dockerfile = dockerfile.replace("\\\n", " ")
    for command in re.split(r"&&|;", normalized_dockerfile):
        if "install.sh" in command and "--task all" in command:
            assert "--no-task" in command
    for role in ("train", "inference", "serve"):
        assert f"io.flagscale.runtime.{role}=/opt/flagscale/runtimes/{role}" in dockerfile

    image_build = config["image_build"]
    smoke_nproc = task.get("runtime_smoke_nproc", image_build.get("runtime_smoke_nproc"))
    device_count = task.get("runtime_device_count", image_build.get("runtime_device_count"))
    if smoke_nproc is not None or device_count is not None:
        assert smoke_nproc is not None and device_count is not None
        assert int(smoke_nproc) <= int(device_count)


@pytest.mark.parametrize("platform", ["cuda", "musa", "ascend", "metax"])
def test_inference_images_include_serve_dependencies(platform):
    root = Path(__file__).parents[2]
    config = yaml.safe_load((root / f".github/configs/{platform}.yml").read_text())
    task = config["image_build"]["tasks"]["inference"]
    env_names = config.get("env_names", {})

    assert task["test_roles"] == ["inference"]
    assert env_names.get("serve", "flagscale-inference") == env_names.get(
        "inference", "flagscale-inference"
    )
    dockerfile = (root / task["dockerfile"]).read_text()
    assert re.search(rf"--platform\s+{platform}\s+--task\s+inference\b", dockerfile)
    assert re.search(rf"--platform\s+{platform}\s+--task\s+serve\b", dockerfile)


def test_validate_catalog_rejects_unknown_policy():
    with pytest.raises(MODULE.SourceResolutionError, match="unsupported policy"):
        MODULE.validate_catalog(
            {
                "dependency": {
                    "repository": "https://github.com/flagos-ai/example.git",
                    "policy": "latest_commit",
                }
            }
        )


@patch.object(MODULE.subprocess, "run")
def test_resolve_branch_returns_head_revision(run):
    revision = "a" * 40
    run.return_value = MagicMock(stdout=f"{revision}\trefs/heads/main\n")

    selector, resolved = MODULE._resolve_branch(
        "https://github.com/flagos-ai/Megatron-LM-FL.git", "main"
    )

    assert selector == "main"
    assert resolved == revision


@patch.object(MODULE, "_latest_release_tag", return_value="v5.3.0")
@patch.object(MODULE.subprocess, "run")
def test_resolve_latest_release_prefers_peeled_tag(run, _latest_release_tag):
    tag_object = "b" * 40
    revision = "c" * 40
    run.return_value = MagicMock(
        stdout=(f"{tag_object}\trefs/tags/v5.3.0\n{revision}\trefs/tags/v5.3.0^{{}}\n")
    )

    selector, resolved = MODULE._resolve_latest_release("https://github.com/flagos-ai/FlagGems.git")

    assert selector == "v5.3.0"
    assert resolved == revision
