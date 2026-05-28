import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from flagscale.models.utils.constants import RNG_STATE
from flagscale.train.utils import random_utils
from flagscale.train.utils.hub import HubMixin
from flagscale.train.utils.io_utils import deserialize_json_into_object


class DummyHubObject(HubMixin):
    def __init__(self):
        self.saved_dirs = []

    def _save_pretrained(self, save_directory: Path) -> None:
        self.saved_dirs.append(save_directory)
        (save_directory / "dummy.txt").write_text("saved", encoding="utf-8")


class FakePlatform:
    def __init__(self, available=False):
        self.available = available
        self.state = torch.tensor([7], dtype=torch.uint8)
        self.restored = None

    def is_available(self):
        return self.available

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.restored = state


def test_deserialize_json_into_object_updates_nested_structures_and_tuples(tmp_path):
    fpath = tmp_path / "config.json"
    fpath.write_text(
        json.dumps({"a": 2, "b": [3, "x"], "c": [4, {"d": False}]}),
        encoding="utf-8",
    )

    target = {"a": 0, "b": [0, ""], "c": (0, {"d": True})}
    result = deserialize_json_into_object(fpath, target)

    assert result == {"a": 2, "b": [3, "x"], "c": (4, {"d": False})}
    assert isinstance(result["c"], tuple)


@pytest.mark.parametrize(
    ("payload", "target", "error"),
    [
        ({"a": 1, "extra": 2}, {"a": 0}, ValueError),
        ({"a": "1"}, {"a": 0}, TypeError),
        ({"a": [1, 2]}, {"a": [0]}, ValueError),
        ({"a": {"bad": 1}}, {"a": (0,)}, TypeError),
    ],
)
def test_deserialize_json_into_object_rejects_shape_and_type_mismatches(
    tmp_path, payload, target, error
):
    fpath = tmp_path / "bad.json"
    fpath.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error):
        deserialize_json_into_object(fpath, target)


def test_rng_serialization_round_trips_python_numpy_and_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(random_utils, "cur_platform", FakePlatform(available=False))

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = random_utils.serialize_rng_state()

    expected_random = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    random_utils.deserialize_rng_state(state)

    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)

    random_utils.save_rng_state(tmp_path)
    assert (tmp_path / RNG_STATE).is_file()
    random_utils.load_rng_state(tmp_path)


def test_torch_rng_state_includes_platform_state_when_available(monkeypatch):
    platform = FakePlatform(available=True)
    monkeypatch.setattr(random_utils, "cur_platform", platform)

    state = random_utils.serialize_torch_rng_state()
    assert torch.equal(state["torch_cuda_rng_state"], platform.state)

    random_utils.deserialize_torch_rng_state(state)
    assert torch.equal(platform.restored, platform.state)


def test_hub_mixin_save_pretrained_local_and_push_to_hub(tmp_path):
    obj = DummyHubObject()

    assert obj.save_pretrained(tmp_path / "local") is None
    assert (tmp_path / "local" / "dummy.txt").read_text(encoding="utf-8") == "saved"

    fake_api = MagicMock()
    fake_api.create_repo.return_value.repo_id = "user/repo"
    fake_api.upload_folder.return_value = "https://huggingface.co/user/repo/commit/1"

    with patch("flagscale.train.utils.hub.HfApi", return_value=fake_api):
        url = obj.push_to_hub(
            "user/repo",
            private=True,
            branch="dev",
            create_pr=True,
            allow_patterns=["*.txt"],
            ignore_patterns=["*.tmp"],
            delete_patterns=["old/*"],
        )

    assert url == "https://huggingface.co/user/repo/commit/1"
    fake_api.create_repo.assert_called_once_with(repo_id="user/repo", private=True, exist_ok=True)
    upload_kwargs = fake_api.upload_folder.call_args.kwargs
    assert upload_kwargs["repo_id"] == "user/repo"
    assert upload_kwargs["commit_message"] == "Upload DummyHubObject"
    assert upload_kwargs["revision"] == "dev"
    assert upload_kwargs["create_pr"] is True


def test_hub_mixin_save_pretrained_push_uses_directory_name_as_default_repo_id(
    tmp_path,
):
    obj = DummyHubObject()

    with patch.object(obj, "push_to_hub", return_value="commit") as push:
        result = obj.save_pretrained(tmp_path / "repo-name", push_to_hub=True)

    assert result == "commit"
    push.assert_called_once()
    assert push.call_args.kwargs["repo_id"] == "repo-name"


def test_hub_mixin_abstract_methods_raise():
    class IncompleteHubObject(HubMixin):
        pass

    with pytest.raises(NotImplementedError):
        IncompleteHubObject()._save_pretrained(Path("unused"))
    with pytest.raises(NotImplementedError):
        IncompleteHubObject.from_pretrained("repo")
