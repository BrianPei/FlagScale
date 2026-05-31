import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def image_resize_processor_step(monkeypatch):
    constants = types.ModuleType("flagscale.models.utils.constants")
    constants.OBS_IMAGES = "observation.images"
    monkeypatch.setitem(sys.modules, "flagscale.models.utils.constants", constants)

    types_module = types.ModuleType("flagscale.models.configs.types")
    types_module.PipelineFeatureType = str
    types_module.PolicyFeature = object
    monkeypatch.setitem(sys.modules, "flagscale.models.configs.types", types_module)

    train_processor = types.ModuleType("flagscale.train.processor")

    class ObservationProcessorStep:
        pass

    class ProcessorStepRegistry:
        @staticmethod
        def register(name):
            def decorator(cls):
                cls.registry_name = name
                return cls

            return decorator

    train_processor.ObservationProcessorStep = ObservationProcessorStep
    train_processor.ProcessorStepRegistry = ProcessorStepRegistry
    monkeypatch.setitem(sys.modules, "flagscale.train.processor", train_processor)

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        image_module = types.ModuleType("PIL.Image")

        class FakeImage:
            def __init__(self, array):
                self.array = array

            def resize(self, target):
                width, height = target
                channels = self.array.shape[2]
                return FakeImage(
                    np.zeros((height, width, channels), dtype=self.array.dtype)
                )

            def __array__(self, dtype=None, copy=None):
                if dtype is not None:
                    return self.array.astype(
                        dtype, copy=copy if copy is not None else True
                    )
                return self.array.copy() if copy else self.array

        image_module.fromarray = FakeImage
        pil.Image = image_module
        monkeypatch.setitem(sys.modules, "PIL", pil)
        monkeypatch.setitem(sys.modules, "PIL.Image", image_module)

    source_path = (
        Path(__file__).resolve().parents[4]
        / "flagscale"
        / "serve"
        / "processor"
        / "image_resize_processor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_image_resize_processor_under_test", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImageResizeProcessorStep


@pytest.fixture
def step(image_resize_processor_step):
    return image_resize_processor_step(image_size=[128, 128])


def _make_obs(**image_kwargs):
    obs = {"observation.state": np.array([1.0, 2.0])}
    for key, img in image_kwargs.items():
        obs[f"observation.images.{key}"] = img
    return obs


def test_resizes_hwc_uint8(step):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)
    assert result["observation.images.image"].dtype == np.uint8


def test_preserves_non_image_keys(step):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    np.testing.assert_array_equal(result["observation.state"], np.array([1.0, 2.0]))


def test_multiple_images(step):
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    obs = _make_obs(image=img1, wrist_image=img2)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)
    assert result["observation.images.wrist_image"].shape == (128, 128, 3)


def test_skips_none_values(step):
    obs = _make_obs(image=None)
    result = step.observation(obs)
    assert result["observation.images.image"] is None


def test_skips_non_ndarray(step):
    obs = _make_obs(image="not_an_image")
    result = step.observation(obs)
    assert result["observation.images.image"] == "not_an_image"


def test_warns_on_wrong_ndim(step, caplog):
    import logging

    fs_logger = logging.getLogger("FlagScale")
    fs_logger.propagate = True
    try:
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        obs = _make_obs(image=img)
        with caplog.at_level(logging.WARNING, logger="FlagScale"):
            result = step.observation(obs)
        assert result["observation.images.image"].shape == (480, 640)
        assert "ndim=2" in caplog.text
    finally:
        fs_logger.propagate = False


def test_noop_when_already_target_size(step):
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)


def test_default_image_size(image_resize_processor_step):
    step = image_resize_processor_step()
    assert step.image_size == [224, 224]


def test_get_config(step):
    assert step.get_config() == {"image_size": [128, 128]}


def test_ignores_non_image_observation_keys(step):
    obs = {
        "observation.state": np.array([1.0]),
        "observation.images.cam": np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        ),
        "task": "pick up the cup",
    }
    result = step.observation(obs)
    assert result["observation.images.cam"].shape == (128, 128, 3)
    assert result["task"] == "pick up the cup"
    np.testing.assert_array_equal(result["observation.state"], np.array([1.0]))
