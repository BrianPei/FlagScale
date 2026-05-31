import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def image_layout_processor_step(monkeypatch):
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

    source_path = (
        Path(__file__).resolve().parents[4]
        / "flagscale"
        / "serve"
        / "processor"
        / "image_layout_processor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_image_layout_processor_under_test", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImageLayoutProcessorStep


def _make_obs(**image_kwargs):
    obs = {"observation.state": np.array([1.0, 2.0]), "task": "pick"}
    for key, img in image_kwargs.items():
        obs[f"observation.images.{key}"] = img
    return obs


def test_hwc_to_chw_with_batch_and_float_normalization(image_layout_processor_step):
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    step = image_layout_processor_step(
        src_layout="hwc", dst_layout="chw", add_batch_dim=True, to_float=True
    )

    result = step.observation(_make_obs(cam=img))

    converted = result["observation.images.cam"]
    assert converted.shape == (1, 3, 2, 2)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(
        converted[0], img.astype(np.float32).transpose(2, 0, 1) / 255.0
    )
    np.testing.assert_array_equal(result["observation.state"], np.array([1.0, 2.0]))
    assert result["task"] == "pick"


def test_chw_to_hwc_without_batch(image_layout_processor_step):
    img = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    step = image_layout_processor_step(src_layout="chw", dst_layout="hwc")

    result = step.observation(_make_obs(cam=img))

    np.testing.assert_array_equal(
        result["observation.images.cam"], img.transpose(1, 2, 0)
    )


def test_same_layout_keeps_image_shape(image_layout_processor_step):
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    step = image_layout_processor_step(src_layout="hwc", dst_layout="hwc")

    result = step.observation(_make_obs(cam=img))

    np.testing.assert_array_equal(result["observation.images.cam"], img)


def test_warns_but_converts_non_uint8_when_to_float(
    caplog, image_layout_processor_step
):
    fs_logger = logging.getLogger("FlagScale")
    fs_logger.propagate = True
    try:
        img = np.ones((2, 2, 3), dtype=np.float64)
        step = image_layout_processor_step(
            src_layout="hwc", dst_layout="chw", to_float=True
        )
        with caplog.at_level(logging.WARNING, logger="FlagScale"):
            result = step.observation(_make_obs(cam=img))
    finally:
        fs_logger.propagate = False

    assert "expected uint8" in caplog.text
    assert result["observation.images.cam"].dtype == np.float32
    np.testing.assert_allclose(
        result["observation.images.cam"],
        img.astype(np.float32).transpose(2, 0, 1) / 255.0,
    )


def test_skips_non_ndarray_and_wrong_ndim_with_warnings(
    caplog, image_layout_processor_step
):
    fs_logger = logging.getLogger("FlagScale")
    fs_logger.propagate = True
    try:
        wrong_ndim = np.zeros((2, 2), dtype=np.uint8)
        obs = _make_obs(text="not-image", flat=wrong_ndim)
        with caplog.at_level(logging.WARNING, logger="FlagScale"):
            result = image_layout_processor_step().observation(obs)
    finally:
        fs_logger.propagate = False

    assert result["observation.images.text"] == "not-image"
    np.testing.assert_array_equal(result["observation.images.flat"], wrong_ndim)
    assert "expected np.ndarray" in caplog.text
    assert "expected ndim=3" in caplog.text


def test_get_config_and_transform_features(image_layout_processor_step):
    step = image_layout_processor_step(
        src_layout="chw", dst_layout="hwc", add_batch_dim=True, to_float=True
    )
    features = {"input": {}}

    assert step.get_config() == {
        "src_layout": "chw",
        "dst_layout": "hwc",
        "add_batch_dim": True,
        "to_float": True,
    }
    assert step.transform_features(features) is features
