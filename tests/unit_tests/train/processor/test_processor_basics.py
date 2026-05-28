import json

import numpy as np
import pytest
import torch

from flagscale.models.configs.types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
)
from flagscale.models.utils.constants import (
    ACTION,
    DONE,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    REWARD,
    TRUNCATED,
)
from flagscale.train.processor.batch_processor import (
    AddBatchDimensionActionStep,
    AddBatchDimensionComplementaryDataStep,
    AddBatchDimensionObservationStep,
    AddBatchDimensionProcessorStep,
)
from flagscale.train.processor.converters import (
    batch_to_transition,
    create_transition,
    identity_transition,
    observation_to_transition,
    policy_action_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    to_tensor,
    transition_to_batch,
    transition_to_observation,
    transition_to_policy_action,
    transition_to_robot_action,
)
from flagscale.train.processor.core import TransitionKey
from flagscale.train.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from flagscale.train.processor.device_processor import (
    DeviceProcessorStep,
    get_safe_torch_device,
)
from flagscale.train.processor.factory import (
    make_default_processors,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    make_default_teleop_action_processor,
)
from flagscale.train.processor.gym_action_processor import (
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
)
from flagscale.train.processor.normalize_processor import (
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    hotswap_stats,
)
from flagscale.train.processor.observation_processor import (
    VanillaObservationProcessorStep,
)
from flagscale.train.processor.pipeline import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    InfoProcessorStep,
    ProcessorMigrationError,
    ProcessorStep,
    ProcessorStepRegistry,
)
from flagscale.train.processor.policy_robot_bridge import (
    PolicyActionToRobotActionProcessorStep,
    RobotActionToPolicyActionProcessorStep,
)
from flagscale.train.processor.rename_processor import (
    RenameObservationsProcessorStep,
    rename_stats,
)
from flagscale.train.processor.tokenizer_processor import TokenizerProcessorStep


class AppendInfoStep(InfoProcessorStep):
    def __init__(self, value="seen"):
        self.value = value
        self.reset_called = False

    def info(self, info):
        info["marker"] = self.value
        return info

    def get_config(self):
        return {"value": self.value}

    def reset(self):
        self.reset_called = True

    def transform_features(self, features):
        return {**features, "transformed": True}


class RegistryOnlyStep(ProcessorStep):
    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features


class StatefulStep(ProcessorStep):
    def __init__(self, scale=1.0):
        self.scale = scale
        self.loaded_state = None

    def __call__(self, transition):
        return transition

    def state_dict(self):
        return {"weight": torch.tensor([self.scale])}

    def load_state_dict(self, state):
        self.loaded_state = state

    def transform_features(self, features):
        return features


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        batch = len(text) if isinstance(text, list) else 1
        max_length = kwargs["max_length"]
        return {
            "input_ids": torch.arange(batch * max_length).reshape(batch, max_length),
            "attention_mask": torch.ones(batch, max_length, dtype=torch.long),
        }


def test_to_tensor_converts_supported_inputs_and_rejects_unknown_types():
    assert to_tensor(torch.tensor([1]), dtype=torch.float64).dtype == torch.float64
    assert to_tensor([1, 2]).shape == (2,)
    assert to_tensor((1, 2), dtype=torch.int64).dtype == torch.int64
    assert to_tensor({"a": 1, "b": None, "c": {"d": 2}})["c"]["d"].item() == 2
    assert to_tensor({}) == {}

    with pytest.raises(TypeError):
        to_tensor(object())


def test_transition_converters_cover_success_and_validation_paths():
    action = {"joint": [1.0]}
    obs = {"camera": "frame"}
    transition = robot_action_observation_to_transition((action, obs))

    assert transition_to_robot_action(transition) == action
    assert transition_to_observation(transition) == obs
    assert robot_action_to_transition(action)[TransitionKey.ACTION] == action
    assert observation_to_transition(obs)[TransitionKey.OBSERVATION] == obs

    tensor_action = torch.ones(2)
    policy_transition = policy_action_to_transition(tensor_action)
    assert torch.equal(transition_to_policy_action(policy_transition), tensor_action)
    assert identity_transition(policy_transition) is policy_transition

    with pytest.raises(ValueError):
        robot_action_observation_to_transition([action, obs])
    with pytest.raises(ValueError):
        robot_action_observation_to_transition((torch.ones(1), obs))
    with pytest.raises(ValueError):
        robot_action_to_transition(torch.ones(1))
    with pytest.raises(ValueError):
        observation_to_transition("bad")
    with pytest.raises(ValueError):
        transition_to_robot_action(create_transition(action=torch.ones(1)))
    with pytest.raises(ValueError):
        transition_to_policy_action(create_transition(action={"bad": 1}))
    with pytest.raises(ValueError):
        transition_to_observation(create_transition(observation=None))


def test_batch_transition_round_trip_extracts_observation_and_complementary_data():
    batch = {
        OBS_STATE: torch.ones(3),
        ACTION: torch.zeros(2),
        REWARD: torch.tensor(1.0),
        DONE: torch.tensor(False),
        TRUNCATED: torch.tensor(True),
        "task": "pick",
        "index": torch.tensor(5),
        "padding_is_pad": torch.tensor(False),
        "info": {"episode": 1},
    }

    transition = batch_to_transition(batch)
    assert transition[TransitionKey.OBSERVATION][OBS_STATE] is batch[OBS_STATE]
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == "pick"

    restored = transition_to_batch(transition)
    assert restored[OBS_STATE] is batch[OBS_STATE]
    assert restored["task"] == "pick"

    with pytest.raises(ValueError):
        batch_to_transition("bad")
    with pytest.raises(ValueError):
        batch_to_transition({ACTION: {"not": "policy-action"}})
    with pytest.raises(ValueError):
        transition_to_batch("bad")


def test_empty_batch_uses_transition_defaults():
    transition = batch_to_transition({})

    assert transition[TransitionKey.OBSERVATION] is None
    assert transition[TransitionKey.ACTION] is None
    assert transition[TransitionKey.REWARD] == 0.0
    assert transition[TransitionKey.DONE] is False
    assert transition[TransitionKey.TRUNCATED] is False
    assert transition[TransitionKey.INFO] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_processor_registry_register_get_unregister_and_duplicate_errors():
    registry_name = "unit_test_identity_step"
    ProcessorStepRegistry.unregister(registry_name)

    decorated = ProcessorStepRegistry.register(registry_name)(RegistryOnlyStep)
    assert decorated is RegistryOnlyStep
    assert ProcessorStepRegistry.get(registry_name) is RegistryOnlyStep
    assert registry_name in ProcessorStepRegistry.list()

    with pytest.raises(ValueError, match="already registered"):
        ProcessorStepRegistry.register(registry_name)(RegistryOnlyStep)

    ProcessorStepRegistry.unregister(registry_name)
    with pytest.raises(KeyError):
        ProcessorStepRegistry.get(registry_name)


def test_data_processor_pipeline_hooks_slicing_processing_and_reset():
    step = AppendInfoStep("ok")
    calls = []
    pipeline = DataProcessorPipeline(
        steps=[step, IdentityProcessorStep()],
        name="demo pipeline",
    )
    pipeline.register_before_step_hook(
        lambda idx, transition: calls.append(("before", idx))
    )
    pipeline.register_after_step_hook(
        lambda idx, transition: calls.append(("after", idx))
    )

    result = pipeline({"info": {"start": True}})

    assert result["info"]["marker"] == "ok"
    assert calls == [("before", 0), ("after", 0), ("before", 1), ("after", 1)]
    assert len(pipeline) == 2
    assert isinstance(pipeline[0], AppendInfoStep)
    assert isinstance(pipeline[:1], DataProcessorPipeline)
    assert "steps=2" in repr(pipeline)
    assert (
        list(pipeline.step_through({"info": {"start": True}}))[-1][TransitionKey.INFO][
            "marker"
        ]
        == "ok"
    )
    assert pipeline.process_info({"x": 1})["marker"] == "ok"
    assert pipeline.transform_features({})["transformed"] is True

    pipeline.reset()
    assert step.reset_called is True

    with pytest.raises(TypeError):
        DataProcessorPipeline(steps=[object()])
    with pytest.raises(ValueError):
        pipeline.unregister_before_step_hook(lambda idx, transition: None)
    with pytest.raises(ValueError):
        pipeline.unregister_after_step_hook(lambda idx, transition: None)


def test_pipeline_save_load_config_validation_and_migration_errors(tmp_path):
    pipeline = DataProcessorPipeline(
        steps=[IdentityProcessorStep()],
        name="Policy Preprocessor",
    )
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")

    config_path = tmp_path / "processor.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["name"] == "Policy Preprocessor"
    assert config["steps"][0]["class"].endswith("IdentityProcessorStep")

    loaded_from_dir = DataProcessorPipeline.from_pretrained(tmp_path, "processor.json")
    loaded_from_file = DataProcessorPipeline.from_pretrained(
        config_path, "ignored.json"
    )
    assert isinstance(loaded_from_dir.steps[0], IdentityProcessorStep)
    assert isinstance(loaded_from_file.steps[0], IdentityProcessorStep)

    assert DataProcessorPipeline._is_processor_config({"steps": []}) is True
    assert DataProcessorPipeline._is_processor_config({"steps": [{}]}) is False
    assert DataProcessorPipeline._is_processor_config({"steps": "bad"}) is False

    with pytest.raises(KeyError, match="Override keys"):
        DataProcessorPipeline.from_pretrained(
            tmp_path, "processor.json", overrides={"missing_step": {}}
        )

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "config.json").write_text(
        json.dumps({"model_type": "old"}), encoding="utf-8"
    )
    assert DataProcessorPipeline._should_suggest_migration(invalid_dir) is True
    with pytest.raises(ProcessorMigrationError):
        DataProcessorPipeline.from_pretrained(invalid_dir, "processor.json")


def test_default_factory_processors_are_identity_pipelines():
    teleop = make_default_teleop_action_processor()
    robot_action = make_default_robot_action_processor()
    robot_obs = make_default_robot_observation_processor()
    all_processors = make_default_processors()

    action = {"joint": 1}
    observation = {"camera": "frame"}
    assert teleop((action, observation)) == action
    assert robot_action((action, observation)) == action
    assert robot_obs(observation) == observation
    assert len(all_processors) == 3
    assert all(len(processor.steps) == 1 for processor in all_processors)
    assert all(
        isinstance(processor.steps[0], IdentityProcessorStep)
        for processor in all_processors
    )


def test_batch_dimension_processor_steps_add_expected_leading_dimensions():
    action = torch.ones(3)
    obs = {
        OBS_STATE: torch.ones(4),
        OBS_IMAGE: torch.ones(3, 8, 8),
        f"{OBS_IMAGES}.cam": torch.ones(3, 4, 4),
        "already_batched": torch.ones(2, 3),
    }
    comp = {"task": "pick", "index": torch.tensor(1), "task_index": torch.tensor(2)}

    assert AddBatchDimensionActionStep().action(action).shape == (1, 3)
    processed_obs = AddBatchDimensionObservationStep().observation(obs)
    assert processed_obs[OBS_STATE].shape == (1, 4)
    assert processed_obs[OBS_IMAGE].shape == (1, 3, 8, 8)
    assert processed_obs[f"{OBS_IMAGES}.cam"].shape == (1, 3, 4, 4)
    processed_comp = AddBatchDimensionComplementaryDataStep().complementary_data(comp)
    assert processed_comp["task"] == ["pick"]
    assert processed_comp["index"].shape == (1,)

    transition = create_transition(
        observation={OBS_STATE: torch.ones(2)},
        action=torch.ones(2),
        complementary_data={"task": "place"},
    )
    processed = AddBatchDimensionProcessorStep()(transition)
    assert processed[TransitionKey.ACTION].shape == (1, 2)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 2)
    assert processed[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["place"]


def test_device_processor_cpu_dtype_config_and_validation(monkeypatch):
    step = DeviceProcessorStep(device="cpu", float_dtype="float64")
    transition = create_transition(
        observation={OBS_STATE: torch.ones(2, dtype=torch.float32), "text": "keep"},
        action=torch.ones(2, dtype=torch.float32),
        reward=torch.tensor(1.0),
        done=torch.tensor(False),
        complementary_data={"index": torch.tensor(1)},
    )

    processed = step(transition)
    assert processed[TransitionKey.ACTION].dtype == torch.float64
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float64
    assert processed[TransitionKey.OBSERVATION]["text"] == "keep"
    assert step.get_config() == {"device": "cpu", "float_dtype": "float64"}
    assert get_safe_torch_device("cpu").type == "cpu"

    with pytest.raises(ValueError, match="Invalid float_dtype"):
        DeviceProcessorStep(device="cpu", float_dtype="bad")
    with pytest.raises(ValueError, match="PolicyAction"):
        step(create_transition(action={"robot": 1}))


def test_rename_observations_processor_and_stats_do_not_mutate_inputs():
    step = RenameObservationsProcessorStep(rename_map={"old": "new"})
    original_obs = {"old": torch.tensor(1), "keep": torch.tensor(2)}

    processed = step.observation(original_obs)
    assert set(processed) == {"new", "keep"}
    assert step.get_config() == {"rename_map": {"old": "new"}}

    stats = {"old": {"mean": [1]}, "keep": None}
    renamed = rename_stats(stats, {"old": "new"})
    renamed["new"]["mean"].append(2)
    assert stats["old"]["mean"] == [1]
    assert renamed["keep"] == {}
    assert rename_stats({}, {"old": "new"}) == {}


def test_pipeline_state_file_round_trip_and_override_merge(tmp_path):
    registry_name = "unit_test_stateful_step"
    ProcessorStepRegistry.unregister(registry_name)
    ProcessorStepRegistry.register(registry_name)(StatefulStep)
    try:
        pipeline = DataProcessorPipeline(
            steps=[StatefulStep(scale=3.0)], name="stateful"
        )
        pipeline.save_pretrained(tmp_path, config_filename="processor.json")

        config = json.loads((tmp_path / "processor.json").read_text(encoding="utf-8"))
        assert config["steps"][0]["registry_name"] == registry_name
        assert config["steps"][0]["state_file"].endswith(".safetensors")

        loaded = DataProcessorPipeline.from_pretrained(
            tmp_path,
            "processor.json",
            overrides={registry_name: {"scale": 9.0}},
        )
        assert isinstance(loaded.steps[0], StatefulStep)
        assert loaded.steps[0].scale == 9.0
        assert torch.equal(loaded.steps[0].loaded_state["weight"], torch.tensor([3.0]))
    finally:
        ProcessorStepRegistry.unregister(registry_name)


def test_pipeline_loading_error_paths_for_config_and_step_resolution(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        DataProcessorPipeline.from_pretrained(tmp_path, "missing.json")

    invalid_config = tmp_path / "invalid_processor.json"
    invalid_config.write_text(
        json.dumps({"steps": [{"registry_name": "not_registered"}]})
    )
    with pytest.raises(ImportError, match="registry"):
        DataProcessorPipeline.from_pretrained(invalid_config, "ignored.json")

    bad_import_config = tmp_path / "bad_import.json"
    bad_import_config.write_text(
        json.dumps({"steps": [{"class": "missing.module.DoesNotExist"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="Failed to load processor step"):
        DataProcessorPipeline.from_pretrained(bad_import_config, "ignored.json")

    bad_init_config = tmp_path / "bad_init.json"
    bad_init_config.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "class": (
                            "flagscale.train.processor.device_processor.DeviceProcessorStep"
                        ),
                        "config": {"device": "cpu", "float_dtype": "invalid"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failed to instantiate"):
        DataProcessorPipeline.from_pretrained(bad_init_config, "ignored.json")


def test_normalizer_and_unnormalizer_cover_modes_state_and_hotswap():
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    stats = {
        OBS_STATE: {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
        ACTION: {"min": [-1.0, 0.0], "max": [1.0, 2.0]},
    }

    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([3.0, 6.0]), "skip": torch.tensor([10.0])},
        action=torch.tensor([0.0, 1.0]),
    )
    normalized = normalizer(transition)

    assert torch.allclose(
        normalized[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([1.0, 1.0])
    )
    assert torch.allclose(normalized[TransitionKey.ACTION], torch.tensor([0.0, 0.0]))
    assert (
        normalizer.get_config()["norm_map"][FeatureType.STATE.value]
        == NormalizationMode.MEAN_STD.value
    )
    assert "observation.state.mean" in normalizer.state_dict()

    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    restored = unnormalizer(normalized)
    assert torch.allclose(
        restored[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([3.0, 6.0])
    )
    assert torch.allclose(restored[TransitionKey.ACTION], torch.tensor([0.0, 1.0]))

    loaded = NormalizerProcessorStep(features=features, norm_map=norm_map)
    loaded.load_state_dict(normalizer.state_dict())
    assert "mean" in loaded.stats[OBS_STATE]

    preserved = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    preserved.load_state_dict({f"{OBS_STATE}.mean": torch.tensor([99.0, 99.0])})
    assert preserved.stats[OBS_STATE]["mean"] == [1.0, 2.0]

    swapped = hotswap_stats(
        DataProcessorPipeline(steps=[normalizer]), {ACTION: stats[ACTION]}
    )
    assert swapped is not normalizer
    assert swapped.steps[0].stats == {ACTION: stats[ACTION]}


def test_normalizer_boundary_modes_and_error_paths():
    features = {
        "q": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }

    quantile = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={"q": {"q01": [0.0], "q99": [10.0]}},
    )
    assert torch.allclose(
        quantile(create_transition(observation={"q": torch.tensor([5.0])}))[
            TransitionKey.OBSERVATION
        ]["q"],
        torch.tensor([0.0]),
    )

    quantile10 = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILE10},
        stats={"q": {"q10": [0.0], "q90": [10.0]}},
    )
    assert torch.allclose(
        quantile10(create_transition(observation={"q": torch.tensor([10.0])}))[
            TransitionKey.OBSERVATION
        ]["q"],
        torch.tensor([1.0]),
    )

    selective = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"q": {"mean": [0.0], "std": [1.0]}},
        normalize_observation_keys={"other"},
    )
    original = torch.tensor([3.0])
    assert (
        selective(create_transition(observation={"q": original}))[
            TransitionKey.OBSERVATION
        ]["q"]
        is original
    )

    missing_mean_std = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"q": {"mean": [0.0]}},
    )
    with pytest.raises(ValueError, match="mean and std"):
        missing_mean_std(create_transition(observation={"q": torch.tensor([1.0])}))

    bad_action = NormalizerProcessorStep(
        features=features,
        norm_map={FeatureType.ACTION: NormalizationMode.MIN_MAX},
        stats={ACTION: {"min": [0.0], "max": [1.0]}},
    )
    with pytest.raises(ValueError, match="PolicyAction"):
        bad_action(create_transition(action={"robot": 1}))


def test_tokenizer_processor_task_extraction_device_and_feature_transform():
    step = TokenizerProcessorStep(tokenizer=FakeTokenizer(), max_length=4)
    transition = create_transition(
        observation={OBS_STATE: torch.ones(2)},
        action=torch.ones(1),
        complementary_data={"task": ["pick", "place"]},
    )

    processed = step(transition)
    assert processed[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].shape == (2, 4)
    assert (
        processed[TransitionKey.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK].dtype
        == torch.bool
    )
    assert step._detect_device(transition).type == "cpu"

    features = {PipelineFeatureType.OBSERVATION: {}}
    transformed = step.transform_features(features)
    assert transformed[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS].shape == (
        4,
    )
    assert step.get_config() == {
        "max_length": 4,
        "task_key": "task",
        "padding_side": "right",
        "padding": "max_length",
        "truncation": True,
    }

    with pytest.raises(ValueError, match="must be provided"):
        TokenizerProcessorStep()
    with pytest.raises(ValueError, match="Complementary data is None"):
        step.get_task({TransitionKey.COMPLEMENTARY_DATA: None})
    with pytest.raises(ValueError, match="Task extracted"):
        step.get_task(create_transition(complementary_data={"task": None}))
    with pytest.raises(ValueError, match="Task cannot be None"):
        step(create_transition(observation={}, complementary_data={"task": 123}))


def test_vanilla_observation_processor_images_states_features_and_errors():
    step = VanillaObservationProcessorStep()
    image = np.ones((4, 5, 3), dtype=np.uint8) * 255
    obs = {
        "pixels": {"front": image},
        "environment_state": np.array([1.0, 2.0], dtype=np.float32),
        "agent_pos": np.array([3.0, 4.0], dtype=np.float32),
    }

    processed = step.observation(obs)
    assert processed[f"{OBS_IMAGES}.front"].shape == (1, 3, 4, 5)
    assert torch.allclose(processed[f"{OBS_IMAGES}.front"].amax(), torch.tensor(1.0))
    assert processed[OBS_ENV_STATE].shape == (1, 2)
    assert processed[OBS_STATE].shape == (1, 2)

    single = step.observation({"pixels": image})
    assert single[OBS_IMAGE].shape == (1, 3, 4, 5)

    features = {
        PipelineFeatureType.OBSERVATION: {
            "pixels": PolicyFeature(FeatureType.VISUAL, (4, 5, 3)),
            "observation.pixels.wrist": PolicyFeature(FeatureType.VISUAL, (4, 5, 3)),
            "agent_pos": PolicyFeature(FeatureType.STATE, (2,)),
            "keep": PolicyFeature(FeatureType.STATE, (1,)),
        }
    }
    transformed = step.transform_features(features)
    assert OBS_IMAGE in transformed[PipelineFeatureType.OBSERVATION]
    assert f"{OBS_IMAGES}.wrist" in transformed[PipelineFeatureType.OBSERVATION]
    assert OBS_STATE in transformed[PipelineFeatureType.OBSERVATION]
    assert "keep" in transformed[PipelineFeatureType.OBSERVATION]

    with pytest.raises(ValueError, match="channel-last"):
        step._process_single_image(np.ones((1, 3, 10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        step._process_single_image(np.ones((4, 5, 3), dtype=np.float32))


def test_action_processors_bridge_numpy_delta_and_robot_formats():
    tensor_action = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    np_action = Torch2NumpyActionProcessorStep().action(tensor_action)
    assert np_action.shape == (4,)
    assert torch.equal(
        Numpy2TorchActionProcessorStep()(create_transition(action=np_action))[
            TransitionKey.ACTION
        ],
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    )

    delta = MapTensorToDeltaActionDictStep().action(tensor_action)
    assert delta == {"delta_x": 1.0, "delta_y": 2.0, "delta_z": 3.0, "gripper": 4.0}
    delta_without_gripper = MapTensorToDeltaActionDictStep(use_gripper=False).action(
        torch.tensor([1.0, 2.0, 3.0])
    )
    assert "gripper" not in delta_without_gripper

    robot = MapDeltaActionToRobotActionStep(position_scale=2.0).action(delta.copy())
    assert robot["enabled"] is True
    assert robot["target_x"] == 2.0
    assert robot["gripper_vel"] == 4.0
    disabled = MapDeltaActionToRobotActionStep(noise_threshold=10.0).action(
        delta.copy()
    )
    assert disabled["enabled"] is False

    to_policy = RobotActionToPolicyActionProcessorStep(motor_names=["m1", "m2"])
    policy = to_policy.action({"m1.pos": 0.5, "m2.pos": 1.5})
    assert torch.equal(policy, torch.tensor([0.5, 1.5]))
    to_robot = PolicyActionToRobotActionProcessorStep(motor_names=["m1", "m2"])
    robot_action = to_robot.action(policy)
    assert set(robot_action) == {"m1.pos", "m2.pos"}
    assert to_policy.get_config() == {"motor_names": ["m1", "m2"]}

    features = {PipelineFeatureType.ACTION: {}}
    assert ACTION in to_policy.transform_features(features)[PipelineFeatureType.ACTION]
    features = {PipelineFeatureType.ACTION: {}}
    assert "m1.pos" in to_robot.transform_features(features)[PipelineFeatureType.ACTION]

    with pytest.raises(TypeError):
        Torch2NumpyActionProcessorStep().action({"bad": 1})
    with pytest.raises(TypeError):
        Numpy2TorchActionProcessorStep()(create_transition(action=torch.ones(1)))
    with pytest.raises(ValueError, match="Only PolicyAction"):
        MapTensorToDeltaActionDictStep().action({"bad": 1})
    with pytest.raises(ValueError, match="2 elements"):
        to_policy.action({"m1.pos": 0.5})
    with pytest.raises(ValueError, match="2 elements"):
        to_robot.action(torch.tensor([1.0]))
