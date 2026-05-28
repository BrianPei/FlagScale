import json

import pytest
import torch

from flagscale.models.utils.constants import (
    ACTION,
    DONE,
    OBS_IMAGE,
    OBS_IMAGES,
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
from flagscale.train.processor.pipeline import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    InfoProcessorStep,
    ProcessorMigrationError,
    ProcessorStep,
    ProcessorStepRegistry,
)
from flagscale.train.processor.rename_processor import (
    RenameObservationsProcessorStep,
    rename_stats,
)


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
    pipeline.register_before_step_hook(lambda idx, transition: calls.append(("before", idx)))
    pipeline.register_after_step_hook(lambda idx, transition: calls.append(("after", idx)))

    result = pipeline({"info": {"start": True}})

    assert result["info"]["marker"] == "ok"
    assert calls == [("before", 0), ("after", 0), ("before", 1), ("after", 1)]
    assert len(pipeline) == 2
    assert isinstance(pipeline[0], AppendInfoStep)
    assert isinstance(pipeline[:1], DataProcessorPipeline)
    assert "steps=2" in repr(pipeline)
    assert (
        list(pipeline.step_through({"info": {"start": True}}))[-1][TransitionKey.INFO]["marker"]
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
    loaded_from_file = DataProcessorPipeline.from_pretrained(config_path, "ignored.json")
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
    (invalid_dir / "config.json").write_text(json.dumps({"model_type": "old"}), encoding="utf-8")
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
        isinstance(processor.steps[0], IdentityProcessorStep) for processor in all_processors
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
