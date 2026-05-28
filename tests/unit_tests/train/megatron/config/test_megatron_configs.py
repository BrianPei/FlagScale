"""
Unit tests for flagscale/train/megatron/training/config/*.py

Coverage targets:
- RNGConfig, ProfilingConfig, DistributedInitConfig (common_config.py)
- TrainingConfig, ValidationConfig, SchedulerConfig, LoggerConfig, CheckpointConfig (training_config.py)
- RerunStateMachineConfig, StragglerDetectionConfig (resilience_config.py)

"""

import importlib.util
import os
import pathlib
import unittest
from dataclasses import fields

_CONFIG_DIR = (
    pathlib.Path(__file__).parents[5] / "flagscale/train/megatron/training/config"
)


def _load_module(filename):
    path = _CONFIG_DIR / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""), str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_module("common_config.py")
_training = _load_module("training_config.py")
_resilience = _load_module("resilience_config.py")

RNGConfig = _common.RNGConfig
ProfilingConfig = _common.ProfilingConfig
DistributedInitConfig = _common.DistributedInitConfig

TrainingConfig = _training.TrainingConfig
ValidationConfig = _training.ValidationConfig
SchedulerConfig = _training.SchedulerConfig
LoggerConfig = _training.LoggerConfig

RerunStateMachineConfig = _resilience.RerunStateMachineConfig
StragglerDetectionConfig = _resilience.StragglerDetectionConfig


class TestRNGConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RNGConfig()
        self.assertEqual(cfg.seed, 1234)
        self.assertFalse(cfg.te_rng_tracker)
        self.assertFalse(cfg.inference_rng_tracker)
        self.assertFalse(cfg.data_parallel_random_init)

    def test_custom_seed(self):
        cfg = RNGConfig(seed=42)
        self.assertEqual(cfg.seed, 42)

    def test_enable_all_flags(self):
        cfg = RNGConfig(
            seed=0,
            te_rng_tracker=True,
            inference_rng_tracker=True,
            data_parallel_random_init=True,
        )
        self.assertTrue(cfg.te_rng_tracker)
        self.assertTrue(cfg.inference_rng_tracker)
        self.assertTrue(cfg.data_parallel_random_init)

    def test_kw_only_rejects_positional(self):
        with self.assertRaises(TypeError):
            RNGConfig(9999)  # positional not allowed


class TestProfilingConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = ProfilingConfig()
        self.assertFalse(cfg.use_nsys_profiler)
        self.assertEqual(cfg.profile_step_start, 10)
        self.assertEqual(cfg.profile_step_end, 12)
        self.assertEqual(cfg.profile_ranks, [])
        self.assertEqual(cfg.memory_snapshot_path, "snapshot.pickle")
        self.assertFalse(cfg.record_memory_history)

    def test_custom_step_range(self):
        cfg = ProfilingConfig(profile_step_start=5, profile_step_end=20)
        self.assertEqual(cfg.profile_step_start, 5)
        self.assertEqual(cfg.profile_step_end, 20)

    def test_custom_profile_ranks(self):
        cfg = ProfilingConfig(profile_ranks=[0, 1, 2])
        self.assertEqual(cfg.profile_ranks, [0, 1, 2])

    def test_profile_ranks_default_is_independent_per_instance(self):
        a = ProfilingConfig()
        b = ProfilingConfig()
        a.profile_ranks.append(99)
        self.assertEqual(
            b.profile_ranks, [], "default_factory must create separate list"
        )

    def test_pytorch_profiler_flags(self):
        cfg = ProfilingConfig(
            use_pytorch_profiler=True,
            pytorch_profiler_collect_shapes=True,
            pytorch_profiler_collect_callstack=True,
        )
        self.assertTrue(cfg.use_pytorch_profiler)
        self.assertTrue(cfg.pytorch_profiler_collect_shapes)
        self.assertTrue(cfg.pytorch_profiler_collect_callstack)


class TestDistributedInitConfig(unittest.TestCase):

    def tearDown(self):
        os.environ.pop("LOCAL_RANK", None)

    def test_defaults(self):
        os.environ.pop("LOCAL_RANK", None)
        cfg = DistributedInitConfig()
        self.assertEqual(cfg.distributed_backend, "nccl")
        self.assertEqual(cfg.distributed_timeout_minutes, 10)
        self.assertFalse(cfg.use_megatron_fsdp)
        self.assertFalse(cfg.use_torch_fsdp2)
        self.assertIsNone(cfg.nccl_communicator_config_path)

    def test_local_rank_from_env(self):
        os.environ["LOCAL_RANK"] = "3"
        cfg = DistributedInitConfig()
        self.assertEqual(cfg.local_rank, 3)

    def test_local_rank_default_zero_when_env_unset(self):
        os.environ.pop("LOCAL_RANK", None)
        cfg = DistributedInitConfig()
        self.assertEqual(cfg.local_rank, 0)

    def test_local_rank_override(self):
        os.environ["LOCAL_RANK"] = "7"
        cfg = DistributedInitConfig(local_rank=2)
        # explicit kwarg overrides env
        self.assertEqual(cfg.local_rank, 2)

    def test_valid_backends(self):
        for backend in ("nccl", "gloo", "flagcx"):
            cfg = DistributedInitConfig(distributed_backend=backend)
            self.assertEqual(cfg.distributed_backend, backend)

    def test_nccl_communicator_config_path(self):
        cfg = DistributedInitConfig(nccl_communicator_config_path="/tmp/nccl.yaml")
        self.assertEqual(cfg.nccl_communicator_config_path, "/tmp/nccl.yaml")

    def test_fsdp_flags(self):
        cfg = DistributedInitConfig(use_megatron_fsdp=True)
        self.assertTrue(cfg.use_megatron_fsdp)
        cfg2 = DistributedInitConfig(use_torch_fsdp2=True)
        self.assertTrue(cfg2.use_torch_fsdp2)


class TestTrainingConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = TrainingConfig()
        self.assertIsNone(cfg.micro_batch_size)
        self.assertIsNone(cfg.global_batch_size)
        self.assertIsNone(cfg.rampup_batch_size)
        self.assertIsNone(cfg.train_iters)
        self.assertIsNone(cfg.train_samples)
        self.assertEqual(cfg.iterations_to_skip, [])
        self.assertFalse(cfg.manual_gc)
        self.assertEqual(cfg.manual_gc_interval, 0)
        self.assertTrue(cfg.manual_gc_eval)

    def test_set_train_iters(self):
        cfg = TrainingConfig(micro_batch_size=4, global_batch_size=32, train_iters=1000)
        self.assertEqual(cfg.train_iters, 1000)
        self.assertEqual(cfg.micro_batch_size, 4)

    def test_iterations_to_skip_default_is_independent(self):
        a = TrainingConfig()
        b = TrainingConfig()
        a.iterations_to_skip.append(5)
        self.assertEqual(b.iterations_to_skip, [])

    def test_empty_unused_memory_level_choices(self):
        for level in (0, 1, 2):
            cfg = TrainingConfig(empty_unused_memory_level=level)
            self.assertEqual(cfg.empty_unused_memory_level, level)

    def test_exit_signal_default(self):
        import signal

        cfg = TrainingConfig()
        self.assertEqual(cfg.exit_signal, signal.SIGTERM)

    def test_rampup_batch_size_list(self):
        cfg = TrainingConfig(rampup_batch_size=[16, 8, 300000])
        self.assertEqual(cfg.rampup_batch_size, [16, 8, 300000])


class TestValidationConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = ValidationConfig()
        self.assertEqual(cfg.eval_iters, 100)
        self.assertIsNone(cfg.eval_interval)
        self.assertFalse(cfg.skip_train)
        self.assertFalse(cfg.test_mode)
        self.assertFalse(cfg.full_validation)
        self.assertFalse(cfg.multiple_validation_sets)

    def test_skip_train(self):
        cfg = ValidationConfig(skip_train=True)
        self.assertTrue(cfg.skip_train)

    def test_eval_iters_none(self):
        cfg = ValidationConfig(eval_iters=None)
        self.assertIsNone(cfg.eval_iters)

    def test_custom_eval_interval(self):
        cfg = ValidationConfig(eval_interval=500)
        self.assertEqual(cfg.eval_interval, 500)


class TestSchedulerConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = SchedulerConfig()
        self.assertEqual(cfg.lr_decay_style, "linear")
        self.assertEqual(cfg.lr_warmup_iters, 0)
        self.assertEqual(cfg.lr_warmup_samples, 0)
        self.assertEqual(cfg.lr_warmup_init, 0.0)
        self.assertIsNone(cfg.lr_warmup_fraction)
        self.assertIsNone(cfg.lr_decay_iters)
        self.assertIsNone(cfg.lr_decay_samples)

    def test_init_false_fields_are_none(self):
        """lr_decay_steps, lr_warmup_steps, wd_incr_steps are init=False → always None at creation."""
        cfg = SchedulerConfig()
        self.assertIsNone(cfg.lr_decay_steps)
        self.assertIsNone(cfg.lr_warmup_steps)
        self.assertIsNone(cfg.wd_incr_steps)

    def test_init_false_fields_not_in_init_signature(self):
        """init=False fields must NOT be accepted as constructor kwargs."""
        field_map = {f.name: f for f in fields(SchedulerConfig)}
        for name in (
            "lr_decay_steps",
            "lr_warmup_steps",
            "wd_incr_steps",
            "wsd_decay_steps",
        ):
            self.assertFalse(field_map[name].init, f"{name} should have init=False")

    def test_valid_lr_decay_styles(self):
        for style in ("constant", "linear", "cosine", "inverse-square-root", "WSD"):
            cfg = SchedulerConfig(lr_decay_style=style)
            self.assertEqual(cfg.lr_decay_style, style)

    def test_lr_warmup_fraction(self):
        cfg = SchedulerConfig(lr_warmup_fraction=0.01)
        self.assertAlmostEqual(cfg.lr_warmup_fraction, 0.01)

    def test_weight_decay_incr_style(self):
        for style in ("constant", "linear", "cosine"):
            cfg = SchedulerConfig(weight_decay_incr_style=style)
            self.assertEqual(cfg.weight_decay_incr_style, style)


class TestLoggerConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = LoggerConfig()
        self.assertEqual(cfg.log_interval, 100)
        self.assertEqual(cfg.timing_log_level, 0)
        self.assertEqual(cfg.timing_log_option, "minmax")
        self.assertIsNone(cfg.tensorboard_dir)
        self.assertEqual(cfg.tensorboard_queue_size, 1000)
        self.assertFalse(cfg.log_params_norm)
        self.assertIsNone(cfg.wandb_project)

    def test_valid_timing_log_levels(self):
        for level in (0, 1, 2):
            cfg = LoggerConfig(timing_log_level=level)
            self.assertEqual(cfg.timing_log_level, level)

    def test_tensorboard_dir(self):
        cfg = LoggerConfig(tensorboard_dir="/tmp/tb")
        self.assertEqual(cfg.tensorboard_dir, "/tmp/tb")

    def test_timing_log_options(self):
        for opt in ("max", "minmax", "all"):
            cfg = LoggerConfig(timing_log_option=opt)
            self.assertEqual(cfg.timing_log_option, opt)

    def test_memory_keys(self):
        keys = {"reserved_bytes.all.peak": "peak_reserved"}
        cfg = LoggerConfig(memory_keys=keys)
        self.assertEqual(cfg.memory_keys, keys)


class TestRerunStateMachineConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RerunStateMachineConfig()
        self.assertEqual(cfg.error_injection_rate, 0)
        self.assertEqual(cfg.error_injection_type, "transient_error")
        self.assertEqual(cfg.rerun_mode, "validate_results")
        self.assertTrue(cfg.check_for_nan_in_loss)
        self.assertFalse(cfg.check_for_spiky_loss)

    def test_valid_error_injection_types(self):
        for t in ("correct_result", "transient_error", "persistent_error"):
            cfg = RerunStateMachineConfig(error_injection_type=t)
            self.assertEqual(cfg.error_injection_type, t)

    def test_valid_rerun_modes(self):
        for mode in ("disabled", "validate_results", "report_stats"):
            cfg = RerunStateMachineConfig(rerun_mode=mode)
            self.assertEqual(cfg.rerun_mode, mode)

    def test_disable_nan_check(self):
        cfg = RerunStateMachineConfig(check_for_nan_in_loss=False)
        self.assertFalse(cfg.check_for_nan_in_loss)

    def test_enable_spiky_loss_check(self):
        cfg = RerunStateMachineConfig(check_for_spiky_loss=True)
        self.assertTrue(cfg.check_for_spiky_loss)

    def test_error_injection_rate(self):
        cfg = RerunStateMachineConfig(error_injection_rate=1000)
        self.assertEqual(cfg.error_injection_rate, 1000)


class TestStragglerDetectionConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = StragglerDetectionConfig()
        self.assertFalse(cfg.log_straggler)
        self.assertEqual(cfg.straggler_ctrlr_port, 65535)
        self.assertEqual(cfg.straggler_minmax_count, 1)
        self.assertFalse(cfg.disable_straggler_on_startup)

    def test_enable_logging(self):
        cfg = StragglerDetectionConfig(log_straggler=True)
        self.assertTrue(cfg.log_straggler)

    def test_custom_port(self):
        cfg = StragglerDetectionConfig(straggler_ctrlr_port=8080)
        self.assertEqual(cfg.straggler_ctrlr_port, 8080)

    def test_disable_on_startup(self):
        cfg = StragglerDetectionConfig(disable_straggler_on_startup=True)
        self.assertTrue(cfg.disable_straggler_on_startup)

    def test_minmax_count(self):
        cfg = StragglerDetectionConfig(straggler_minmax_count=4)
        self.assertEqual(cfg.straggler_minmax_count, 4)


if __name__ == "__main__":
    unittest.main()
