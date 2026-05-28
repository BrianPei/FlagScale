"""
Unit tests for flagscale/train/megatron/training/arguments_fs.py
- _add_hetero_args
- _add_auto_tuner_args
- _add_auto_skip_spiky_loss
- _add_peft_args
- _add_network_size_args
- _add_logging_args
- _add_training_args
- _add_learning_rate_args
- _add_checkpointing_args
- _add_distributed_args
- _add_validation_args
- _add_flagscale_specific_args

"""

import argparse
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Lazy-load: import only the _add_* standalone functions
# ---------------------------------------------------------------------------


def _load_arguments_fs():
    src_path = (
        pathlib.Path(__file__).parents[5]
        / "flagscale/train/megatron/training/arguments_fs.py"
    )

    # Stub out heavy deps before importing
    _stub_modules = {}

    def _stub(name):
        m = types.ModuleType(name)
        _stub_modules[name] = m
        return m

    megatron = _stub("megatron")
    megatron.core = _stub("megatron.core")
    megatron.core.utils = _stub("megatron.core.utils")
    megatron.core.utils.get_torch_version = lambda: "2.0.0"
    megatron.core.utils.is_torch_min_version = lambda v: True
    megatron.training = _stub("megatron.training")
    megatron.training.arguments = _stub("megatron.training.arguments")
    megatron.training.arguments.parse_args_decorator = lambda f: f

    fake_platform_obj = SimpleNamespace(get_device_capability=lambda: (8, 0))
    megatron.plugin = _stub("megatron.plugin")
    megatron.plugin.platform = _stub("megatron.plugin.platform")
    megatron.plugin.platform.get_platform = lambda: fake_platform_obj

    # torch stub (only the parts used at module level)
    import torch as _real_torch  # torch IS available in unit test env

    _stub_modules["torch"] = _real_torch

    saved = {}
    for name in list(sys.modules):
        if name.startswith("megatron"):
            saved[name] = sys.modules.pop(name)

    for name, mod in _stub_modules.items():
        if name != "torch":
            sys.modules[name] = mod
    # also register sub-modules
    for name, mod in list(_stub_modules.items()):
        sys.modules.setdefault(name, mod)

    try:
        spec = importlib.util.spec_from_file_location(
            "_arguments_fs_stub", str(src_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name in list(sys.modules):
            if name.startswith("megatron"):
                sys.modules.pop(name)
        sys.modules.update(saved)

    return mod


try:
    _fs_mod = _load_arguments_fs()
    _HAS_FS_MOD = True
except Exception as _e:
    _fs_mod = None
    _HAS_FS_MOD = False
    _FS_MOD_ERR = str(_e)


def _fresh_parser():
    return argparse.ArgumentParser(add_help=False)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _dest(arg_string: str) -> str:
    """Convert '--foo-bar' to 'foo_bar' (argparse dest convention)."""
    return arg_string.lstrip("-").replace("-", "_")


def _has_dest(parser, dest):
    return any(a.dest == dest for a in parser._actions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddHeteroArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_hetero_args(p)

    def test_returns_parser(self):
        self.assertIsInstance(self.parser, argparse.ArgumentParser)

    def test_enable_hetero_flag(self):
        args = self.parser.parse_args(["--enable-hetero"])
        self.assertTrue(args.enable_hetero)

    def test_enable_hetero_default_false(self):
        args = self.parser.parse_args([])
        self.assertFalse(args.enable_hetero)

    def test_hetero_device_types(self):
        args = self.parser.parse_args(["--hetero-device-types", "a100", "v100"])
        self.assertEqual(args.hetero_device_types, ["a100", "v100"])

    def test_hetero_device_types_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.hetero_device_types)

    def test_hetero_current_device_type(self):
        args = self.parser.parse_args(["--hetero-current-device-type", "h800"])
        self.assertEqual(args.hetero_current_device_type, "h800")

    def test_hetero_process_meshes(self):
        args = self.parser.parse_args(["--hetero-process-meshes", "1", "1", "4", "1"])
        self.assertEqual(args.hetero_process_meshes, [1, 1, 4, 1])

    def test_hetero_pipeline_layer_split(self):
        args = self.parser.parse_args(["--hetero-pipeline-layer-split", "4", "8"])
        self.assertEqual(args.hetero_pipeline_layer_split, [4, 8])

    def test_hetero_use_cpu_communication(self):
        args = self.parser.parse_args(["--hetero-use-cpu-communication"])
        self.assertTrue(args.hetero_use_cpu_communication)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddAutoTunerArgs(unittest.TestCase):
    def test_auto_tune_flag(self):
        p = _fresh_parser()
        parser = _fs_mod._add_auto_tuner_args(p)
        args = parser.parse_args(["--auto-tune"])
        self.assertTrue(args.auto_tune)

    def test_auto_tune_default_false(self):
        p = _fresh_parser()
        parser = _fs_mod._add_auto_tuner_args(p)
        args = parser.parse_args([])
        self.assertFalse(args.auto_tune)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddAutoSkipSpikyLoss(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_auto_skip_spiky_loss(p)

    def test_flag_present(self):
        args = self.parser.parse_args(["--auto-skip-spiky-loss"])
        self.assertTrue(args.auto_skip_spiky_loss)

    def test_default_false(self):
        args = self.parser.parse_args([])
        self.assertFalse(args.auto_skip_spiky_loss)

    def test_threshold_default(self):
        args = self.parser.parse_args([])
        self.assertAlmostEqual(args.spiky_loss_threshold, 0.2)

    def test_threshold_custom(self):
        args = self.parser.parse_args(["--spiky-loss-threshold", "0.5"])
        self.assertAlmostEqual(args.spiky_loss_threshold, 0.5)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddPeftArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_peft_args(p)

    def test_peft_type_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.peft_type)

    def test_peft_type_lora(self):
        args = self.parser.parse_args(["--peft-type", "lora"])
        self.assertEqual(args.peft_type, "lora")

    def test_lora_dim_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.lora_dim, 8)

    def test_lora_alpha_default(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.lora_alpha, 16)

    def test_lora_dropout_default(self):
        args = self.parser.parse_args([])
        self.assertAlmostEqual(args.lora_dropout, 0.0)

    def test_lora_target_modules_default_has_four(self):
        args = self.parser.parse_args([])
        self.assertEqual(len(args.lora_target_modules), 4)

    def test_lora_target_modules_custom(self):
        args = self.parser.parse_args(["--lora-target-modules", "linear_qkv"])
        self.assertIn("linear_qkv", args.lora_target_modules)

    def test_lora_in_init_default_xavier(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.lora_in_init_method, "xavier")

    def test_lora_out_init_default_zero(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.lora_out_init_method, "zero")

    def test_lora_dropout_position(self):
        args = self.parser.parse_args(["--lora-dropout-position", "post"])
        self.assertEqual(args.lora_dropout_position, "post")


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddNetworkSizeArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_network_size_args(p)

    def test_norm_init_weight_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.norm_init_weight)

    def test_norm_init_weight_set(self):
        args = self.parser.parse_args(["--norm-init-weight", "0.01"])
        self.assertAlmostEqual(args.norm_init_weight, 0.01)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddLoggingArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_logging_args(p)

    def test_has_log_interval(self):
        self.assertTrue(_has_dest(self.parser, "log_interval"))

    def test_log_interval_parses(self):
        args = self.parser.parse_args(["--log-interval", "50"])
        self.assertEqual(args.log_interval, 50)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddTrainingArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_training_args(p)

    def test_has_train_iters(self):
        self.assertTrue(_has_dest(self.parser, "train_iters"))

    def test_train_iters_parses(self):
        args = self.parser.parse_args(["--train-iters", "5000"])
        self.assertEqual(args.train_iters, 5000)

    def test_train_iters_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.train_iters)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddLearningRateArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_learning_rate_args(p)

    def test_lr_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.lr)

    def test_lr_set(self):
        args = self.parser.parse_args(["--lr", "1e-4"])
        self.assertAlmostEqual(args.lr, 1e-4)

    def test_min_lr_default(self):
        args = self.parser.parse_args([])
        # min-lr should have a numeric default (usually 0.0)
        self.assertIsNotNone(args.min_lr)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddCheckpointingArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_checkpointing_args(p)

    def test_has_save(self):
        self.assertTrue(_has_dest(self.parser, "save"))

    def test_save_parses(self):
        args = self.parser.parse_args(["--save", "/tmp/ckpt"])
        self.assertEqual(args.save, "/tmp/ckpt")


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddDistributedArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_distributed_args(p)

    def test_has_local_rank(self):
        self.assertTrue(_has_dest(self.parser, "local_rank"))

    def test_local_rank_default(self):
        args = self.parser.parse_args([])
        self.assertIsNotNone(args.local_rank)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddValidationArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_validation_args(p)

    def test_eval_iters_default(self):
        args = self.parser.parse_args([])
        self.assertIsNotNone(args.eval_iters)

    def test_eval_interval_default_none(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.eval_interval)

    def test_eval_interval_set(self):
        args = self.parser.parse_args(["--eval-interval", "200"])
        self.assertEqual(args.eval_interval, 200)


@unittest.skipUnless(_HAS_FS_MOD, "arguments_fs could not be imported")
class TestAddFlagscaleSpecificArgs(unittest.TestCase):
    def setUp(self):
        p = _fresh_parser()
        self.parser = _fs_mod._add_flagscale_specific_args(p)

    def test_parser_returned(self):
        self.assertIsInstance(self.parser, argparse.ArgumentParser)

    def test_no_crash_on_empty_args(self):
        # Should not raise
        args = self.parser.parse_args([])
        self.assertIsNotNone(args)


if __name__ == "__main__":
    unittest.main()
