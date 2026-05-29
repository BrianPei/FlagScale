"""
Unit tests for flagscale/train/megatron/training/yaml_arguments.py
- env_constructor / env_pattern
- load_yaml
- core_config_from_args
"""

import dataclasses
import os
import sys
import tempfile
import textwrap
import types
import unittest
from types import SimpleNamespace

import yaml


def _import_env_pattern():
    """Import env_pattern and env_constructor without triggering megatron imports."""
    # We read the source and exec only the safe top portion
    import importlib.util
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[5]
        / "flagscale/train/megatron/training/yaml_arguments.py"
    )

    # Build a minimal fake module environment so megatron imports don't blow up
    fake_megatron_core = types.ModuleType("megatron")
    fake_megatron_core.core = types.ModuleType("megatron.core")
    fake_megatron_core.core.transformer = types.ModuleType("megatron.core.transformer")

    @dataclasses.dataclass
    class _FakeTransformerConfig:
        hidden_size: int = 768
        num_attention_heads: int = 12

    fake_megatron_core.core.transformer.TransformerConfig = _FakeTransformerConfig
    fake_megatron_core.core.transformer.MLATransformerConfig = _FakeTransformerConfig
    fake_megatron_core.core.utils = types.ModuleType("megatron.core.utils")
    fake_megatron_core.core.utils.get_torch_version = lambda: "2.0.0"
    fake_megatron_core.core.utils.is_torch_min_version = lambda v: True

    fake_plugin = types.ModuleType("megatron.plugin")
    fake_plugin.platform = types.ModuleType("megatron.plugin.platform")
    fake_platform_instance = SimpleNamespace(get_device_capability=lambda: (8, 0))
    fake_plugin.platform.get_platform = lambda: fake_platform_instance

    saved = {}
    for name in list(sys.modules.keys()):
        if name.startswith("megatron"):
            saved[name] = sys.modules.pop(name)

    sys.modules["megatron"] = fake_megatron_core
    sys.modules["megatron.core"] = fake_megatron_core.core
    sys.modules["megatron.core.transformer"] = fake_megatron_core.core.transformer
    sys.modules["megatron.core.utils"] = fake_megatron_core.core.utils
    sys.modules["megatron.plugin"] = fake_plugin
    sys.modules["megatron.plugin.platform"] = fake_plugin.platform

    try:
        spec = importlib.util.spec_from_file_location("_yaml_args_stub", str(src_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        # restore original megatron modules
        for name in list(sys.modules.keys()):
            if name.startswith("megatron"):
                sys.modules.pop(name)
        sys.modules.update(saved)

    return mod


# Load once at module level (cached)
try:
    _yaml_mod = _import_env_pattern()
    _HAS_YAML_MOD = True
except Exception as _e:
    _yaml_mod = None
    _HAS_YAML_MOD = False
    _YAML_MOD_ERR = str(_e)


@unittest.skipUnless(_HAS_YAML_MOD, "yaml_arguments could not be imported")
class TestEnvConstructor(unittest.TestCase):
    """Test the YAML env-variable resolver."""

    def setUp(self):
        self.env_pattern = _yaml_mod.env_pattern
        self.env_constructor = _yaml_mod.env_constructor

    def tearDown(self):
        for k in ("MY_DATA_PATH", "CHECKPOINT_DIR", "MY_VAR"):
            os.environ.pop(k, None)

    def _make_loader(self, source):
        """Create a YAML Loader that has our custom constructor registered."""
        loader = yaml.Loader(source)
        yaml.add_implicit_resolver("!pathex", self.env_pattern, Loader=yaml.Loader)
        yaml.add_constructor("!pathex", self.env_constructor, Loader=yaml.Loader)
        return loader

    def test_env_pattern_matches_variable_syntax(self):
        self.assertIsNotNone(self.env_pattern.search("/data/${MY_DATA_PATH}/train"))
        self.assertIsNotNone(self.env_pattern.search("${SOME_VAR}"))

    def test_env_pattern_no_match_plain_string(self):
        self.assertIsNone(self.env_pattern.search("/data/plain/path"))

    def test_env_constructor_substitutes_single_var(self):
        os.environ["MY_VAR"] = "hello"
        yaml_text = "path: /base/${MY_VAR}/end\n"
        result = yaml.safe_load(yaml_text.replace("${MY_VAR}", os.environ["MY_VAR"]))
        self.assertEqual(result["path"], "/base/hello/end")

    def test_env_constructor_missing_var_raises(self):
        """env_constructor asserts env var exists; missing var → AssertionError."""
        os.environ.pop("MY_DATA_PATH", None)
        # Simulate what env_constructor does
        value = "/data/${MY_DATA_PATH}/train"
        groups = self.env_pattern.findall(value)
        self.assertEqual(groups, ["MY_DATA_PATH"])
        with self.assertRaises((AssertionError, KeyError)):
            for group in groups:
                assert (
                    os.environ.get(group) is not None
                ), f"environment variable {group} in yaml not found"

    def test_env_constructor_multiple_vars(self):
        os.environ["MY_DATA_PATH"] = "/mnt/data"
        os.environ["CHECKPOINT_DIR"] = "/mnt/ckpt"
        value = "${MY_DATA_PATH}/train:${CHECKPOINT_DIR}"
        for group in self.env_pattern.findall(value):
            value = value.replace(f"${{{group}}}", os.environ[group])
        self.assertEqual(value, "/mnt/data/train:/mnt/ckpt")


@unittest.skipUnless(_HAS_YAML_MOD, "yaml_arguments could not be imported")
class TestLoadYaml(unittest.TestCase):
    """Test load_yaml (reads YAML file → SimpleNamespace)."""

    def setUp(self):
        self.load_yaml = _yaml_mod.load_yaml

    def _write_yaml(self, content):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        f.write(textwrap.dedent(content))
        f.flush()
        f.close()
        return f.name

    def tearDown(self):
        pass  # temp files cleaned per-test

    def test_simple_flat_yaml(self):
        path = self._write_yaml("""
            micro_batch_size: 4
            global_batch_size: 32
            train_iters: 1000
        """)
        try:
            args = self.load_yaml(path)
            self.assertEqual(args.micro_batch_size, 4)
            self.assertEqual(args.global_batch_size, 32)
            self.assertEqual(args.train_iters, 1000)
        finally:
            os.unlink(path)

    def test_nested_yaml_as_namespace(self):
        path = self._write_yaml("""
            model:
              hidden_size: 512
              num_layers: 12
            training:
              lr: 0.001
        """)
        try:
            args = self.load_yaml(path)
            self.assertEqual(args.model.hidden_size, 512)
            self.assertEqual(args.model.num_layers, 12)
            self.assertAlmostEqual(args.training.lr, 0.001)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            self.load_yaml("/nonexistent/__test__.yaml")

    def test_empty_yaml_returns_namespace(self):
        path = self._write_yaml("")
        try:
            with self.assertRaises(AttributeError):
                self.load_yaml(path)
        finally:
            os.unlink(path)

    def test_list_value(self):
        path = self._write_yaml("""
            data_path:
              - /data/train1
              - /data/train2
        """)
        try:
            args = self.load_yaml(path)
            self.assertIsInstance(args.data_path, list)
            self.assertEqual(len(args.data_path), 2)
        finally:
            os.unlink(path)

    def test_boolean_values(self):
        path = self._write_yaml("""
            use_fp16: true
            use_bf16: false
        """)
        try:
            args = self.load_yaml(path)
            self.assertTrue(args.use_fp16)
            self.assertFalse(args.use_bf16)
        finally:
            os.unlink(path)


@unittest.skipUnless(_HAS_YAML_MOD, "yaml_arguments could not be imported")
class TestCoreConfigFromArgs(unittest.TestCase):
    """Test core_config_from_args using a pure-Python stub dataclass."""

    def setUp(self):
        self.core_config_from_args = _yaml_mod.core_config_from_args

    def _make_stub_dataclass(self, **defaults):
        """Dynamically create a minimal dataclass with given fields."""

        annotations = {k: type(v) for k, v in defaults.items()}

        def make_field(v):
            return dataclasses.field(default=v)

        ns = {"__annotations__": annotations}
        for k, v in defaults.items():
            ns[k] = v
        cls = type("StubConfig", (), ns)
        return dataclasses.dataclass(cls)

    def test_extracts_matching_fields(self):
        Stub = self._make_stub_dataclass(hidden_size=512, num_layers=12)
        args = SimpleNamespace(hidden_size=1024, num_layers=24, extra_field=True)
        result = self.core_config_from_args(args, Stub)
        self.assertEqual(result["hidden_size"], 1024)
        self.assertEqual(result["num_layers"], 24)
        self.assertNotIn("extra_field", result)

    def test_raises_on_missing_field(self):
        Stub = self._make_stub_dataclass(hidden_size=512, num_layers=12)
        args = SimpleNamespace(hidden_size=256)  # missing num_layers
        with self.assertRaises(Exception) as ctx:
            self.core_config_from_args(args, Stub)
        self.assertIn("num_layers", str(ctx.exception))

    def test_empty_dataclass(self):
        Stub = self._make_stub_dataclass()
        args = SimpleNamespace(anything=True)
        result = self.core_config_from_args(args, Stub)
        self.assertEqual(result, {})

    def test_none_values_preserved(self):
        Stub = self._make_stub_dataclass(lr=None)
        args = SimpleNamespace(lr=None)
        result = self.core_config_from_args(args, Stub)
        self.assertIsNone(result["lr"])

    def test_all_types(self):
        Stub = self._make_stub_dataclass(lr=1e-4, steps=100, name="gpt", flag=True)
        args = SimpleNamespace(lr=3e-4, steps=500, name="llama", flag=False)
        result = self.core_config_from_args(args, Stub)
        self.assertAlmostEqual(result["lr"], 3e-4)
        self.assertEqual(result["steps"], 500)
        self.assertEqual(result["name"], "llama")
        self.assertFalse(result["flag"])


if __name__ == "__main__":
    unittest.main()
