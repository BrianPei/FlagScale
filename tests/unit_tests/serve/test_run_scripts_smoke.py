import importlib
import runpy
import sys
import types
from unittest.mock import MagicMock

from omegaconf import OmegaConf


def _reload_module(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_run_inference_engine_dispatches_to_configured_backend(monkeypatch):
    module = _reload_module("flagscale.serve.run_inference_engine")
    config = OmegaConf.create(
        {
            "serve": [
                {
                    "serve_id": "serve_model",
                    "engine": "vllm",
                    "engine_args": {"model": "m"},
                }
            ],
            "experiment": {"task": {}},
        }
    )
    monkeypatch.setattr(module.serve, "load_args", MagicMock())
    monkeypatch.setattr(module.serve, "task_config", config)
    monkeypatch.setattr(module, "vllm_serve", MagicMock(return_value=0))

    module.main()

    module.serve.load_args.assert_called_once()
    module.vllm_serve.assert_called_once_with(config.serve[0])


def test_run_inference_engine_builds_vllm_command_without_starting_service(monkeypatch):
    module = _reload_module("flagscale.serve.run_inference_engine")
    process = MagicMock()
    process.communicate.return_value = ("out", "err")
    process.returncode = 3
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    result = module.vllm_serve(
        {
            "engine_args": {"model": "base", "tensor_parallel_size": 2},
            "engine_args_specific": {"vllm": {"dtype": "float16"}},
        }
    )

    assert result == 3
    command = popen.call_args.args[0]
    assert command[:3] == ["vllm", "serve", "base"]
    assert "--tensor-parallel-size" in command
    assert "--dtype" in command


def test_run_inference_engine_builds_llama_cpp_command_without_starting_service(
    monkeypatch,
):
    module = _reload_module("flagscale.serve.run_inference_engine")
    process = MagicMock()
    process.communicate.return_value = ("out", "err")
    process.returncode = 5
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(
        module.ARGS_CONVERTER,
        "convert",
        MagicMock(return_value={"model": "/tmp/model.gguf", "ctx_size": 1024}),
    )

    result = module.llama_cpp_serve(
        {
            "engine_args": {"model": "/tmp/model"},
            "engine_args_specific": {"llama_cpp": {"threads": 4}},
        }
    )

    assert result == 5
    command = module.subprocess.Popen.call_args.args[0]
    assert command[:3] == ["llama-server", "--model", "/tmp/model.gguf"]
    assert "--ctx-size" in command
    assert "--threads" in command


def test_run_fs_serve_vllm_import_builds_llm_config_with_stubs(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_serve = types.ModuleType("ray.serve")
    fake_llm = types.ModuleType("ray.serve.llm")

    class FakeLLMConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_llm.LLMConfig = FakeLLMConfig
    fake_llm.build_openai_app = MagicMock(return_value="app")
    fake_serve.start = MagicMock()
    fake_serve.run = MagicMock()
    fake_ray.serve = fake_serve
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.serve", fake_serve)
    monkeypatch.setitem(sys.modules, "ray.serve.llm", fake_llm)

    import flagscale.serve as serve_pkg

    config = OmegaConf.create(
        {
            "serve": [
                {
                    "serve_id": "serve_model",
                    "engine_args": {
                        "model": "m",
                        "served_model_name": "served",
                        "tensor_parallel_size": 2,
                        "custom_arg": "kept-for-main-block",
                    },
                    "resources": {"num_replicas": 2},
                }
            ],
            "experiment": {"runner": {"deploy": {"port": 9000}}},
        }
    )
    monkeypatch.setattr(serve_pkg, "load_args", MagicMock())
    monkeypatch.setattr(serve_pkg, "task_config", config)

    module = _reload_module("flagscale.serve.run_fs_serve_vllm")

    assert module.model_config == config.serve[0]
    assert module.llm_config.kwargs["model_loading_config"] == {
        "model_id": "served",
        "model_source": "m",
    }
    assert (
        module.llm_config.kwargs["deployment_config"]["autoscaling_config"][
            "min_replicas"
        ]
        == 2
    )
    assert module.llm_config.kwargs["engine_kwargs"] == {"tensor_parallel_size": 2}


def test_run_serve_pi_parse_config_and_main_dispatch_with_stubs(monkeypatch, tmp_path):
    module = _import_pi_module_with_stubs(monkeypatch)
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "serve:\n  - engine_args:\n      model: pi0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_serve_pi.py", "--config-path", str(config_path), "--log-dir", "logs"],
    )

    parsed = module.parse_config()
    assert parsed.serve[0].engine_args.model == "pi0"

    server = MagicMock()
    fake_server_cls = MagicMock(return_value=server)
    monkeypatch.setattr(module, "PI0Server", fake_server_cls)
    module.main(parsed.serve[0])

    fake_server_cls.assert_called_once_with(parsed.serve[0])
    server.serve.assert_called_once()


def test_run_serve_qwen_gr00t_validate_batch_and_main_dispatch_with_stubs(
    monkeypatch, tmp_path
):
    module = _import_qwen_module_with_stubs(monkeypatch)

    valid = {
        "task": "pick",
        "observation.state": [1.0, 2.0],
        "observation.images.cam": __import__("numpy").zeros(
            (2, 2, 3), dtype=__import__("numpy").uint8
        ),
    }
    assert module.validate_batch(valid) == []
    assert "Missing required key 'task'" in module.validate_batch({})

    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "serve:\n  - engine_args:\n      model: qwen\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_serve_qwen_gr00t.py",
            "--config-path",
            str(config_path),
            "--log-dir",
            "logs",
        ],
    )
    parsed = module.parse_config()

    fake_policy = MagicMock(host="127.0.0.1", port=6000)
    monkeypatch.setattr(module, "Policy", MagicMock(return_value=fake_policy))
    fake_server = MagicMock()
    monkeypatch.setattr(
        module, "WebsocketPolicyServer", MagicMock(return_value=fake_server)
    )
    module.main(parsed.serve[0])

    module.Policy.assert_called_once_with(parsed.serve[0])
    module.WebsocketPolicyServer.assert_called_once()
    fake_server.serve_forever.assert_called_once()


def test_run_fs_serve_vllm_main_block_uses_ray_serve_stubs(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_serve = types.ModuleType("ray.serve")
    fake_llm = types.ModuleType("ray.serve.llm")
    fake_llm.LLMConfig = lambda **kwargs: kwargs
    fake_llm.build_openai_app = MagicMock(return_value="app")
    fake_serve.start = MagicMock()
    fake_serve.run = MagicMock()
    fake_ray.serve = fake_serve
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.serve", fake_serve)
    monkeypatch.setitem(sys.modules, "ray.serve.llm", fake_llm)

    import flagscale.serve as serve_pkg

    config = OmegaConf.create(
        {
            "serve": [
                {
                    "serve_id": "serve_model",
                    "engine_args": {"model": "m"},
                    "resources": {},
                }
            ],
            "experiment": {"runner": {"deploy": {"port": 8001}}},
        }
    )
    monkeypatch.setattr(serve_pkg, "load_args", MagicMock())
    monkeypatch.setattr(serve_pkg, "task_config", config)

    sys.modules.pop("flagscale.serve.run_fs_serve_vllm", None)
    runpy.run_module("flagscale.serve.run_fs_serve_vllm", run_name="__main__")

    fake_serve.start.assert_called_once_with(
        http_options={"host": "0.0.0.0", "port": 8001}
    )
    fake_llm.build_openai_app.assert_called_once()
    fake_serve.run.assert_called_once_with("app", name="vllm_service", blocking=True)


def _install_common_vla_stubs(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    constants = types.ModuleType("flagscale.models.utils.constants")
    constants.ACTION = "action"
    constants.OBS_IMAGES = "observation.images"
    constants.OBS_STATE = "observation.state"
    monkeypatch.setitem(sys.modules, "flagscale.models.utils.constants", constants)

    logger_module = types.ModuleType("flagscale.logger")
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.logger", logger_module)

    runner_utils = types.ModuleType("flagscale.runner.utils")
    runner_utils.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.runner.utils", runner_utils)


def _import_pi_module_with_stubs(monkeypatch):
    _install_common_vla_stubs(monkeypatch)

    flask = types.ModuleType("flask")

    class FakeFlask:
        def __init__(self, name):
            self.name = name

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def run(self, *args, **kwargs):
            return None

    flask.Flask = FakeFlask
    flask.jsonify = lambda value: value
    flask.request = MagicMock()
    monkeypatch.setitem(sys.modules, "flask", flask)

    flask_cors = types.ModuleType("flask_cors")
    flask_cors.CORS = MagicMock()
    monkeypatch.setitem(sys.modules, "flask_cors", flask_cors)

    pil = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.open = MagicMock()
    pil.Image = image_module
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)

    types_module = types.ModuleType("flagscale.models.configs.types")
    types_module.FeatureType = types.SimpleNamespace(ACTION="action")
    types_module.NormalizationMode = types.SimpleNamespace(
        IDENTITY="identity", MEAN_STD="mean_std"
    )
    types_module.PolicyFeature = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.models.configs.types", types_module)

    for module_name, class_name in [
        ("flagscale.models.pi0.configuration_pi0", "PI0Config"),
        ("flagscale.models.pi0.modeling_pi0", "PI0Policy"),
        ("flagscale.models.pi05.configuration_pi05", "PI05Config"),
        ("flagscale.models.pi05.modeling_pi05", "PI05Policy"),
    ]:
        module = types.ModuleType(module_name)
        setattr(module, class_name, MagicMock())
        monkeypatch.setitem(sys.modules, module_name, module)

    train_pi = types.ModuleType("flagscale.train.train_pi")
    train_pi.make_pre_post_processors = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.train.train_pi", train_pi)

    return _reload_module("flagscale.serve.run_serve_pi")


def _import_qwen_module_with_stubs(monkeypatch):
    _install_common_vla_stubs(monkeypatch)

    monkeypatch.setitem(
        sys.modules,
        "flagscale.serve.processor",
        types.ModuleType("flagscale.serve.processor"),
    )

    image_layout = types.ModuleType("flagscale.serve.processor.image_layout_processor")
    image_layout.ImageLayoutProcessorStep = MagicMock()
    monkeypatch.setitem(
        sys.modules, "flagscale.serve.processor.image_layout_processor", image_layout
    )

    image_resize = types.ModuleType("flagscale.serve.processor.image_resize_processor")
    image_resize.ImageResizeProcessorStep = MagicMock()
    monkeypatch.setitem(
        sys.modules, "flagscale.serve.processor.image_resize_processor", image_resize
    )

    vla = types.ModuleType("flagscale.models.vla")
    vla.TrainablePolicy = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.models.vla", vla)

    websocket = types.ModuleType("flagscale.serve.websocket_policy_server")
    websocket.WebsocketPolicyServer = MagicMock()
    monkeypatch.setitem(
        sys.modules, "flagscale.serve.websocket_policy_server", websocket
    )

    train_processor = types.ModuleType("flagscale.train.processor")
    train_processor.PolicyProcessorPipeline = MagicMock()
    train_processor.ProcessorStepRegistry = MagicMock()
    monkeypatch.setitem(sys.modules, "flagscale.train.processor", train_processor)

    return _reload_module("flagscale.serve.run_serve_qwen_gr00t")
