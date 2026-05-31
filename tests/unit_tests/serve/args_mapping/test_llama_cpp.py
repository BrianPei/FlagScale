import json

import pytest

from flagscale.serve.args_mapping.mapping_funcs import llama_cpp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"rope_type": "none", "factor": 1.0},
            {"rope_scaling": "none", "rope_scale": 1.0},
        ),
        (
            json.dumps({"rope_type": "linear", "factor": 2.5}),
            {"rope_scaling": "linear", "rope_scale": 2.5},
        ),
    ],
)
def test_rope_scaling_converter_accepts_dict_and_json(value, expected):
    assert llama_cpp.llama_cpp_rope_scaling_converter(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        ["linear"],
        {"rope_type": "dynamic", "factor": 2},
        {"rope_type": "yarn"},
    ],
)
def test_rope_scaling_converter_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        llama_cpp.llama_cpp_rope_scaling_converter(value)


@pytest.mark.parametrize(
    "dtype", ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
)
def test_kv_cache_dtype_converter_accepts_llama_cpp_native_dtypes(dtype):
    assert llama_cpp.llama_cpp_kv_cache_dtype_converter(dtype) == {
        "cache_type_k": dtype,
        "cache_type_v": dtype,
    }


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_kv_cache_dtype_converter_maps_vllm_fp8_aliases(dtype):
    assert llama_cpp.llama_cpp_kv_cache_dtype_converter(dtype) == {
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
    }


def test_kv_cache_dtype_converter_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Invalid kv_cache_dtype"):
        llama_cpp.llama_cpp_kv_cache_dtype_converter("int8")


@pytest.mark.parametrize(
    "parser", ["deepseek", "deepseek_r1", "deep-seek", "DeepSeek", "Deep-Seek"]
)
def test_reasoning_parser_converter_normalizes_deepseek_aliases(parser):
    assert llama_cpp.llama_cpp_reasoning_parser_converter(parser) == {
        "reasoning_format": "deepseek"
    }


def test_reasoning_parser_converter_rejects_unknown_parser():
    with pytest.raises(ValueError, match="Invalid reasoning_parser"):
        llama_cpp.llama_cpp_reasoning_parser_converter("qwen")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(4096, 4096), ("2k", 2000), ("2K", 2048)],
)
def test_max_model_len_converter_accepts_int_and_k_suffixes(value, expected):
    assert llama_cpp.llama_cpp_max_model_len_converter(value) == {"ctx_size": expected}


@pytest.mark.parametrize("value", [None, "bad", "2m"])
def test_max_model_len_converter_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        llama_cpp.llama_cpp_max_model_len_converter(value)


@pytest.mark.parametrize(
    "level", ["warning", "error", "critical", "trace", "WARNING", "ERROR"]
)
def test_uvicorn_log_level_converter_maps_quiet_levels(level):
    assert llama_cpp.llama_cpp_uvicorn_log_level_converter(level) == {
        "log_verbosity": 0
    }


@pytest.mark.parametrize("level", ["debug", "info", "DEBUG", "INFO"])
def test_uvicorn_log_level_converter_maps_verbose_levels(level):
    assert llama_cpp.llama_cpp_uvicorn_log_level_converter(level) == {
        "log_verbosity": 1
    }


def test_uvicorn_log_level_converter_rejects_invalid_level():
    with pytest.raises(ValueError, match="Invalid uvicorn_log_level"):
        llama_cpp.llama_cpp_uvicorn_log_level_converter("notice")


def test_model_converter_returns_first_nested_gguf(tmp_path):
    model_dir = tmp_path / "model"
    nested_dir = model_dir / "nested"
    nested_dir.mkdir(parents=True)
    gguf = nested_dir / "weights.GGUF"
    gguf.write_text("stub")

    assert llama_cpp.llama_cpp_model_converter(str(model_dir)) == {"model": str(gguf)}


def test_model_converter_returns_directory_when_no_gguf_exists(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert llama_cpp.llama_cpp_model_converter(str(model_dir)) == {
        "model": str(model_dir)
    }


def test_model_converter_rejects_missing_or_file_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        llama_cpp.llama_cpp_model_converter(str(tmp_path / "missing"))

    file_path = tmp_path / "model.gguf"
    file_path.write_text("stub")
    with pytest.raises(NotADirectoryError):
        llama_cpp.llama_cpp_model_converter(str(file_path))
