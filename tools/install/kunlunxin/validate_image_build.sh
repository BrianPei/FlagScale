#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

phase="${IMAGE_BUILD_PHASE:?IMAGE_BUILD_PHASE is required}"
task="${IMAGE_BUILD_TASK:?IMAGE_BUILD_TASK is required}"
base_image="${IMAGE_BUILD_BASE_IMAGE:?IMAGE_BUILD_BASE_IMAGE is required}"
candidate="${IMAGE_BUILD_CANDIDATE_IMAGE:?IMAGE_BUILD_CANDIDATE_IMAGE is required}"
expected_devices="${IMAGE_BUILD_RUNTIME_DEVICE_COUNT:-8}"
smoke_nproc="${IMAGE_BUILD_RUNTIME_SMOKE_NPROC:-2}"

case "$task" in
    train|inference|all) ;;
    *) exit 0 ;;
esac

validate_cuda_runtime() {
    local image="$1"
    local runtime_task="$2"
    local expected_world_size="$3"
    local runtime_phase="$4"

    docker run --rm \
        --privileged \
        --ipc=host \
        --shm-size=64g \
        --env EXPECTED_WORLD_SIZE="$expected_world_size" \
        --env FLAGSCALE_RUNTIME_TASK="$runtime_task" \
        --env FLAGSCALE_RUNTIME_PHASE="$runtime_phase" \
        --entrypoint bash "$image" -lc '
set -euo pipefail
export FLAGSCALE_CONDA="${FLAGSCALE_CONDA:-/root/miniconda}"
export FLAGSCALE_ENV_NAME="${FLAGSCALE_ENV_NAME:-python310_torch29_cuda}"
if [ -f "$FLAGSCALE_CONDA/etc/profile.d/conda.sh" ]; then
    . "$FLAGSCALE_CONDA/etc/profile.d/conda.sh"
    conda activate "$FLAGSCALE_ENV_NAME"
fi
if [ -f /etc/profile.d/flagscale-env.sh ]; then
    . /etc/profile.d/flagscale-env.sh
else
    export PYTHONPATH="/opt/Megatron-LM-FL:${PYTHONPATH:-}"
fi
python - "$FLAGSCALE_RUNTIME_TASK" "$FLAGSCALE_RUNTIME_PHASE" <<"PY"
import os
import sys
from pathlib import Path

import torch

task = sys.argv[1]
phase = sys.argv[2]
expected_world_size = int(os.environ["EXPECTED_WORLD_SIZE"])

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= expected_world_size
assert torch.tensor([1.0], device="cuda").item() == 1.0
if phase == "pre":
    print("Kunlunxin base runtime:", torch.__version__, torch.cuda.device_count())
    raise SystemExit(0)

if task == "train":
    import flagcx
    import megatron.core
    import transformer_engine
    import transformer_engine_torch

    assert flagcx is not None
    assert transformer_engine is not None
    assert transformer_engine_torch is not None
    expected = Path(os.environ.get("FLAGSCALE_MEGATRON_PATH", "/opt/flagscale/deps/Megatron-LM-FL")).resolve()
    actual = Path(megatron.core.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
    print("Kunlunxin train runtime:", torch.__version__, megatron.core.__file__)
elif task == "inference":
    import sentencepiece
    import tiktoken
    import transformers

    assert sentencepiece is not None
    assert tiktoken is not None
    assert transformers is not None
    print("Kunlunxin inference runtime:", torch.__version__, transformers.__version__)

    # Verify the triton.autotune compat shim lets flag_gems import on P800 and
    # that vLLM loads the fl plugin onto the Kunlunxin platform. Without the
    # shim, `import flag_gems` raises TypeError (generate_configs) and vLLM
    # falls back to UnspecifiedPlatform ("Device string must not be empty").
    import flag_gems
    os.environ.setdefault("VLLM_PLUGINS", "fl")
    os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
    os.environ.setdefault("USE_FLAGGEMS", "false")
    # The vllm plugin loader swallows register() failures, leaving
    # current_platform as UnspecifiedPlatform with no traceback. Run
    # register() explicitly so the real exception -- which step of
    # vllm_fl:register fails on the P800 -- surfaces in the CI log.
    import vllm_fl
    try:
        print("vllm_fl.register() ->", vllm_fl.register())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    # Diagnose the Kunlunxin vendor attention deps. The functional test
    # crashes at attention __init__ when torch_xmlir or xtorch_ops are not
    # importable; surface the real state at image-build time so we know
    # whether to install the missing lib or fix env / PYTHONPATH.
    import importlib as _il, subprocess as _sp, sys, glob, site
    _sp_dirs = site.getsitepackages()
    for _mod in ("torch_xmlir", "xtorch_ops"):
        try:
            _m = _il.import_module(_mod)
            print(_mod, "OK:", getattr(_m, "__file__", "built-in"))
        except Exception as _e:
            print(_mod, "IMPORT FAIL:", repr(_e))
            try:
                _out = _sp.check_output(
                    [sys.executable, "-m", "pip", "show", _mod],
                    stderr=_sp.STDOUT, text=True, timeout=30)
                print(_mod, "pip show:", _out.strip()[-500:])
            except Exception as _pe:
                print(_mod, "pip show err:", repr(_pe))
    print("site-packages:", _sp_dirs[0])
    print("glob xmlir/xtorch:",
          glob.glob(_sp_dirs[0] + "/*xmlir*") + glob.glob(_sp_dirs[0] + "/*xtorch*"))
    # Probe whether flag_gems actually DISPATCHES a triton kernel (not just
    # imports). The 4b_tp2_kunlunxin functional test crashed here:
    # flag_gems/ops/zeros.py -> triton load_binary -> CUDA_ERROR_NOT_SUPPORTED,
    # because flag_gems took the generic CUDA ops path instead of the
    # _kunlunxin-adapted path the official 20260812 PDF shows (its server log
    # runs flag_gems._kunlunxin.ops.*). USE_FLAGGEMS was false above (only
    # needed for platform detection); force-enable flag_gems and dispatch
    # torch.zeros here to reproduce (or rule out) the load_binary failure at
    # image-build time, before the functional test wastes a run. Print version
    # + runtime device first so the log shows which backend was selected.
    import importlib.metadata as _fg_meta
    try:
        print("flag_gems version:", _fg_meta.version("flag_gems"))
    except Exception as _e:
        print("flag_gems version err:", repr(_e))
    try:
        import flag_gems.runtime as _fgrt
        print("flag_gems.runtime attrs:",
              [a for a in dir(_fgrt) if not a.startswith("_")][:40])
    except Exception as _e:
        print("flag_gems.runtime probe err:", repr(_e))
    os.environ["USE_FLAGGEMS"] = "true"
    try:
        import flag_gems as _fg
        if hasattr(_fg, "enable"):
            _fg.enable()
        import torch as _torch
        _z = _torch.zeros(4, dtype=_torch.bool, device="cuda")
        print("flag_gems zeros dispatch: OK", _z.device, int(_z.sum().item()))
    except Exception as _e:
        print("flag_gems zeros dispatch: FAIL:", repr(_e))
        import traceback as _tb
        _tb.print_exc()
        print("^^ flag_gems did NOT select the _kunlunxin backend -- the "
              "generic CUDA triton load_binary fails on Kunlunxin "
              "(CUDA_ERROR_NOT_SUPPORTED). The functional test will hit the "
              "same crash. Fix: source-reinstall FlagGems v5.0.0 per the "
              "official PDF, or fix device detection.")
        raise SystemExit(1)
    from vllm.platforms import current_platform
    platform_module = type(current_platform).__module__
    platform_class = type(current_platform).__name__
    print("flag_gems:", getattr(flag_gems, "__file__", "built-in"))
    print("platform_module:", platform_module, "platform_class:", platform_class)
    assert "vllm_fl" in platform_module, (
        f"vLLM did not load fl plugin; platform={platform_module}.{platform_class}"
    )
elif task == "all":
    # Surface the flagcx install root before importing. The runtime image
    # ships flagcx0.13.0 as a pip editable install (egg-link in site-packages
    # whose source dir env.sh adds to PYTHONPATH). If import still fails, the
    # flagcx pkg glob + sys.path below reveal why (wrong layout / not on path).
    import glob as _g, site as _site
    print("FLAGCX_PATH:", os.environ.get("FLAGCX_PATH"))
    _sp = _site.getsitepackages()[0]
    _el = _sp + "/flagcx.egg-link"
    _src = ""
    if os.path.exists(_el):
        _lines = Path(_el).read_text().splitlines()
        _src = _lines[0] if _lines else ""
        print("egg-link source:", _src)
    if _src:
        print("flagcx pkg dirs:",
              _g.glob(_src + "/flagcx") + _g.glob(_src + "/src/flagcx")
              + _g.glob(_src + "/*/flagcx"))
    print("sys.path:", sys.path)
    import flagcx
    import megatron.core
    import sentencepiece
    import tiktoken
    import transformers
    import transformer_engine
    import transformer_engine_torch

    assert flagcx is not None
    assert megatron.core is not None
    assert sentencepiece is not None
    assert tiktoken is not None
    assert transformers is not None
    assert transformer_engine is not None
    assert transformer_engine_torch is not None
    expected = Path(os.environ.get("FLAGSCALE_MEGATRON_PATH", "/opt/flagscale/deps/Megatron-LM-FL")).resolve()
    actual = Path(megatron.core.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)

    # Same flag_gems / vLLM platform probe as the inference task. Catches the
    # triton.autotune(generate_configs) import failure at build time instead of
    # at the inference functional test.
    import flag_gems
    os.environ.setdefault("VLLM_PLUGINS", "fl")
    os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
    os.environ.setdefault("USE_FLAGGEMS", "false")
    # The vllm plugin loader swallows register() failures, leaving
    # current_platform as UnspecifiedPlatform with no traceback. Run
    # register() explicitly so the real exception -- which step of
    # vllm_fl:register fails on the P800 -- surfaces in the CI log.
    import vllm_fl
    try:
        print("vllm_fl.register() ->", vllm_fl.register())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    # Diagnose the Kunlunxin vendor attention deps. The functional test
    # crashes at attention __init__ when torch_xmlir or xtorch_ops are not
    # importable; surface the real state at image-build time so we know
    # whether to install the missing lib or fix env / PYTHONPATH.
    import importlib as _il, subprocess as _sp, sys, glob, site
    _sp_dirs = site.getsitepackages()
    for _mod in ("torch_xmlir", "xtorch_ops"):
        try:
            _m = _il.import_module(_mod)
            print(_mod, "OK:", getattr(_m, "__file__", "built-in"))
        except Exception as _e:
            print(_mod, "IMPORT FAIL:", repr(_e))
            try:
                _out = _sp.check_output(
                    [sys.executable, "-m", "pip", "show", _mod],
                    stderr=_sp.STDOUT, text=True, timeout=30)
                print(_mod, "pip show:", _out.strip()[-500:])
            except Exception as _pe:
                print(_mod, "pip show err:", repr(_pe))
    print("site-packages:", _sp_dirs[0])
    print("glob xmlir/xtorch:",
          glob.glob(_sp_dirs[0] + "/*xmlir*") + glob.glob(_sp_dirs[0] + "/*xtorch*"))
    # Probe whether flag_gems actually DISPATCHES a triton kernel (not just
    # imports). The 4b_tp2_kunlunxin functional test crashed here:
    # flag_gems/ops/zeros.py -> triton load_binary -> CUDA_ERROR_NOT_SUPPORTED,
    # because flag_gems took the generic CUDA ops path instead of the
    # _kunlunxin-adapted path the official 20260812 PDF shows (its server log
    # runs flag_gems._kunlunxin.ops.*). USE_FLAGGEMS was false above (only
    # needed for platform detection); force-enable flag_gems and dispatch
    # torch.zeros here to reproduce (or rule out) the load_binary failure at
    # image-build time, before the functional test wastes a run. Print version
    # + runtime device first so the log shows which backend was selected.
    import importlib.metadata as _fg_meta
    try:
        print("flag_gems version:", _fg_meta.version("flag_gems"))
    except Exception as _e:
        print("flag_gems version err:", repr(_e))
    try:
        import flag_gems.runtime as _fgrt
        print("flag_gems.runtime attrs:",
              [a for a in dir(_fgrt) if not a.startswith("_")][:40])
    except Exception as _e:
        print("flag_gems.runtime probe err:", repr(_e))
    os.environ["USE_FLAGGEMS"] = "true"
    try:
        import flag_gems as _fg
        if hasattr(_fg, "enable"):
            _fg.enable()
        import torch as _torch
        _z = _torch.zeros(4, dtype=_torch.bool, device="cuda")
        print("flag_gems zeros dispatch: OK", _z.device, int(_z.sum().item()))
    except Exception as _e:
        print("flag_gems zeros dispatch: FAIL:", repr(_e))
        import traceback as _tb
        _tb.print_exc()
        print("^^ flag_gems did NOT select the _kunlunxin backend -- the "
              "generic CUDA triton load_binary fails on Kunlunxin "
              "(CUDA_ERROR_NOT_SUPPORTED). The functional test will hit the "
              "same crash. Fix: source-reinstall FlagGems v5.0.0 per the "
              "official PDF, or fix device detection.")
        raise SystemExit(1)
    from vllm.platforms import current_platform
    platform_module = type(current_platform).__module__
    platform_class = type(current_platform).__name__
    print("flag_gems:", getattr(flag_gems, "__file__", "built-in"))
    print("platform_module:", platform_module, "platform_class:", platform_class)
    assert "vllm_fl" in platform_module, (
        f"vLLM did not load fl plugin; platform={platform_module}.{platform_class}"
    )
    print("Kunlunxin all runtime:", torch.__version__, megatron.core.__file__)
else:
    raise SystemExit(f"Unsupported Kunlunxin runtime task: {task}")
PY
'
}

case "$phase" in
    pre)
        if [[ "$base_image" == */* ]]; then
            docker pull "$base_image"
        elif ! docker image inspect "$base_image" >/dev/null 2>&1; then
            echo "Kunlunxin base image is not available locally: $base_image" >&2
            echo "Use a registry-qualified base image or load the vendor image on the P800 runner." >&2
            exit 1
        fi
        validate_cuda_runtime "$base_image" "$task" "$expected_devices" pre
        ;;
    post)
        validate_cuda_runtime "$candidate" "$task" "$smoke_nproc" post
        ;;
    *)
        exit 0
        ;;
esac
