#!/usr/bin/env python3
"""PPU inventory (no required imports), or fail-closed device/TE smoke tests."""

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import json
import io
import os
import platform
import subprocess
import sys
from datetime import timedelta
from pathlib import Path


def inventory():
    report = {"python": sys.version, "executable": sys.executable, "arch": platform.machine()}
    for name, command in {
        "os": ["cat", "/etc/os-release"],
        "cpu": ["lscpu"],
        "memory": ["free", "-h"],
        "driver_firmware_devices": ["ppu-smi"],
        "runtime_libraries": ["ldconfig", "-p"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            report[name] = {"returncode": result.returncode, "output": result.stdout + result.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report[name] = {"unavailable": str(exc)}
    report["packages"] = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    report["modules"] = {}
    for name in ("torch", "torch_cpu", "torch_ppu", "flagcx", "megatron.core", "transformer_engine", "transformer_engine_torch"):
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                module = importlib.import_module(name)
            report["modules"][name] = {"path": getattr(module, "__file__", None), "diagnostics": captured.getvalue()}
        except Exception as exc:
            report["modules"][name] = {"unavailable": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=False))


def smoke(training):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world >= 2, "PPU collective acceptance requires at least two ranks"
    assert torch.cuda.is_available() and torch.cuda.device_count() >= world
    torch.cuda.set_device(rank)
    print("PPU device:", rank, torch.cuda.get_device_name(rank), flush=True)
    device = torch.device("cuda", rank)
    x = torch.ones((16, 16), device=device, requires_grad=True)
    (x @ x).sum().backward()
    torch.testing.assert_close(x.grad.cpu(), torch.full((16, 16), 32.0))
    assert dist.is_nccl_available(), "Vendor NCCL/PCCL backend is unavailable"
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        if training:
            from megatron.core.extensions.transformer_engine import HAVE_TE
            from megatron.core.models.gpt import GPTModel
            from transformer_engine.plugin.core import get_manager
            from transformer_engine.pytorch import Linear

            assert HAVE_TE and GPTModel is not None
            selected = get_manager().get_selected_impl_id("generic_gemm")
            assert selected == "reference.torch", selected
            layer = Linear(32, 32, params_dtype=torch.bfloat16).to(device)
            inputs = torch.randn(8, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
            output = layer(inputs)
            output.float().square().mean().backward()
            assert torch.isfinite(output).all().item()
            assert inputs.grad is not None and torch.isfinite(inputs.grad).all().item()
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all().item()
            for package in ("Megatron-LM-FL", "TransformerEngine-FL"):
                path = Path("/opt/flagscale/deps") / package
                print(package, subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip())
        torch.cuda.synchronize()
        print(f"rank={rank}: collective and {'TE backward' if training else 'device backward'} PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inventory", "device", "train"))
    args = parser.parse_args()
    if args.mode == "inventory":
        inventory()
    else:
        smoke(args.mode == "train")
