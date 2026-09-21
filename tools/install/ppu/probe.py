#!/usr/bin/env python3
"""Device, collective and TE checks for the shared image-build contract."""

import argparse
import os
from datetime import timedelta
from pathlib import Path


def smoke(training):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world >= 2, "PPU collective acceptance requires at least two ranks"
    expected_devices = int(os.environ["EXPECTED_DEVICE_COUNT"])
    assert expected_devices >= world
    assert torch.cuda.is_available() and torch.cuda.device_count() >= expected_devices
    print("torch:", torch.__version__, "devices:", torch.cuda.device_count(), flush=True)
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
            from transformer_engine.plugin.core import get_manager
            from transformer_engine.pytorch import Linear

            from megatron.core.extensions.transformer_engine import HAVE_TE
            from megatron.core.models.gpt import GPTModel

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
            for package in ("megatron-lm-fl", "transformer-engine-fl"):
                revision_file = Path("/opt/flagscale/deps") / f".{package}.ref"
                print(package, revision_file.read_text().strip())
        torch.cuda.synchronize()
        print(
            f"rank={rank}: collective and {'TE backward' if training else 'device backward'} PASS",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("device", "train"))
    args = parser.parse_args()
    smoke(args.mode == "train")
