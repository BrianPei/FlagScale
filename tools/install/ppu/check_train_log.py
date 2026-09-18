#!/usr/bin/env python3
"""Check all ten PPU iterations; write an observed baseline for human review."""

import json
import math
import re
from pathlib import Path

import yaml


def extract_losses(text, expected):
    entries = re.findall(
        r"iteration\s+(\d+)\s*/\s*\d+[^\n]*?lm loss:\s*([^\s|]+)", text
    )
    iterations = [int(step) for step, _ in entries]
    values = [float(value) for _, value in entries]
    if iterations != list(range(1, expected + 1)):
        raise ValueError(f"Expected iterations 1..{expected}, got {iterations}")
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"Non-finite or invalid training loss: {values}")
    return values


def main():
    root = Path("tests/functional_tests/train/qwen3")
    config = yaml.safe_load((root / "conf/train/0_6b_ppu.yaml").read_text())
    logs = [
        path for path in (root / "test_results/0_6b_ppu/logs/details").rglob("stdout.log")
        if "lm loss:" in path.read_text(errors="replace")
    ]
    if len(logs) != 1:
        raise RuntimeError(f"Expected one current training output, found {logs}")
    values = extract_losses(logs[0].read_text(), config["model"]["train_iters"])
    output = Path("ppu-artifacts/observed-loss.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"lm loss:": {"values": values}}, indent=2) + "\n")
    print(f"All iterations have finite losses. Review {output}; it is not an approved baseline.")


if __name__ == "__main__":
    main()
