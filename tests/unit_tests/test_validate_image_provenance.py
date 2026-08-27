# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "tools" / "install" / "utils" / "validate_image_provenance.py"
SPEC = importlib.util.spec_from_file_location("validate_image_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _labels(revision: str) -> dict[str, str]:
    return {
        "org.opencontainers.image.source": "https://github.com/flagos-ai/FlagScale",
        "org.opencontainers.image.revision": "f" * 40,
        "org.opencontainers.image.base.name": "example/base@sha256:abc",
        "io.flagscale.source.megatron-lm-fl.selector": "main",
        "io.flagscale.source.megatron-lm-fl.revision": revision,
    }


def test_validate_provenance_accepts_matching_source_labels():
    revision = "a" * 40
    MODULE.validate_provenance(
        _labels(revision),
        {
            "megatron_lm_fl": {
                "selector": "main",
                "revision": revision,
            }
        },
    )


def test_validate_provenance_rejects_mismatched_revision():
    with pytest.raises(MODULE.ProvenanceError, match="image label mismatch"):
        MODULE.validate_provenance(
            _labels("a" * 40),
            {
                "megatron_lm_fl": {
                    "selector": "main",
                    "revision": "b" * 40,
                }
            },
        )
