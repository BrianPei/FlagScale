#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Validate source-resolution labels on a built container image."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from typing import Any

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(RuntimeError):
    """Raised when image labels do not match resolved source metadata."""


def validate_provenance(labels: Any, sources: Any) -> None:
    if not isinstance(labels, Mapping):
        raise ProvenanceError("image labels must be a mapping")
    if not isinstance(sources, Mapping):
        raise ProvenanceError("resolved sources must be a mapping")

    for required in (
        "org.opencontainers.image.source",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.base.name",
    ):
        if not labels.get(required):
            raise ProvenanceError(f"missing required image label: {required}")

    for source_name, source in sources.items():
        if not isinstance(source, Mapping):
            raise ProvenanceError(f"invalid source metadata: {source_name}")
        revision = source.get("revision")
        selector = source.get("selector")
        if not isinstance(revision, str) or not SHA_RE.fullmatch(revision):
            raise ProvenanceError(f"invalid resolved revision: {source_name}")
        if not isinstance(selector, str) or not selector:
            raise ProvenanceError(f"invalid resolved selector: {source_name}")

        label_name = source_name.replace("_", "-")
        expected = {
            f"io.flagscale.source.{label_name}.revision": revision,
            f"io.flagscale.source.{label_name}.selector": selector,
        }
        for key, value in expected.items():
            if labels.get(key) != value:
                raise ProvenanceError(
                    f"image label mismatch: {key}: expected {value!r}, found {labels.get(key)!r}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--sources-json", required=True)
    args = parser.parse_args()

    try:
        labels = json.loads(args.labels_json)
        sources = json.loads(args.sources_json)
        validate_provenance(labels, sources)
    except (json.JSONDecodeError, ProvenanceError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
