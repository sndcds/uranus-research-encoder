"""Artifact digest helper; no ONNX manifest is accepted by v5."""

import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(root: Path) -> dict:
    raise ValueError("onnx_not_supported_for_jina_v5")
