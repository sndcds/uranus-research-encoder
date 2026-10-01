"""Versioned, operator-owned provenance for locally derived ONNX artifacts."""

import hashlib
import json
from pathlib import Path

from .version import MODEL_REPOSITORY, MODEL_REVISION

EXPORT_CONTRACT = "jina-v3-merged-f32-v1"
EXPORTER_VERSION = "1"
OPSET = 18
TASKS = {"query": "retrieval_query", "passage": "retrieval_passage"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("contract") != EXPORT_CONTRACT
        or manifest.get("exporter_version") != EXPORTER_VERSION
        or manifest.get("source") != {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}
        or manifest.get("opset") != OPSET
        or manifest.get("dtype") != "float32"
        or set(manifest.get("tasks", {})) != set(TASKS)
    ):
        raise ValueError("merged_manifest_mismatch")
    files = manifest.get("files", {})
    required = {"tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    for kind, adapter in TASKS.items():
        graph = f"retrieval-{kind}/model.onnx"
        if manifest["tasks"][kind] != {"task": kind, "source_adapter": adapter, "graph": graph}:
            raise ValueError("merged_task_mismatch")
        required.add(graph)
        required.add(f"retrieval-{kind}/model.onnx.data")
    if not required <= files.keys():
        raise ValueError("incomplete_merged_manifest")
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe_artifact_path")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("unsafe_artifact_path")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("merged_artifact_mismatch")
    return manifest
