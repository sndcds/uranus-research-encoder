"""Fail-closed verification for the explicitly selected FP32 v5 ONNX artifact."""

import hashlib
import json
from pathlib import Path

from .version import MODEL_REPOSITORY, MODEL_REVISION

SCHEMA = "jina-v5-merged-onnx-v1"
BACKEND = "v5-onnx-merged"
CONTRACT = {
    "query_prefix": "Query: ",
    "passage_prefix": "Document: ",
    "pooling": "last_non_padding_token",
    "normalization": "l2_once_dim1_eps1e-12",
    "dimensions": 1024,
    "dtype": "float32",
    "padding_side": "right",
    "max_tokens": 32768,
    "attention": "native Qwen3 causal eager",
    "native_inference_batching": "one text per forward",
    "native_loader_threads": 1,
}
SOURCE_HASHES = {
    "config.json": "1af1e1269488c83d8b2332e42099f0d2201d687fbe074d1ed096c6201f283546",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model.safetensors": "045fa75ff963a528cda2589fb1ca0a9ad848b53511780ed4f08f6fe10f6167c3",
    "adapters/retrieval/adapter_config.json": (
        "f37c5d6dd368e2675e54e01b685252d4d44eed042c773a48f56ebe1565cd0320"
    ),
    "adapters/retrieval/adapter_model.safetensors": (
        "2bc6ab71895eb04664e4d995ee29e1620603f3a3fc4dfc573bc2383dfc85bb94"
    ),
}
ARTIFACT_FILES = {
    "model.onnx",
    "model.onnx.data",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merge-audit.json",
    "native-provenance.json",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_native_snapshot(snapshot: Path) -> None:
    if snapshot.name != MODEL_REVISION:
        raise ValueError("revision_mismatch")
    for name, digest in SOURCE_HASHES.items():
        if sha256(snapshot / name) != digest:
            raise ValueError("native_artifact_hash_mismatch")


def verify_manifest(root: Path, trusted_manifest_sha256: str) -> dict:
    # The trusted digest is supplied separately by configuration. A changed graph
    # plus a rewritten colocated manifest must not authenticate itself.
    if not root.is_absolute() or sha256(root / "manifest.json") != trusted_manifest_sha256:
        raise ValueError("manifest_hash_mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("backend") != BACKEND
        or manifest.get("model_repository") != MODEL_REPOSITORY
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("contract") != CONTRACT
        or manifest.get("design") != "B"
        or manifest.get("opset") != 18
        or manifest.get("optimization") != "disabled"
    ):
        raise ValueError("onnx_contract_mismatch")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != ARTIFACT_FILES:
        raise ValueError("onnx_artifact_set_mismatch")
    for name, digest in artifacts.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("onnx_artifact_hash_mismatch")
        if name in SOURCE_HASHES and digest != SOURCE_HASHES[name]:
            raise ValueError("tokenizer_hash_mismatch")
    provenance = json.loads((root / "native-provenance.json").read_text())
    if (
        {r["path"]: r["sha256"] for r in provenance["artifacts"]} != SOURCE_HASHES
        or provenance["model_revision"] != MODEL_REVISION
        or provenance["model_repository"] != MODEL_REPOSITORY
        or provenance["adapter"] != "adapters/retrieval"
        or provenance["contract"] != CONTRACT
    ):
        raise ValueError("provenance_mismatch")
    audit = json.loads((root / "merge-audit.json").read_text())
    if (
        audit.get("pass") is not True
        or audit.get("exact_matches") != 196
        or audit.get("observed_target_count") != 196
        or audit.get("expected_target_count") != 196
        or audit.get("adapter_tensor_count") != 392
        or audit.get("max_difference") != 0
        or audit.get("active_lora_layers") != 0
        or audit.get("missing_targets") != []
        or audit.get("unexpected_targets") != []
    ):
        raise ValueError("merge_audit_mismatch")
    return manifest
