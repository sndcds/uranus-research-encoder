"""Manifest checks must reject corruption before ORT sees any graph."""

import json
from pathlib import Path

import pytest

from uranus_research_encoder import v5_onnx_artifacts as artifacts


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    frozen = Path(__file__).resolve().parents[1] / "validation/jina-v5-merged-v1"
    manifest = json.loads((frozen / "export-manifest.json").read_text())
    for name in artifacts.ARTIFACT_FILES:
        (tmp_path / name).write_text("synthetic " + name)
    provenance = json.loads((frozen / "native-provenance.json").read_text())
    # Synthetic tokenizer files, real immutable source-provenance identities.
    synthetic_hashes = dict(artifacts.SOURCE_HASHES)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        synthetic_hashes[name] = artifacts.sha256(tmp_path / name)
    monkeypatch.setattr(artifacts, "SOURCE_HASHES", synthetic_hashes)
    for row in provenance["artifacts"]:
        row["sha256"] = synthetic_hashes[row["path"]]
    (tmp_path / "native-provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "merge-audit.json").write_bytes((frozen / "merge-audit.json").read_bytes())
    manifest["artifacts"] = {n: artifacts.sha256(tmp_path / n) for n in artifacts.ARTIFACT_FILES}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_verified_manifest(artifact_dir):
    assert (
        artifacts.verify_manifest(artifact_dir, artifacts.sha256(artifact_dir / "manifest.json"))[
            "backend"
        ]
        == "v5-onnx-merged"
    )


@pytest.mark.parametrize("name", sorted(artifacts.ARTIFACT_FILES))
def test_corrupt_artifact_rejected(artifact_dir, name):
    digest = artifacts.sha256(artifact_dir / "manifest.json")
    with (artifact_dir / name).open("a") as f:
        f.write("corruption")
    with pytest.raises(ValueError, match="hash_mismatch"):
        artifacts.verify_manifest(artifact_dir, digest)


def test_rewritten_manifest_does_not_authenticate_itself(artifact_dir):
    digest = artifacts.sha256(artifact_dir / "manifest.json")
    m = json.loads((artifact_dir / "manifest.json").read_text())
    (artifact_dir / "model.onnx").write_text("different graph")
    m["artifacts"]["model.onnx"] = artifacts.sha256(artifact_dir / "model.onnx")
    (artifact_dir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="manifest_hash_mismatch"):
        artifacts.verify_manifest(artifact_dir, digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_revision", "main"),
        ("schema", "unknown"),
        ("backend", "torch"),
        ("design", "A"),
        ("contract", {}),
        ("optimization", "basic"),
        ("artifacts", {"../model.onnx": "0" * 64}),
    ],
)
def test_wrong_contract_even_with_trusted_digest(artifact_dir, field, value):
    m = json.loads((artifact_dir / "manifest.json").read_text())
    m[field] = value
    (artifact_dir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        artifacts.verify_manifest(artifact_dir, artifacts.sha256(artifact_dir / "manifest.json"))


def test_symlink_rejected(artifact_dir):
    p = artifact_dir / "model.onnx"
    target = artifact_dir / "other"
    p.rename(target)
    p.symlink_to(target)
    with pytest.raises(ValueError, match="hash_mismatch"):
        artifacts.verify_manifest(artifact_dir, artifacts.sha256(artifact_dir / "manifest.json"))


def test_only_runtime_dependencies_are_imported():
    import subprocess
    import sys

    script = """
import sys
from uranus_research_encoder.v5_onnx_artifacts import verify_manifest
from uranus_research_encoder.v5_onnx_backend import V5MergedOnnxBackend
from uranus_research_encoder.app import create_app
assert 'torch' not in sys.modules
assert 'peft' not in sys.modules
assert 'onnx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
