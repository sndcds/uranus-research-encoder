"""Unsupported ONNX paths must fail before touching any cached graph or output."""

import runpy
import socket
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app
from uranus_research_encoder.merged_artifacts import verify_manifest
from uranus_research_encoder.merged_onnx_backend import MergedOnnxBackend
from uranus_research_encoder.model_artifacts import artifact_patterns
from uranus_research_encoder.onnx_backend import OnnxBackend


@pytest.mark.parametrize("backend_type", [OnnxBackend, MergedOnnxBackend])
@pytest.mark.parametrize("operation", ["load", "count", "query", "passage"])
def test_unsupported_backend_never_reads_artifacts(tmp_path, monkeypatch, backend_type, operation):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported backend attempted IO")

    backend = backend_type(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="onnx_not_supported_for_jina_v5"):
        if operation == "load":
            backend.load()
        elif operation == "count":
            backend.count("text")
        else:
            backend.embed(["text"], operation)
    assert not backend.loaded and not backend.tokenizer_available
    assert backend.revision == ""


@pytest.mark.parametrize("backend", ["onnx", "all", "onnx-merged"])
def test_no_onnx_provisioning(backend):
    with pytest.raises(ValueError, match="onnx_not_supported_for_jina_v5"):
        artifact_patterns(backend)


def test_no_merged_manifest_is_accepted(tmp_path):
    with pytest.raises(ValueError, match="onnx_not_supported_for_jina_v5"):
        verify_manifest(tmp_path)


@pytest.mark.parametrize("backend_type", [OnnxBackend, MergedOnnxBackend])
def test_unsupported_backend_not_ready(tmp_path, settings, auth, backend_type):
    with TestClient(create_app(settings, backend_type(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready", headers=auth).status_code == 503


@pytest.mark.parametrize("force", [False, True])
def test_export_cannot_modify_existing_output(tmp_path, monkeypatch, capsys, force):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "model.onnx"
    sentinel.write_bytes(b"existing artifacts stay intact")
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/export_merged_onnx.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--model-root",
            str(tmp_path / "absent"),
            "--output-dir",
            str(output),
            *(["--force"] if force else []),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        namespace["main"]()
    assert exc.value.code == 2
    assert "onnx_not_supported_for_jina_v5" in capsys.readouterr().err
    assert sentinel.read_bytes() == b"existing artifacts stay intact"
