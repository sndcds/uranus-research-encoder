"""Provisioning uses the same pin/allowlist as runtime; tests never download."""

import runpy
import sys
from pathlib import Path

import pytest

from uranus_research_encoder.model_artifacts import artifact_patterns
from uranus_research_encoder.version import MODEL_REPOSITORY, MODEL_REVISION


def test_prefetch_pins_revision_and_allowlist(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / MODEL_REVISION)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setenv("JINA_NONCOMMERCIAL", "1")
    monkeypatch.setattr(sys, "argv", ["prefetch", "--model-root", str(tmp_path)])
    script = Path(__file__).parents[1] / "scripts/prefetch_model.py"
    runpy.run_path(str(script))["main"]()
    assert calls == [
        dict(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            cache_dir=str(tmp_path),
            allow_patterns=artifact_patterns("torch"),
        )
    ]


@pytest.mark.parametrize("backend", ["onnx", "all", "onnx-merged"])
def test_prefetch_rejects_unsupported_backend(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("JINA_NONCOMMERCIAL", "1")
    monkeypatch.setattr(
        sys, "argv", ["prefetch", "--model-root", str(tmp_path), "--backend", backend]
    )
    script = Path(__file__).parents[1] / "scripts/prefetch_model.py"
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script))["main"]()
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []
