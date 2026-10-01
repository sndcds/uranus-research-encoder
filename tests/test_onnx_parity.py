"""Explicit real-cache parity gate; ordinary CI never downloads model weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_onnx_torch_parity(tmp_path):
    root = os.environ.get("ENCODER_INTEGRATION_MODEL_ROOT")
    if not root:
        pytest.skip("set ENCODER_INTEGRATION_MODEL_ROOT to a pinned cache with both backends")
    assert os.environ.get("JINA_NONCOMMERCIAL") == "1"
    report = tmp_path / "parity.json"
    script = Path(__file__).resolve().parents[1] / "scripts/compare_backends.py"
    subprocess.run(
        [sys.executable, str(script), "--model-root", root, "--output", str(report)], check=True
    )
    result = json.loads(report.read_text())
    assert result["passed"]
    assert len(result["metrics"]) == 26


@pytest.mark.integration
def test_real_merged_three_backend_parity(tmp_path):
    root = os.environ.get("ENCODER_INTEGRATION_MODEL_ROOT")
    merged = os.environ.get("ENCODER_INTEGRATION_MERGED_ROOT")
    if not root or not merged:
        pytest.skip("set both integration cache roots; weights are never downloaded")
    assert os.environ.get("JINA_NONCOMMERCIAL") == "1"
    report = tmp_path / "merged-parity.json"
    script = Path(__file__).resolve().parents[1] / "scripts/compare_backends.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-root",
            root,
            "--merged-root",
            merged,
            "--output",
            str(report),
        ],
        check=True,
    )
    result = json.loads(report.read_text())
    assert result["passed"]
    assert len(result["comparisons"]) == 2
