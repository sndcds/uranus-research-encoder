"""Exercise failure handling without loading models or running timings."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("returncode", [-9, 0])
def test_failed_or_empty_worker_stops_sweep(tmp_path, monkeypatch, returncode):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_backends.py"))
    monkeypatch.setattr(namespace["platform"], "platform", lambda: "test-platform")
    monkeypatch.setattr(namespace["platform"], "processor", lambda: "test-cpu")
    output = tmp_path / "report.json"
    monkeypatch.setenv("JINA_NONCOMMERCIAL", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_backends.py",
            "--model-root",
            str(tmp_path / "cache"),
            "--merged-root",
            str(tmp_path / "derived"),
            "--output",
            str(output),
            "--backends",
            "onnx-merged",
            "--thread-sweep",
            "1",
            "2",
        ],
    )
    calls = []

    def failed_worker(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(namespace["subprocess"], "run", failed_worker)
    with pytest.raises(SystemExit, match="benchmark incomplete"):
        namespace["main"]()
    assert len(calls) == 1
    (result,) = json.loads(output.read_text())["results"]
    assert result["requested_intra_op_threads"] == 1
    assert result["intra_op_threads"] == 1
    assert result["exit_code"] == returncode
    assert not result["complete"]
    assert result["expected_cases"] == 16
    assert result["cases"] == []
