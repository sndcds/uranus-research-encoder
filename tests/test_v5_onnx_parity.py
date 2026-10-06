"""Differential metric and ranking gate tests independent of model weights."""

import importlib.util
from pathlib import Path

import numpy as np


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/check_v5_onnx_parity.py"
    spec = importlib.util.spec_from_file_location("v5_parity_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_tie_break_matches_research_event_aggregation():
    m = runner()
    docs = [
        {"id": "z", "event_id": "b"},
        {"id": "a", "event_id": "b"},
        {"id": "m", "event_id": "a"},
    ]
    result = m.aggregate([0.5, 0.5, 0.5], docs)
    assert [(r["event_id"], r["point_id"]) for r in result] == [("a", "m"), ("b", "a")]
    # A nonzero difference must not be rounded into a tie.
    result = m.aggregate([0.5 + 1e-12, 0.5, 0.5], docs)
    assert [(r["event_id"], r["point_id"]) for r in result] == [("b", "z"), ("a", "m")]


def test_metrics_reject_component_and_norm_failures():
    import json

    m = runner()
    t = json.loads(
        (Path(__file__).resolve().parents[1] / "validation/jina-v5-merged-v1/plan.json").read_text()
    )["tolerances"]
    a = np.ones(1024) / 32
    assert m.compare(a, a, t)["pass"]
    b = a.copy()
    b[0] += 2e-5
    assert not m.compare(a, b, t)["pass"]
    assert not m.compare(a, a * 1.01, t)["pass"]


def test_performance_refuses_failed_parity(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    report = tmp_path / "parity.json"
    report.write_text(
        json.dumps({"pass": False, "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()})
    )
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_v5_onnx.py"
    output = tmp_path / "performance.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "onnx",
            "--model-root",
            str(tmp_path),
            "--graph-dir",
            str(tmp_path),
            "--plan-dir",
            str(tmp_path),
            "--parity-report",
            str(report),
            "--contract-report",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "parity_gate_not_passed" in result.stderr
    assert not output.exists()
