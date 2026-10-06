"""Full-corpus eligibility, strict ordering, tie evidence and frozen input guards."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reporter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "full_corpus_reporter", ROOT / "scripts/report_v5_full_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(eid, text="text", contexts=None):
    return {"entity_id": eid, "chunk": {"text": text, "contexts": contexts or [{"scope": "event"}]}}


def test_event_and_occurrence_eligibility_both_apply(reporter):
    chunks = {
        "a": chunk("e1"),
        "b": chunk("e1", contexts=[{"scope": "occurrence", "occurrence_id": "old"}]),
        "c": chunk("e1", contexts=[{"scope": "occurrence", "occurrence_id": "allowed"}]),
        "d": chunk("ineligible", contexts=[{"scope": "occurrence", "occurrence_id": "allowed"}]),
        "e": chunk(
            "e1",
            contexts=[
                {"scope": "occurrence", "occurrence_id": "old"},
                {"scope": "occurrence", "occurrence_id": "allowed"},
            ],
        ),
    }
    inputs = {"point_ids": list(chunks), "chunks": chunks}
    profile = {"eligible_ids": ["e1"], "eligible_occurrence_ids": ["allowed"]}
    assert reporter.eligible_points(inputs, profile) == ["a", "c", "e"]


def test_cosine_reduction_preserves_identical_vector_ties(reporter):
    rng = np.random.default_rng(13)
    v = rng.normal(size=1024).astype("float32")
    q = rng.normal(size=1024).astype("float32")
    actual = reporter.cosine_scores(np.tile(v, (2080, 1)), q)
    assert np.all(actual == actual[0])
    assert actual.dtype == np.float64
    expected = (
        np.dot(v.astype("float64"), q.astype("float64"))
        / np.linalg.norm(v.astype("float64"))
        / np.linalg.norm(q.astype("float64"))
    )
    assert actual[0] == pytest.approx(expected, abs=1e-15)


def test_exact_ties_use_point_then_event_ids(reporter):
    chunks = {"z": chunk("b"), "a": chunk("b"), "m": chunk("a")}
    scores = np.asarray([0.5, 0.5, 0.5])
    r = reporter.compare_rankings(list(chunks), chunks, scores, scores)
    assert r["native_top20"] == ["a", "b"]
    assert r["ordering_sha256"]["native_chunks"] == reporter.digest(["a", "m", "z"])
    assert r["ordering_sha256"]["native_winners"] == reporter.digest({"a": "m", "b": "a"})
    assert r["chunk_winners_identical"] and r["full_ranking_identical"]


def test_near_tie_winner_change_is_never_hidden(reporter):
    chunks = {"a": chunk("e", "first"), "b": chunk("e", "second")}
    r = reporter.compare_rankings(
        list(chunks), chunks, np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5 + 1e-12])
    )
    assert r["full_ranking_identical"] and r["top10_identical"]
    assert not r["chunk_winners_identical"] and not r["chunk_ranking_identical"]
    assert r["winner_differences"][0]["native_exact_tie"]
    assert not r["winner_differences"][0]["onnx_exact_tie"]
    assert r["exact_ties_involving_changed_ids"]["native_chunks"] == [
        {"score": 0.5, "ordered_ids": ["a", "b"]}
    ]
    assert len(r["chunk_rank_differences"]) == 2


def test_unchanged_top20_does_not_mask_full_ranking_failure(reporter):
    chunks = {f"p{i:02}": chunk(f"e{i:02}") for i in range(22)}
    native = np.arange(22, 0, -1, dtype=np.float64) / 25
    onnx = native.copy()
    onnx[-2:] = onnx[-2:][::-1]
    r = reporter.compare_rankings(list(chunks), chunks, native, onnx)
    assert r["top10_identical"] and r["top20_identical"]
    assert not r["full_ranking_identical"]
    assert r["max_event_rank_difference"] == 1
    assert {d["id"] for d in r["event_rank_differences"]} == {"e20", "e21"}


def test_full_frozen_coverage_and_unchanged_thresholds(reporter):
    base = ROOT / "validation/jina-v5-merged-v1"
    plan = json.loads((base / "full-corpus-plan.json").read_text())
    inputs = json.loads((base / "full-corpus-inputs.json").read_text())
    parent = json.loads((base / "plan.json").read_text())
    assert reporter.sha(base / "full-corpus-inputs.json") == plan["inputs_sha256"]
    assert reporter.sha(base / "plan.json") == plan["parent_plan_sha256"]
    assert plan["tolerances"] == parent["tolerances"]
    assert plan["max_score_difference"] == parent["ranking"]["max_score_difference"]
    assert len(inputs["documents"]) == 611
    assert len(inputs["chunks"]) == len(set(inputs["point_ids"])) == 2080
    assert len(inputs["queries"]) == 120
    assert len({r["chunk"]["text"] for r in inputs["chunks"].values()}) == 1094
    assert not inputs["judgments_used"] and not inputs["phase2d_used"]
    comparisons = 0
    covered = set()
    for q in inputs["queries"]:
        p = inputs["eligibility_profiles"][q["profile"]]
        allowed = reporter.eligible_points(inputs, p)
        assert len(allowed) == p["candidate_count"]
        assert reporter.digest(allowed) == p["eligible_point_ids_sha256"]
        comparisons += len(allowed)
        covered.update(allowed)
    assert comparisons == 238314 and covered == set(inputs["point_ids"])
    for name, expected in plan["encoder_source_sha256"].items():
        assert reporter.sha(ROOT / name) == expected


def test_changed_input_stops_before_vectors_are_loaded(reporter, tmp_path):
    (tmp_path / "full-corpus-plan.json").write_text(json.dumps({"inputs_sha256": "0" * 64}))
    (tmp_path / "full-corpus-inputs.json").write_text("{}")
    with pytest.raises(ValueError, match="input_hash"):
        reporter.evaluate(tmp_path, tmp_path / "absent-vectors", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()
