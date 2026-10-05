"""Historical v3 report tests validate gate arithmetic, not v5 ONNX parity."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
acceptance = runpy.run_path(str(ROOT / "scripts/compare_backends.py"))["acceptance"]


@pytest.fixture
def observed():
    return json.loads((ROOT / "validation/onnx-parity.json").read_text())


def test_observed_pinned_embeddings_meet_frozen_gate(observed):
    assert acceptance(observed)


@pytest.mark.parametrize(
    "change",
    [
        {"max_absolute_difference": 0.01},
        {"min_cosine_similarity": 0.99},
        {"rankings_identical": False},
        {"task_specific": False},
        {"stable": False},
        {"token_counts_match": False},
    ],
)
def test_incompatible_embeddings_fail_gate(observed, change):
    observed.update(change)
    assert not acceptance(observed)


def test_unnormalized_embeddings_fail_gate(observed):
    changed = copy.deepcopy(observed)
    changed["metrics"][0]["onnx_norm"] = 0.8
    assert not acceptance(changed)
