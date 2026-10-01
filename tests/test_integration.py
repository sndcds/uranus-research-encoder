"""Opt-in real-model acceptance; no network or prefetch here."""

import math
import os
from pathlib import Path

import pytest

from uranus_research_encoder.chunking import chunk_sections
from uranus_research_encoder.contracts import Section
from uranus_research_encoder.model import JinaBackend
from uranus_research_encoder.version import MODEL_REVISION


@pytest.mark.integration
def test_local_jina():
    root = os.environ.get("ENCODER_INTEGRATION_MODEL_ROOT")
    if not root:
        pytest.skip("set ENCODER_INTEGRATION_MODEL_ROOT to an existing pinned cache")
    assert os.environ.get("JINA_NONCOMMERCIAL") == "1"
    backend = JinaBackend(Path(root))
    backend.load()  # Missing/incomplete cache is a failure when explicitly selected.
    assert backend.revision == MODEL_REVISION
    text = "Gemeinsames Communitytreffen in Flensburg."
    queries = backend.embed([text], "query")
    passages = backend.embed([text], "passage")
    assert queries == backend.embed([text], "query")
    assert passages == backend.embed([text, "Other input"], "passage")[:1]
    assert queries != passages
    for vector in queries + passages:
        assert len(vector) == 1024
        assert all(math.isfinite(x) for x in vector)
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1, abs=1e-5)
    chunks = chunk_sections(
        [Section(kind="content", context={"scope": "event"}, text=(text + "\n\n") * 200)],
        backend.count,
    )
    assert all(c.token_count == backend.count(c.text) <= 480 for c in chunks)
    assert all(c.contexts[0].scope == "event" for c in chunks)
