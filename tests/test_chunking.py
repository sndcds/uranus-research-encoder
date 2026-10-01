import hashlib

import pytest

from uranus_research_encoder.chunking import chunk_sections
from uranus_research_encoder.contracts import Section


def count(text):
    return len(text) + 2  # Character-token fake with two model special tokens.


def test_hash_limit_overlap_and_order():
    text = "".join(chr(0x400 + i) for i in range(1000))
    sections = [Section(kind="content", text=text)]
    chunks = chunk_sections(sections, count)
    assert chunks == chunk_sections(sections, count)
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.content_hash == hashlib.sha256(chunk.text.encode()).hexdigest()
        assert chunk.token_count == count(chunk.text) <= 480
    assert chunks[0].text[-62:] == chunks[1].text[:62]
    assert chunks[0].text + chunks[1].text[62:] + chunks[2].text[62:] == text


def test_boundaries():
    text = "A" * 350 + ".\n\n" + "B" * 300
    assert chunk_sections([Section(kind="content", text=text)], count)[0].text.endswith(".")


def test_context_union_and_kinds():
    sections = [
        Section(kind="tickets", text="same", context=context)
        for context in [
            {"scope": "venue", "venue_id": "00000000-0000-0000-0000-000000000002"},
            {"scope": "event"},
            {"scope": "event"},
        ]
    ]
    chunks = chunk_sections(sections, count)
    assert len(chunks) == 1
    assert chunks[0].chunk_kind == "tickets"
    assert [c.scope for c in chunks[0].contexts] == ["event", "venue"]
    assert chunks[0].contexts == chunk_sections(list(reversed(sections)), count)[0].contexts
    sections.append(Section(kind="content", text="same", context={"scope": "event"}))
    assert len(chunk_sections(sections, count)) == 2


def test_contextual_sections_independent():
    sections = [Section(kind="content", text=t, context={"scope": "event"}) for t in ["a", "b"]]
    assert [c.text for c in chunk_sections(sections, count)] == ["a", "b"]


def test_noncontextual_small_combined():
    sections = [Section(kind="tickets", text="a"), Section(kind="additional", text="b")]
    chunks = chunk_sections(sections, count)
    assert [(c.chunk_kind, c.text, c.contexts) for c in chunks] == [("content", "a\n\nb", [])]


def test_impossible_tokenizer():
    with pytest.raises(ValueError):
        chunk_sections([Section(kind="content", text="abc")], lambda _: 900)


def test_chunk_limit():
    sections = [Section(kind="content", text="x" * 200000)]
    # A model-token count that fits few characters, causing >1000 chunks.
    with pytest.raises(ValueError, match="chunk_limit"):
        chunk_sections(sections, lambda text: len(text) * 10 + 2)


def test_mixed_context_guard():
    with pytest.raises(ValueError, match="mixed_context"):
        chunk_sections(
            [
                Section(kind="content", text="a", context={"scope": "event"}),
                Section(kind="content", text="b"),
            ],
            count,
        )


def test_full_context_identity_union():
    space = "00000000-0000-0000-0000-000000000003"
    venue = "00000000-0000-0000-0000-000000000002"
    occurrence = "00000000-0000-0000-0000-000000000004"
    contexts = [
        {"scope": "space", "space_id": space},
        {"scope": "space", "space_id": space, "venue_id": venue},
        {"scope": "occurrence", "occurrence_id": occurrence},
        {"scope": "occurrence", "occurrence_id": occurrence, "space_id": space, "venue_id": venue},
    ]
    sections = [Section(kind="facilities", text="Shared prose", context=c) for c in contexts]
    chunks = chunk_sections(sections, count)
    assert len(chunks) == 1
    assert set(chunks[0].contexts) == {s.context for s in sections}
    assert len(chunks[0].contexts) == 4


def test_document_sections_never_lose_text_at_split():
    # Cover multilingual text and paragraph/sentence boundaries with unique markers.
    text = "\n\n".join(f"Gemeinde {i}: Flensburg — Räume 活动. Ende!" for i in range(70))
    chunks = chunk_sections([Section(kind="content", text=text)], count)
    assert all(c.token_count <= 480 for c in chunks)
    assert all(f"Gemeinde {i}:" in " ".join(c.text for c in chunks) for i in range(70))
