from typing import get_args
from uuid import UUID

import pytest
from pydantic import ValidationError

from uranus_research_encoder.contracts import ChunkDocument, EmbedRequest, Kind, Section


@pytest.mark.parametrize("kind", get_args(Kind))
def test_canonical_kinds(kind):
    assert Section(kind=kind, text="hello").kind == kind


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "unknown", "text": "hello"},
        {"kind": "content", "text": "hello", "extra": 1},
        {"kind": "content", "text": ""},
        {"kind": "content", "text": "  \n"},
        {"kind": "content", "text": "\ud800"},
    ],
)
def test_bad_section(data):
    with pytest.raises(ValidationError):
        Section(**data)


def test_mixed_context_rejected():
    with pytest.raises(ValidationError):
        ChunkDocument(
            entity_id=UUID(int=1),
            sections=[
                Section(kind="content", text="one", context={"scope": "event"}),
                Section(kind="content", text="two"),
            ],
        )


@pytest.mark.parametrize(
    "change", [{"model": "remote"}, {"kind": "other"}, {"texts": [""]}, {"extra": 1}]
)
def test_embed_contract(change):
    with pytest.raises(ValidationError):
        EmbedRequest(**({"model": "jina-v3", "kind": "query", "texts": ["hi"]} | change))
