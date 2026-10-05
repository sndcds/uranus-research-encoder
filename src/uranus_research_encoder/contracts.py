"""Strict, bounded HTTP contracts independent of the embedding implementation."""

import math
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_TEXTS = 64
MAX_DOCUMENTS = 4
MAX_SECTIONS = 1000
MAX_TEXT_CHARS = 200_000
MAX_TOTAL_CHARS = 800_000
MAX_CHUNKS = 1000
MAX_TOKENS = 480
OVERLAP = 64
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024

Kind = Literal[
    "content",
    "participation",
    "accessibility",
    "tickets",
    "additional",
    "facilities",
    "location_context",
    "activities",
    "categories",
]
EmbeddingKind = Literal["query", "passage"]
Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_TEXT_CHARS)]
Vector = Annotated[
    list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=1024, max_length=1024)
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceContext(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: Literal["event", "venue", "space", "occurrence"]
    venue_id: UUID | None = None
    space_id: UUID | None = None
    occurrence_id: UUID | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        match self.scope:
            case "event":
                valid = all(x is None for x in (self.venue_id, self.space_id, self.occurrence_id))
            case "venue":
                valid = self.venue_id is not None and self.space_id is self.occurrence_id is None
            case "space":
                valid = self.space_id is not None and self.occurrence_id is None
            case "occurrence":
                valid = self.occurrence_id is not None
        if not valid:
            raise ValueError("invalid_context")
        return self


def check_text(text: str) -> None:
    # Surrogate escapes are valid JSON but cannot be hashed as UTF-8.
    if not text.strip() or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        raise ValueError("invalid_text")


class Section(Contract):
    kind: Kind
    context: EvidenceContext | None = None
    text: Text

    @model_validator(mode="after")
    def validate_text(self) -> Self:
        check_text(self.text)
        return self


class ChunkDocument(Contract):
    entity_id: UUID
    sections: list[Section] = Field(min_length=1, max_length=MAX_SECTIONS)

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        contextual = [s.context is not None for s in self.sections]
        if any(contextual) and not all(contextual):
            raise ValueError("mixed_context")
        if sum(len(s.text) for s in self.sections) > MAX_TEXT_CHARS:
            raise ValueError("document_limit")
        return self


class EmbedRequest(Contract):
    model: Literal["jina-v5"]
    texts: list[Text] = Field(min_length=1, max_length=MAX_TEXTS)
    kind: EmbeddingKind

    @model_validator(mode="after")
    def validate_texts(self) -> Self:
        if sum(map(len, self.texts)) > MAX_TOTAL_CHARS:
            raise ValueError("text_limit")
        for text in self.texts:
            check_text(text)
        return self


class ChunkRequest(Contract):
    model: Literal["jina-v5"]
    documents: list[ChunkDocument] = Field(min_length=1, max_length=MAX_DOCUMENTS)

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        if sum(len(s.text) for d in self.documents for s in d.sections) > MAX_TOTAL_CHARS:
            raise ValueError("text_limit")
        return self


class Chunk(Contract):
    chunk_index: int = Field(ge=0, lt=MAX_CHUNKS)
    chunk_kind: Kind
    contexts: list[EvidenceContext] = Field(default_factory=list, max_length=MAX_SECTIONS)
    text: Text
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1, le=MAX_TOKENS)


class ChunkResult(Contract):
    entity_id: UUID
    chunks: list[Chunk] = Field(min_length=1, max_length=MAX_CHUNKS)


class ChunkResponse(Contract):
    embedding_version: str
    documents: list[ChunkResult] = Field(min_length=1, max_length=MAX_DOCUMENTS)


class EmbedMetrics(Contract):
    text_count: int = Field(ge=1, le=MAX_TEXTS)
    token_count: int = Field(ge=1)


class EmbedResponse(Contract):
    embedding_version: str
    vectors: list[Vector] = Field(min_length=1, max_length=MAX_TEXTS)
    metrics: EmbedMetrics

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        if len(self.vectors) != self.metrics.text_count:
            raise ValueError("invalid_vectors")
        for vector in self.vectors:
            if abs(math.sqrt(math.fsum(x * x for x in vector)) - 1) > 1e-5:
                raise ValueError("invalid_norm")
        return self
