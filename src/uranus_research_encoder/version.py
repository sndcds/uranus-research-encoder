"""Reviewed compatibility identifiers. Never derive the embedding version."""

SERVICE_VERSION = "0.2.0"
CONTRACT_VERSION = "uranus-research-encoder-v1"
MODEL = "jina-v5"
MODEL_REPOSITORY = "jinaai/jina-embeddings-v5-text-small"
MODEL_REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
DIMENSIONS = 1024
CHUNK_VERSION = "sections-480-overlap64-v2"
EMBEDDING_VERSION = (
    "dd76d535f5447ca3897a9c893fb1e612ead98192:"
    "native-qwen3-torch2.11.0-transformers5.17.0-peft0.21.1-cpu-eager-"
    "retrieval-query-document-last-token-l2-f32-d1024:sections-480-overlap64-v2"
)


def metadata() -> dict[str, str | int]:
    return {
        "service_version": SERVICE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "model": MODEL,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "dimensions": DIMENSIONS,
        "embedding_version": EMBEDDING_VERSION,
        "chunk_version": CHUNK_VERSION,
    }
