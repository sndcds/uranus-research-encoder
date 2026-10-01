"""Reviewed compatibility identifiers. Never derive the embedding version."""

SERVICE_VERSION = "0.1.0"
CONTRACT_VERSION = "uranus-research-encoder-v1"
MODEL = "jina-v3"
MODEL_REPOSITORY = "jinaai/jina-embeddings-v3-hf"
MODEL_REVISION = "d18862d9a48706220815554fac3ebb4dfa46fc28"
DIMENSIONS = 1024
CHUNK_VERSION = "sections-480-overlap64-v2"
EMBEDDING_VERSION = (
    "d18862d9a48706220815554fac3ebb4dfa46fc28:"
    "native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2"
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
