from uranus_research_encoder.version import CONTRACT_VERSION, EMBEDDING_VERSION


def test_pinned_version():
    assert CONTRACT_VERSION == "uranus-research-encoder-v1"
    assert EMBEDDING_VERSION == (
        "dd76d535f5447ca3897a9c893fb1e612ead98192:"
        "native-qwen3-torch2.11.0-transformers5.17.0-peft0.21.1-cpu-eager-"
        "retrieval-query-document-last-token-l2-f32-d1024:sections-480-overlap64-v2"
    )


def test_package_and_service_versions_match():
    import tomllib
    from pathlib import Path

    from uranus_research_encoder.version import SERVICE_VERSION

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert project["project"]["version"] == SERVICE_VERSION


def test_model_identity_and_space():
    from uranus_research_encoder.version import (
        CHUNK_VERSION,
        DIMENSIONS,
        MODEL,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        metadata,
    )

    assert MODEL == "jina-v5"
    assert MODEL_REPOSITORY == "jinaai/jina-embeddings-v5-text-small"
    assert MODEL_REVISION == "dd76d535f5447ca3897a9c893fb1e612ead98192"
    assert DIMENSIONS == 1024
    assert CHUNK_VERSION == "sections-480-overlap64-v2"
    assert metadata()["embedding_version"] == EMBEDDING_VERSION
