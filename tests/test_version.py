from uranus_research_encoder.version import CONTRACT_VERSION, EMBEDDING_VERSION


def test_pinned_version():
    assert CONTRACT_VERSION == "uranus-research-encoder-v1"
    assert EMBEDDING_VERSION == (
        "d18862d9a48706220815554fac3ebb4dfa46fc28:"
        "native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2"
    )


def test_package_and_service_versions_match():
    import tomllib
    from pathlib import Path

    from uranus_research_encoder.version import SERVICE_VERSION

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert project["project"]["version"] == SERVICE_VERSION
