import pytest
from conftest import KEY, FakeBackend
from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app


def test_load_failure_keeps_health_and_version(settings, auth):
    class Broken(FakeBackend):
        def load(self):
            raise RuntimeError("sensitive /path " + KEY)

    with TestClient(create_app(settings, Broken())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready", headers=auth).status_code == 503
        assert client.get("/version", headers=auth).status_code == 200
        assert (
            client.post(
                "/embed", headers=auth, json={"model": "jina-v5", "texts": ["hi"], "kind": "query"}
            ).status_code
            == 503
        )


def test_bad_configuration_keeps_health(monkeypatch):
    monkeypatch.delenv("ENCODER_API_KEY_FILE", raising=False)
    monkeypatch.setenv("ENCODER_API_KEY", "bad")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


@pytest.mark.parametrize(
    "attribute,value", [("revision", "wrong"), ("loaded", False), ("tokenizer_available", False)]
)
def test_readiness_checks(client, backend, auth, attribute, value):
    setattr(backend, attribute, value)
    assert client.get("/ready", headers=auth).status_code == 503
    assert client.get("/health").status_code == 200


def test_loaded_once_and_readiness_no_inference(client, backend, auth):
    for _ in range(3):
        assert client.get("/ready", headers=auth).status_code == 200
    assert backend.loads == 1
    assert backend.calls == []


@pytest.mark.parametrize(
    "vector", [[float("nan")] * 1024, [float("inf")] * 1024, [0.0] * 1024, [1.0], [0.5] * 1024]
)
def test_invalid_backend_vector_is_not_exposed(client, backend, auth, vector):
    backend.embed = lambda texts, kind: [vector]
    response = client.post(
        "/embed", headers=auth, json={"model": "jina-v5", "texts": ["private"], "kind": "query"}
    )
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}
