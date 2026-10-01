def test_health_without_auth(client, backend):
    assert client.get("/health").json() == {"status": "ok"}
    assert backend.calls == []


def test_readiness_and_version(client, auth):
    ready = client.get("/ready", headers=auth)
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    version = client.get("/version", headers=auth).json()
    assert ready.json()["embedding_version"] == version["embedding_version"]
    assert ready.json()["contract_version"] == version["contract_version"]
    assert set(version) == {
        "service_version",
        "contract_version",
        "model",
        "model_repository",
        "model_revision",
        "dimensions",
        "embedding_version",
        "chunk_version",
    }


def test_docs_and_redirects_disabled(client):
    for path in ["/docs", "/redoc", "/openapi.json", "/health/"]:
        assert client.get(path, follow_redirects=False).status_code in (401, 404)


def test_no_cors(client, auth):
    response = client.get("/version", headers=auth | {"Origin": "https://example.org"})
    assert "access-control-allow-origin" not in response.headers


def test_dev_schema_opt_in_still_authenticated(settings, backend, auth):
    from fastapi.testclient import TestClient

    from uranus_research_encoder.app import create_app

    with TestClient(
        create_app(settings.model_copy(update={"enable_docs": True}), backend)
    ) as client:
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/openapi.json", headers=auth).status_code == 200
