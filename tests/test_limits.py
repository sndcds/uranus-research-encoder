import pytest


@pytest.mark.parametrize("body", [b"{", b"null", b"[]", b'{"model":"unknown"}'])
def test_invalid_json_generic(client, auth, body):
    response = client.post("/embed", headers=auth, content=body)
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request"}


@pytest.mark.parametrize("texts", [["x"] * 65, ["x" * 200001], ["x" * 200000] * 5, [" "]])
def test_text_limits(client, auth, texts):
    assert (
        client.post(
            "/embed",
            headers=auth,
            json={
                "model": "jina-v5",
                "texts": texts,
                "kind": "query",
            },
        ).status_code
        == 422
    )


def test_body_limit(client, auth):
    response = client.post("/embed", headers=auth, content=b"x" * (4 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_no_truncation(client, auth, backend):
    response = client.post(
        "/embed",
        headers=auth,
        json={
            "model": "jina-v5",
            "texts": ["x" * 32768],
            "kind": "query",
        },
    )
    assert response.status_code == 422
    assert backend.calls == []


def test_streamed_body_limit(client, auth):
    def parts():
        for _ in range(5):
            yield b"x" * 1024 * 1024

    assert client.post("/chunks", headers=auth, content=parts()).status_code == 413


@pytest.mark.parametrize("which", ["documents", "sections", "document_text", "mixed"])
def test_document_limits(client, auth, which):
    section = {"kind": "content", "text": "hi"}
    doc = {"entity_id": "00000000-0000-0000-0000-000000000001", "sections": [section]}
    docs = [doc]
    if which == "documents":
        docs = [doc] * 5
    elif which == "sections":
        doc["sections"] = [section] * 1001
    elif which == "document_text":
        doc["sections"] = [{"kind": "content", "text": "x" * 100001}] * 2
    else:
        doc["sections"].append({"kind": "content", "text": "bye", "context": {"scope": "event"}})
    response = client.post("/chunks", headers=auth, json={"model": "jina-v5", "documents": docs})
    assert response.status_code == 422
    assert response.json() == {"error": "invalid_request"}


def test_compressed_body_rejected(client, auth):
    response = client.post("/embed", headers=auth | {"Content-Encoding": "gzip"}, content=b"x")
    assert response.status_code == 415


def test_response_limit(client, auth, monkeypatch):
    monkeypatch.setattr("uranus_research_encoder.app.MAX_RESPONSE_BYTES", 8)
    response = client.post(
        "/embed", headers=auth, json={"model": "jina-v5", "texts": ["hi"], "kind": "query"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": "response_limit"}
