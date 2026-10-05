import math

import pytest


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_embeddings(client, auth, backend, kind):
    data = {"model": "jina-v5", "texts": ["hello", "world"], "kind": kind}
    response = client.post("/embed", headers=auth, json=data)
    assert response.status_code == 200
    assert response.json() == client.post("/embed", headers=auth, json=data).json()
    assert backend.calls[-1] == (["hello", "world"], kind)
    for vector in response.json()["vectors"]:
        assert len(vector) == 1024
        assert all(math.isfinite(x) for x in vector)
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1, abs=1e-6)
    assert set(response.json()["metrics"]) == {"text_count", "token_count"}


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_api_counts_requested_kind(client, auth, backend, monkeypatch, kind):
    calls = []

    def count(text, selected_kind="passage"):
        calls.append(selected_kind)
        return 7 if selected_kind == "query" else 9

    monkeypatch.setattr(backend, "count", count)
    result = client.post(
        "/embed", headers=auth, json={"model": "jina-v5", "kind": kind, "texts": ["text"]}
    )
    assert result.status_code == 200
    assert calls == [kind]
    assert result.json()["metrics"]["token_count"] == (7 if kind == "query" else 9)


def test_previous_embedding_space_rejected(client, auth):
    assert (
        client.post(
            "/embed", headers=auth, json={"model": "jina-v3", "kind": "query", "texts": ["text"]}
        ).status_code
        == 422
    )
