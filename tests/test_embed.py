import math

import pytest


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_embeddings(client, auth, backend, kind):
    data = {"model": "jina-v3", "texts": ["hello", "world"], "kind": kind}
    response = client.post("/embed", headers=auth, json=data)
    assert response.status_code == 200
    assert response.json() == client.post("/embed", headers=auth, json=data).json()
    assert backend.calls[-1] == (["hello", "world"], kind)
    for vector in response.json()["vectors"]:
        assert len(vector) == 1024
        assert all(math.isfinite(x) for x in vector)
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1, abs=1e-6)
    assert set(response.json()["metrics"]) == {"text_count", "token_count"}
