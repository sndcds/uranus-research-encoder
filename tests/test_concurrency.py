from concurrent.futures import ThreadPoolExecutor
from threading import Event

from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app


def test_busy_requests_do_not_queue(settings, backend, auth):
    entered, release = Event(), Event()
    original = backend.embed

    def blocked(texts, kind):
        entered.set()
        assert release.wait(10)
        return original(texts, kind)

    backend.embed = blocked
    data = {"model": "jina-v3", "texts": ["hello"], "kind": "query"}
    settings = settings.model_copy(update={"max_concurrent_requests": 1})
    with TestClient(create_app(settings, backend)) as client, ThreadPoolExecutor() as pool:
        first = pool.submit(client.post, "/embed", headers=auth, json=data)
        try:
            assert entered.wait(5)
            second = client.post("/embed", headers=auth, json=data)
            assert second.status_code == 503
            assert second.json() == {"error": "busy"}
            assert client.get("/health").status_code == 200
            assert client.get("/ready", headers=auth).status_code == 200
        finally:
            release.set()
        assert first.result().status_code == 200
        assert client.post("/embed", headers=auth, json=data).status_code == 200
