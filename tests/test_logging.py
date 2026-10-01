import json
import logging

from uranus_research_encoder.errors import JsonFormatter


def test_third_party_logs_do_not_render_secrets():
    record = logging.LogRecord(
        "uvicorn.error", logging.ERROR, "private_path", 1, "private %s", ("secret",), None
    )
    assert json.loads(JsonFormatter().format(record)) == {
        "event": "http_server",
        "level": "error",
        "logger": "uvicorn.error",
    }


def test_failure_logs_bounded(client, backend, auth, caplog):
    def broken(texts, kind):
        raise RuntimeError("private text and secret")

    backend.embed = broken
    with caplog.at_level(logging.WARNING, logger="encoder"):
        response = client.post(
            "/embed", headers=auth, json={"model": "jina-v3", "texts": ["hi"], "kind": "query"}
        )
    assert response.status_code == 500
    assert "request_failed" in caplog.text
    assert "private text and secret" not in caplog.text
