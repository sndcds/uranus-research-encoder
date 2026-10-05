"""Opt-in local container smoke probe; reads only the mounted test key and HTTP API."""

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

assert os.getuid() == 10001
assert os.statvfs("/").f_flag & os.ST_RDONLY
interfaces = sorted(path.name for path in Path("/sys/class/net").iterdir())
assert interfaces == ["lo"]

key = Path("/run/secrets/embedding.key").read_text().strip()


def request(path, body=None, authenticated=True):
    headers = {"Authorization": "Bearer " + key} if authenticated else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


assert request("/health", authenticated=False) == (200, {"status": "ok"})
assert request("/ready", authenticated=False)[0] == 401
status, ready = request("/ready")
assert status == 200, ready
status, version = request("/version")
assert status == 200
assert version["model"] == "jina-v5"
assert version["model_revision"] == "dd76d535f5447ca3897a9c893fb1e612ead98192"
assert version["service_version"] == "0.2.0"
assert version["chunk_version"] == "sections-480-overlap64-v2"
assert version["dimensions"] == 1024
assert version["embedding_version"] == ready["embedding_version"]
text = "Barrierefreie Veranstaltung mit Gebärdensprache in Flensburg."
results = {}
for kind in ("query", "passage"):
    body = {"model": "jina-v5", "kind": kind, "texts": [text]}
    status, result = request("/embed", body)
    assert status == 200, result
    assert result == request("/embed", body)[1]
    assert result["embedding_version"] == version["embedding_version"]
    (vector,) = result["vectors"]
    assert len(vector) == 1024 and all(math.isfinite(v) for v in vector)
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1) < 1e-6
    results[kind] = result
assert results["query"]["vectors"] != results["passage"]["vectors"]
body = {
    "model": "jina-v5",
    "documents": [
        {
            "entity_id": "00000000-0000-0000-0000-000000000001",
            "sections": [
                {
                    "kind": "accessibility",
                    "context": {"scope": "event"},
                    "text": (text + "\n\n") * 100,
                }
            ],
        }
    ],
}
status, chunk_result = request("/chunks", body)
assert status == 200, chunk_result
assert chunk_result == request("/chunks", body)[1]
chunks = chunk_result["documents"][0]["chunks"]
assert len(chunks) > 1
for index, chunk in enumerate(chunks):
    assert chunk["chunk_index"] == index
    assert chunk["chunk_kind"] == "accessibility"
    assert chunk["token_count"] <= 480
    assert chunk["content_hash"] == hashlib.sha256(chunk["text"].encode()).hexdigest()
assert request("/embed", {"model": "jina-v3", "kind": "query", "texts": [text]})[0] == 422
print(
    json.dumps(
        {
            "status": "passed",
            "version": version,
            "health": 200,
            "unauthenticated_ready": 401,
            "ready": 200,
            "query_tokens": results["query"]["metrics"]["token_count"],
            "passage_tokens": results["passage"]["metrics"]["token_count"],
            "chunks": len(chunks),
            "normalized_1024": True,
            "deterministic": True,
            "query_passage_distinct": True,
            "network_interfaces": interfaces,
            "root": "read-only",
            "uid": os.getuid(),
        },
        indent=2,
    )
)
