import hashlib

import pytest
from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app
from uranus_research_encoder.config import Settings
from uranus_research_encoder.version import MODEL_REVISION

KEY = "test-only-encoder-key-0123456789abcdef"


class FakeBackend:
    max_tokens = 8192
    revision = MODEL_REVISION
    loaded = False
    tokenizer_available = False

    def __init__(self):
        self.calls = []
        self.loads = 0

    def load(self):
        self.loads += 1
        self.loaded = self.tokenizer_available = True

    def count(self, text):
        return len(text) + 2

    def embed(self, texts, kind):
        self.calls.append((texts, kind))
        # Signed uniform vector: exactly representable float32, exactly unit norm.
        return [
            [
                (1 if b & 1 else -1) / 32
                for b in hashlib.shake_256((kind + text).encode()).digest(1024)
            ]
            for text in texts
        ]


@pytest.fixture
def settings():
    return Settings(api_key=KEY, jina_noncommercial=True)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def auth():
    return {"Authorization": "Bearer " + KEY}


@pytest.fixture
def client(settings, backend):
    with TestClient(create_app(settings, backend), raise_server_exceptions=False) as client:
        yield client
