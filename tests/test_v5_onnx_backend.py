"""Explicit configuration, output validation and fail-closed API behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import KEY
from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app
from uranus_research_encoder.config import Settings
from uranus_research_encoder.v5_onnx_backend import V5MergedOnnxBackend


class Tokenizer:
    def __init__(self):
        self.inputs = []

    def encode(self, text, **kwargs):
        self.inputs.append(text)
        assert kwargs["truncation"] is False
        return list(range(len(text)))

    def __call__(self, text, **kwargs):
        ids = self.encode(text, **kwargs)
        return {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }


class Session:
    def __init__(self):
        self.vector = np.ones((1, 1024), dtype=np.float32) / 32
        self.calls = []

    def run(self, names, inputs):
        assert names == ["text_embeds"]
        assert all(v.dtype == np.int64 for v in inputs.values())
        self.calls.append(inputs)
        return [self.vector.copy()]

    def get_inputs(self):
        return [
            SimpleNamespace(name=n, type="tensor(int64)", shape=["batch", "sequence"])
            for n in ("input_ids", "attention_mask")
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="text_embeds", type="tensor(float)", shape=["batch", 1024])]

    def get_providers(self):
        return ["CPUExecutionProvider"]


@pytest.fixture
def fake_loaded(tmp_path):
    backend = V5MergedOnnxBackend(tmp_path, manifest_sha256="a" * 64)
    backend._tokenizer = Tokenizer()
    backend._session = Session()
    backend.loaded = True
    return backend


def test_exact_prefix_and_no_second_normalization(fake_loaded):
    backend = fake_loaded
    # A tiny allowed norm error exposes any unintended Python renormalization.
    backend._session.vector *= 1 + 3e-6
    expected = backend._session.vector[0].tolist()
    assert backend.embed(["eins", "zwei"], "query") == [expected, expected]
    assert backend.embed(["eins"], "passage") == [expected]
    assert backend._tokenizer.inputs == ["Query: eins", "Query: zwei", "Document: eins"]
    assert backend.count("eins", "query") == len("Query: eins")
    assert backend.count("eins", "passage") == len("Document: eins")
    with pytest.raises(ValueError, match="embedding_kind"):
        backend.embed(["eins"], "document")


def test_token_limit_before_inference(fake_loaded):
    fake_loaded.max_tokens = 4
    with pytest.raises(ValueError, match="token_limit"):
        fake_loaded.embed(["too long"], "query")
    assert not fake_loaded._session.calls


@pytest.mark.parametrize(
    "vector",
    [
        np.ones((1, 512), dtype=np.float32),
        np.full((1, 1024), np.nan, dtype=np.float32),
        np.zeros((1, 1024), dtype=np.float32),
        np.ones((1, 1024), dtype=np.float64) / 32,
    ],
)
def test_invalid_vectors_fail_closed(fake_loaded, vector):
    fake_loaded._session.vector = vector
    with pytest.raises(RuntimeError, match="invalid_vector"):
        fake_loaded.embed(["test"], "query")


@pytest.fixture
def fake_load_environment(tmp_path, monkeypatch):
    import onnxruntime
    from transformers import AutoTokenizer

    from uranus_research_encoder import v5_onnx_backend as module

    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(
        module,
        "verify_manifest",
        lambda *a: {"versions": {p: module.version(p) for p in ("onnxruntime", "transformers")}},
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    session = Session()
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **kw: session)
    return tmp_path, session


def test_load_then_failed_reload_clears_state(fake_load_environment, monkeypatch):
    from uranus_research_encoder import v5_onnx_backend as module

    root, session = fake_load_environment
    backend = V5MergedOnnxBackend(root, manifest_sha256="a" * 64)
    backend.load()
    assert backend.loaded and backend.tokenizer_available and backend._session is session

    def reject(*a):
        raise ValueError("manifest_hash_mismatch")

    monkeypatch.setattr(module, "verify_manifest", reject)
    with pytest.raises(ValueError, match="manifest_hash_mismatch"):
        backend.load()
    assert not backend.loaded and not backend.tokenizer_available and not backend.revision
    assert backend._session is None and backend._tokenizer is None


def test_wrong_graph_metadata_rejected(fake_load_environment, monkeypatch):
    root, session = fake_load_environment
    monkeypatch.setattr(
        session,
        "get_outputs",
        lambda: [
            SimpleNamespace(
                name="last_hidden_state", type="tensor(float)", shape=["batch", "sequence", 1024]
            )
        ],
    )
    backend = V5MergedOnnxBackend(root, manifest_sha256="a" * 64)
    with pytest.raises(ValueError, match="onnx_graph_contract_mismatch"):
        backend.load()
    assert not backend.loaded


@pytest.mark.parametrize(
    "change",
    [
        {"merged_onnx_root": None},
        {"merged_onnx_root": "relative"},
        {"v5_onnx_manifest_sha256": None},
        {"v5_onnx_manifest_sha256": "bad"},
        {"merged_onnx_optimization": "basic"},
        {"onnx_inter_op_threads": 2},
    ],
)
def test_onnx_configuration_requires_complete_explicit_contract(tmp_path, change):
    values = {
        "api_key": KEY,
        "jina_noncommercial": True,
        "backend": "v5-onnx-merged",
        "merged_onnx_root": tmp_path,
        "v5_onnx_manifest_sha256": "a" * 64,
    }
    with pytest.raises(ValueError):
        Settings(**(values | change))


def test_backend_selection_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ENCODER_API_KEY_FILE", raising=False)
    for name, value in {
        "ENCODER_API_KEY": KEY,
        "JINA_NONCOMMERCIAL": "1",
        "ENCODER_BACKEND": "v5-onnx-merged",
        "ENCODER_MERGED_ONNX_ROOT": str(tmp_path),
        "ENCODER_V5_ONNX_MANIFEST_SHA256": "a" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    assert Settings.from_env().backend == "v5-onnx-merged"


def test_api_load_failure_has_no_torch_fallback(tmp_path, monkeypatch):
    from uranus_research_encoder.model import TorchBackend

    def reject(self):
        raise ValueError("manifest_hash_mismatch")

    def forbidden(self):
        pytest.fail("automatic Torch fallback")

    monkeypatch.setattr(V5MergedOnnxBackend, "load", reject)
    monkeypatch.setattr(TorchBackend, "load", forbidden)
    settings = Settings(
        api_key=KEY,
        jina_noncommercial=True,
        backend="v5-onnx-merged",
        merged_onnx_root=tmp_path,
        v5_onnx_manifest_sha256="a" * 64,
    )
    with TestClient(create_app(settings)) as client:
        auth = {"Authorization": "Bearer " + KEY}
        assert client.get("/health").status_code == 200
        assert client.get("/ready", headers=auth).status_code == 503
        assert (
            client.post(
                "/embed",
                headers=auth,
                json={"model": "jina-v5", "kind": "query", "texts": ["test"]},
            ).status_code
            == 503
        )
        version = client.get("/version", headers=auth).json()
        assert version["backend"] == "v5-onnx-merged"
        assert version["runtime"].startswith("onnxruntime-")
