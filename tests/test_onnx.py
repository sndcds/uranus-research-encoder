"""Real CPU ORT sessions over tiny local task-conditioned external-data graphs."""

import hashlib
import json
import socket

import numpy as np
import pytest

from uranus_research_encoder import onnx_backend
from uranus_research_encoder.onnx_backend import OnnxBackend
from uranus_research_encoder.version import MODEL_REVISION


@pytest.fixture
def onnx_cache(tmp_path, monkeypatch):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    def no_network(*args, **kwargs):
        pytest.fail("runtime attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    repo = tmp_path / "models--jinaai--jina-embeddings-v3-hf"
    snapshot = repo / "snapshots" / MODEL_REVISION
    (snapshot / "onnx").mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "hello": 4, "world": 5}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", model_max_length=8192
    ).save_pretrained(snapshot)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "jina_embeddings_v3", "hidden_size": 1024})
    )
    weights = np.zeros((2, 6, 1024), dtype=np.float32)
    weights[0, 4, 0], weights[0, 5, 1] = 3, 4
    weights[1, 4, 0], weights[1, 5, 1] = -4, 3
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["banks", "task_id"], ["bank"], axis=0),
            helper.make_node("Gather", ["bank", "input_ids"], ["text_embeds"], axis=0),
        ],
        "tiny-task-encoder",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            ),
            helper.make_tensor_value_info("task_id", TensorProto.INT64, []),
        ],
        [
            helper.make_tensor_value_info(
                "text_embeds", TensorProto.FLOAT, ["batch", "sequence", 1024]
            )
        ],
        [numpy_helper.from_array(weights, name="banks")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=10)
    onnx.save_model(
        model,
        snapshot / "onnx/model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=0,
    )
    hashes = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in ("onnx/model.onnx", "onnx/model.onnx_data")
    }
    # Only generated fixtures replace the reviewed production digests.
    monkeypatch.setattr(onnx_backend, "ONNX_SHA256", hashes)
    files = {
        path.relative_to(snapshot).as_posix(): {"size": path.stat().st_size, "blob_id": "0" * 40}
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    for name in (
        "custom_st.py",
        "model.safetensors",
        "classification/adapter_model.safetensors",
        "retrieval_query/adapter_model.safetensors",
        "onnx/model_fp16.onnx",
    ):
        files[name] = {"size": 1, "blob_id": "0" * 40}
    (repo / "trees").mkdir()
    (repo / "trees" / f"{MODEL_REVISION}.json").write_text(
        json.dumps({"format_version": 1, "files": files})
    )
    return tmp_path, snapshot


def test_onnx_offline_tasks_pooling_and_session_options(onnx_cache):
    import onnxruntime as ort

    root, _ = onnx_cache
    backend = OnnxBackend(root, intra_op_threads=2)
    backend.load()
    assert backend.loaded and backend.tokenizer_available
    assert backend.backend == "onnx"
    assert backend.revision == MODEL_REVISION
    assert backend.count("hello world") == 4
    assert backend._session.get_providers() == ["CPUExecutionProvider"]
    options = backend._session.get_session_options()
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    query = backend.embed(["hello world"], "query")
    passage = backend.embed(["hello world"], "passage")
    assert query[0][:2] == pytest.approx([0.6, 0.8])
    assert passage[0][:2] == pytest.approx([-0.8, 0.6])
    assert query == backend.embed(["hello world", "world"], "query")[:1]
    assert passage == backend.embed(["hello world"], "passage")
    for vector in query + passage:
        assert len(vector) == 1024
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError, match="token_limit"):
        backend.embed(["hello " * 8192], "query")


@pytest.mark.parametrize("name", ["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"])
def test_onnx_missing_required_file_fails_offline(onnx_cache, name):
    from huggingface_hub.errors import IncompleteSnapshotError

    root, snapshot = onnx_cache
    (snapshot / name).unlink()
    backend = OnnxBackend(root)
    with pytest.raises(IncompleteSnapshotError):
        backend.load()
    assert not backend.loaded


@pytest.mark.parametrize("name", ["onnx/model.onnx", "onnx/model.onnx_data"])
def test_onnx_changed_graph_or_adapter_weights_fail_integrity(onnx_cache, name):
    root, snapshot = onnx_cache
    with (snapshot / name).open("ab") as stream:
        stream.write(b"changed")
    backend = OnnxBackend(root)
    with pytest.raises(ValueError, match="onnx_artifact_mismatch"):
        backend.load()
    assert not backend.loaded


def test_onnx_rejects_zero_vectors(onnx_cache):
    root, _ = onnx_cache
    backend = OnnxBackend(root)
    backend.load()
    with pytest.raises(RuntimeError, match="invalid_vector"):
        backend.embed(["unknown"], "query")


def test_onnx_pooling_excludes_masked_tokens(onnx_cache, monkeypatch):
    root, _ = onnx_cache
    backend = OnnxBackend(root)
    backend.load()
    tokenizer = backend._tokenizer

    def masked(*args, **kwargs):
        encoded = tokenizer(*args, **kwargs)
        encoded["attention_mask"][0, 2] = 0  # Exclude "world", leaving the first axis only.
        return encoded

    monkeypatch.setattr(backend, "_tokenizer", masked)
    assert backend.embed(["hello world"], "query")[0][:2] == [1.0, 0.0]


def test_http_selects_onnx_and_reports_runtime(onnx_cache, settings, auth):
    from fastapi.testclient import TestClient

    from uranus_research_encoder.app import create_app

    root, _ = onnx_cache
    configured = settings.model_copy(update={"backend": "onnx", "model_root": root})
    with TestClient(create_app(configured)) as client:
        ready = client.get("/ready", headers=auth)
        assert ready.status_code == 200
        version = client.get("/version", headers=auth).json()
        for name in (
            "backend",
            "runtime",
            "model_revision",
            "dimensions",
            "embedding_version",
            "contract_version",
        ):
            assert ready.json()[name] == version[name]
        assert version["backend"] == "onnx"
        assert version["runtime"].startswith("onnxruntime-")
        response = client.post(
            "/embed",
            headers=auth,
            json={"model": "jina-v3", "kind": "query", "texts": ["hello world"]},
        )
        assert response.status_code == 200
        assert response.json()["vectors"][0][:2] == pytest.approx([0.6, 0.8])
