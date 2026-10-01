"""Generated local graphs and tiny native LoRA models; no real weight downloads."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from test_onnx import onnx_cache  # noqa: F401

from uranus_research_encoder.merged_artifacts import (
    EXPORT_CONTRACT,
    EXPORTER_VERSION,
    OPSET,
    TASKS,
    digest,
    verify_manifest,
)
from uranus_research_encoder.merged_onnx_backend import MergedOnnxBackend
from uranus_research_encoder.version import MODEL_REPOSITORY, MODEL_REVISION


@pytest.fixture
def merged_cache(onnx_cache, tmp_path):  # noqa: F811
    import shutil

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    _, snapshot = onnx_cache
    root = tmp_path / "merged"
    root.mkdir()
    shutil.copytree(snapshot, root / "tokenizer", ignore=shutil.ignore_patterns("onnx"))
    for index, kind in enumerate(TASKS):
        folder = root / f"retrieval-{kind}"
        folder.mkdir()
        weights = np.zeros((6, 1024), dtype=np.float32)
        weights[1, 2] = 10  # Nonzero PAD state must be excluded by the pooling mask.
        weights[4, :2] = [3, 0] if index == 0 else [-4, 0]
        weights[5, :2] = [0, 4] if index == 0 else [0, 3]
        graph = helper.make_graph(
            [helper.make_node("Gather", ["weight", "input_ids"], ["text_embeds"], axis=0)],
            "tiny-merged",
            [
                helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "sequence"])
                for name in ("input_ids", "attention_mask")
            ],
            [
                helper.make_tensor_value_info(
                    "text_embeds", TensorProto.FLOAT, ["batch", "sequence", 1024]
                )
            ],
            [numpy_helper.from_array(weights, name="weight")],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=10
        )
        onnx.save_model(
            model,
            folder / "model.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=0,
        )
    manifest = {
        "contract": EXPORT_CONTRACT,
        "exporter_version": EXPORTER_VERSION,
        "opset": OPSET,
        "dtype": "float32",
        "source": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "tasks": {
            kind: {"task": kind, "source_adapter": adapter, "graph": f"retrieval-{kind}/model.onnx"}
            for kind, adapter in TASKS.items()
        },
        "files": {
            p.relative_to(root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_tasks_pooling_and_request_invariance(merged_cache, monkeypatch):
    backend = MergedOnnxBackend(merged_cache, intra_op_threads=2)
    backend.load()
    assert backend.count("hello world") == 4
    query = backend.embed(["hello world"], "query")
    passage = backend.embed(["hello world"], "passage")
    assert query[0][:2] == pytest.approx([0.6, 0.8])
    assert passage[0][:2] == pytest.approx([-0.8, 0.6])
    assert query == backend.embed(["hello world", "world"], "query")[:1]
    assert query == backend.embed(["hello world"], "query")
    assert np.linalg.norm(query[0]) == pytest.approx(1, abs=1e-6)
    for session in backend._sessions.values():
        assert {item.name for item in session.get_inputs()} == {"input_ids", "attention_mask"}
        assert session.get_session_options().intra_op_num_threads == 2
    tokenizer = backend._tokenizer

    def masked(*args, **kwargs):
        result = tokenizer(*args, **kwargs)
        result["attention_mask"][0, 2] = 0
        return result

    monkeypatch.setattr(backend, "_tokenizer", masked)
    assert backend.embed(["hello world"], "query")[0][:2] == [1, 0]


@pytest.mark.parametrize(
    "file",
    [
        "retrieval-query/model.onnx",
        "retrieval-passage/model.onnx",
        "retrieval-query/model.onnx.data",
        "tokenizer/tokenizer.json",
    ],
)
@pytest.mark.parametrize("tamper", [True, False])
def test_integrity_fails_closed(merged_cache, file, tamper):
    target = merged_cache / file
    if tamper:
        with target.open("ab") as stream:
            stream.write(b"tampered")
    else:
        target.unlink()
    backend = MergedOnnxBackend(merged_cache)
    with pytest.raises((ValueError, FileNotFoundError)):
        backend.load()
    assert not backend.loaded


def test_manifest_identity_and_path_safety(merged_cache):
    path = merged_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["revision"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest_mismatch"):
        verify_manifest(merged_cache)
    manifest["source"]["revision"] = MODEL_REVISION
    manifest["files"]["../escape"] = {"sha256": "bad", "bytes": 0}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe_artifact_path"):
        verify_manifest(merged_cache)


@pytest.mark.parametrize("task", list(TASKS.values()))
def test_tiny_native_merge_export_and_dynamic_shapes(tmp_path, task):
    import onnxruntime as ort
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import LoraLayer
    from transformers import JinaEmbeddingsV3Config, JinaEmbeddingsV3Model

    from uranus_research_encoder.merged_export import TokenEmbeddings, merge_verified

    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = JinaEmbeddingsV3Model(
        JinaEmbeddingsV3Config(
            vocab_size=8,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8192,
        )
    ).eval()
    model.add_adapter(
        LoraConfig(
            r=2,
            lora_alpha=1,
            target_modules=[
                "word_embeddings",
                "token_type_embeddings",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "fc1",
                "fc2",
                "dense",
            ],
        ),
        adapter_name=task,
    )
    torch.manual_seed(23 if task.endswith("query") else 29)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.normal_(0, 0.01)
    model.eval()
    ids = torch.tensor([[0, 4, 5, 2]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        reference = model(input_ids=ids, attention_mask=mask).last_hidden_state.clone()
    expected = {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLayer):
            expected[name] = (layer.get_base_layer().weight + layer.get_delta_weight(task)).detach()
    audit = merge_verified(model, task)
    assert len(audit) == len(expected) == 15
    for name, weight in expected.items():
        assert torch.equal(model.get_submodule(name).weight, weight)
    with torch.no_grad():
        assert torch.allclose(TokenEmbeddings(model)(ids, mask), reference, atol=1e-6, rtol=0)
    export = runpy.run_path(str(Path(__file__).parents[1] / "scripts/export_merged_onnx.py"))
    export["export_graph"](model, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    for batch, length in [(1, 3), (2, 7), (4, 9)]:
        ids = torch.full((batch, length), 4, dtype=torch.int64)
        mask = torch.ones_like(ids)
        mask[:, -1] = 0
        with torch.no_grad():
            reference = TokenEmbeddings(model)(ids, mask).numpy()
        (actual,) = session.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
        np.testing.assert_allclose(reference, actual, rtol=0, atol=1e-6)


def test_http_merged_selection(merged_cache, settings, auth):
    from fastapi.testclient import TestClient

    from uranus_research_encoder.app import create_app

    configured = settings.model_copy(
        update={"backend": "onnx-merged", "merged_onnx_root": merged_cache}
    )
    with TestClient(create_app(configured)) as client:
        assert client.get("/ready", headers=auth).status_code == 200
        assert client.get("/version", headers=auth).json()["backend"] == "onnx-merged"
        response = client.post(
            "/embed",
            headers=auth,
            json={"model": "jina-v3", "kind": "passage", "texts": ["hello world"]},
        )
        assert response.status_code == 200
        assert response.json()["vectors"][0][:2] == pytest.approx([-0.8, 0.6])


def test_export_requires_acknowledgement_and_separate_output(tmp_path):
    import os
    import subprocess
    import sys

    script = Path(__file__).parents[1] / "scripts/export_merged_onnx.py"
    command = [
        sys.executable,
        str(script),
        "--model-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "bad"),
    ]
    result = subprocess.run(
        command, env=os.environ | {"JINA_NONCOMMERCIAL": "0"}, capture_output=True, text=True
    )
    assert result.returncode != 0 and "acknowledge" in result.stderr
    result = subprocess.run(
        command, env=os.environ | {"JINA_NONCOMMERCIAL": "1"}, capture_output=True, text=True
    )
    assert result.returncode != 0 and "isolated" in result.stderr
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("mutation", ["task_id", "pooled_output", "swapped_adapter"])
def test_rejects_wrong_graph_contract(merged_cache, mutation):
    import onnx
    from onnx import TensorProto, helper

    path = merged_cache / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "swapped_adapter":
        manifest["tasks"]["query"]["source_adapter"] = "retrieval_passage"
    else:
        graph_path = merged_cache / "retrieval-query/model.onnx"
        model = onnx.load(graph_path, load_external_data=False)
        if mutation == "task_id":
            model.graph.input.append(
                helper.make_tensor_value_info("task_id", TensorProto.INT64, [])
            )
        else:
            model.graph.output[0].name = "pooled_output"
            model.graph.node[0].output[0] = "pooled_output"
        onnx.save(model, graph_path)
        manifest["files"]["retrieval-query/model.onnx"] = {
            "bytes": graph_path.stat().st_size,
            "sha256": digest(graph_path),
        }
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        MergedOnnxBackend(merged_cache).load()


@pytest.mark.parametrize("exact_arithmetic", [True, False], ids=["dyadic", "random-float32"])
def test_embedding_merge_across_row_blocks_matches_peft(exact_arithmetic):
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Embedding

    from uranus_research_encoder.merged_export import merge_verified

    torch.manual_seed(42)
    task = "retrieval_query"
    layer = Embedding(torch.nn.Embedding(2051, 8), task, LoraConfig(), r=4, lora_alpha=1)
    with torch.no_grad():
        for tensor in (
            layer.get_base_layer().weight,
            layer.lora_embedding_A[task],
            layer.lora_embedding_B[task],
        ):
            if exact_arithmetic:
                # Small dyadic numbers make every product/sum exactly representable.
                # Bit equality then tests indexing, scaling and the final partial block
                # independently of which GEMM kernel the CPU selects.
                tensor.copy_(torch.randint(-8, 9, tensor.shape).float() / 8)
            else:
                tensor.normal_()
        base = layer.get_base_layer().weight.double().clone()
        a, b = layer.lora_embedding_A[task].double(), layer.lora_embedding_B[task].double()
        scale = layer.scaling[task]
        ideal = base + scale * (b @ a).T
        # Standard forward-error bound for r-term dot products, scaling and addition:
        # gamma_n = n*u/(1-n*u). Absolute products handle cancellation correctly.
        n, u = a.shape[0] + 2, torch.finfo(torch.float32).eps / 2
        bound = (n * u / (1 - n * u)) * (base.abs() + abs(scale) * (b.abs() @ a.abs()).T)
        expected = layer.get_base_layer().weight + layer.get_delta_weight(task)
    model = torch.nn.Sequential(layer)
    rows = merge_verified(model, task)
    assert rows[0]["exact_weight_match"]
    actual = model[0].weight
    if exact_arithmetic:
        assert torch.equal(actual, expected)
        assert torch.equal(actual.double(), ideal)
    else:
        # Full vs blocked GEMM can round differently on AVX2/AVX512 runners.
        # Both must independently satisfy the mathematical float64 reference bound.
        assert torch.all((actual.double() - ideal).abs() <= bound)
        assert torch.all((expected.double() - ideal).abs() <= bound)
        assert torch.all((actual.double() - expected.double()).abs() <= 2 * bound)


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_padded_batching_preserves_vectors_and_limits(merged_cache, kind):
    from uranus_research_encoder.merged_batching import embed_padded

    backend = MergedOnnxBackend(merged_cache)
    backend.load()
    texts = ["hello world", "hello", "world", "hello hello world", "hello"]
    expected = backend.embed(texts, kind)
    actual = embed_padded(backend, texts, kind)
    assert actual == expected
    assert actual == embed_padded(backend, texts, kind)
    assert actual == embed_padded(backend, texts, kind, batch_size=2)
    for index, text in enumerate(texts):
        assert actual[index] == embed_padded(backend, [text], kind)[0]
    with pytest.raises(ValueError, match="invalid_batch_size"):
        embed_padded(backend, texts, kind, batch_size=0)
    backend.max_tokens = 3
    with pytest.raises(ValueError, match="token_limit"):
        embed_padded(backend, texts, kind)
