"""Exercise the native loader and adapters with small locally generated weights."""

import math
import socket

import pytest

from uranus_research_encoder.model import JinaBackend
from uranus_research_encoder.version import MODEL_REVISION


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import torch
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import JinaEmbeddingsV3Config, JinaEmbeddingsV3Model, PreTrainedTokenizerFast

    torch.manual_seed(123)
    torch.set_num_threads(1)
    snapshot = tmp_path / "models--jinaai--jina-embeddings-v3-hf" / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
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
    config = JinaEmbeddingsV3Config(
        vocab_size=6,
        hidden_size=1024,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=16,
    )
    model = JinaEmbeddingsV3Model(config)
    model.save_pretrained(snapshot)
    for task in ("retrieval_query", "retrieval_passage"):
        model.add_adapter(
            LoraConfig(
                r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False
            ),
            adapter_name=task,
        )
        model.set_adapter(task)
        model.save_pretrained(snapshot / task)
    return tmp_path


def test_native_backend_offline_and_task_isolation(tiny_cache, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("runtime attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    backend = JinaBackend(tiny_cache)
    backend.load()
    assert backend.loaded and backend.tokenizer_available
    assert backend.revision == MODEL_REVISION
    assert backend.count("hello world") == 4
    # Every loaded adapter tensor must match its file even after loading the other task.
    import torch
    from safetensors.torch import load_file

    snapshot = tiny_cache / "models--jinaai--jina-embeddings-v3-hf" / "snapshots" / MODEL_REVISION
    parameters = dict(backend._model.named_parameters())
    for task in ("retrieval_query", "retrieval_passage"):
        for key, tensor in load_file(snapshot / task / "adapter_model.safetensors").items():
            name = key.removeprefix("base_model.model.")
            name = name.replace(".lora_A.weight", f".lora_A.{task}.weight")
            name = name.replace(".lora_B.weight", f".lora_B.{task}.weight")
            assert torch.equal(parameters[name], tensor)
    query = backend.embed(["hello world"], "query")
    passage = backend.embed(["hello world"], "passage")
    assert query != passage
    assert query == backend.embed(["hello world"], "query")
    assert passage == backend.embed(["hello world", "world"], "passage")[:1]
    for vector in query + passage:
        assert len(vector) == 1024
        assert all(math.isfinite(x) for x in vector)
        assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1, abs=1e-6)
    assert str(next(backend._model.parameters()).dtype) == "torch.float32"
    assert not backend._model.training
    with pytest.raises(ValueError, match="token_limit"):
        backend.embed(["hello " * 8192], "query")


def test_missing_cache_fails_offline(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("runtime attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    from huggingface_hub.errors import LocalEntryNotFoundError

    with pytest.raises(LocalEntryNotFoundError):
        JinaBackend(tmp_path).load()


def test_incomplete_weights_fail_readiness(tiny_cache):
    from safetensors.torch import load_file, save_file

    weights = (
        tiny_cache
        / "models--jinaai--jina-embeddings-v3-hf"
        / "snapshots"
        / MODEL_REVISION
        / "model.safetensors"
    )
    tensors = load_file(weights)
    tensors.pop(next(iter(tensors)))
    save_file(tensors, weights)
    backend = JinaBackend(tiny_cache)
    with pytest.raises(ValueError, match="incomplete_weights"):
        backend.load()
    assert not backend.loaded
