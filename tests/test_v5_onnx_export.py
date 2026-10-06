"""Tests of actual LoRA arithmetic, native causal masking and ONNX dynamic shapes."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def v5_tiny():
    import torch
    from peft import LoraConfig
    from transformers import Qwen3Config, Qwen3Model

    from uranus_research_encoder.v5_onnx_export import TARGETS

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = Qwen3Model(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            attn_implementation="eager",
        )
    )
    model.add_adapter(
        LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=sorted(TARGETS),
            task_type="FEATURE_EXTRACTION",
        ),
        adapter_name="retrieval",
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(std=0.01)
    model.eval()
    return model


def test_preregistered_pins_and_source_unchanged():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    frozen = root / "validation/jina-v5-merged-v1"
    plan = json.loads((frozen / "plan.json").read_text())
    prov = json.loads((frozen / "native-provenance.json").read_text())
    for n, k in [
        ("selection.json", "selection_sha256"),
        ("native-provenance.json", "native_provenance_sha256"),
    ]:
        assert hashlib.sha256((frozen / n).read_bytes()).hexdigest() == plan[k]
    assert (
        hashlib.sha256((root / "src/uranus_research_encoder/model.py").read_bytes()).hexdigest()
        == prov["model_py_sha256"]
    )
    assert prov["model_revision"] == "dd76d535f5447ca3897a9c893fb1e612ead98192"
    assert prov["adapter"] == "adapters/retrieval"


def test_exact_merge_and_no_active_adapter(v5_tiny):
    import torch
    from peft.tuners.lora.layer import LoraLayer

    from uranus_research_encoder.v5_onnx_export import merge_retrieval

    base = {
        n: m.base_layer.weight.detach().clone()
        for n, m in v5_tiny.named_modules()
        if isinstance(m, LoraLayer)
    }
    expected = {
        n: base[n] + m.lora_B["retrieval"].weight @ m.lora_A["retrieval"].weight
        for n, m in v5_tiny.named_modules()
        if isinstance(m, LoraLayer)
    }
    audit = merge_retrieval(v5_tiny)
    assert audit["exact_matches"] == 196 and audit["adapter_tensor_count"] == 392
    assert not any(isinstance(m, LoraLayer) for m in v5_tiny.modules())
    assert not hasattr(v5_tiny, "peft_config")
    for n, w in expected.items():
        assert torch.equal(v5_tiny.get_submodule(n).weight, w)


@pytest.mark.parametrize("failure", ["missing", "extra", "scale", "adapter"])
def test_invalid_targets_fail_before_merge(v5_tiny, failure):
    from uranus_research_encoder.v5_onnx_export import merge_retrieval

    m = v5_tiny.layers[0].self_attn.q_proj
    if failure == "missing":
        v5_tiny.layers[0].self_attn.q_proj = m.base_layer
    elif failure == "extra":
        v5_tiny.extra = m
    elif failure == "scale":
        m.scaling["retrieval"] = 2
    else:
        v5_tiny.peft_config["other"] = v5_tiny.peft_config["retrieval"]
    with pytest.raises(ValueError):
        merge_retrieval(v5_tiny)


def test_pooling_left_right_and_normalization():
    import torch

    from uranus_research_encoder.v5_onnx_export import last_token_normalized

    hidden = torch.arange(1, 49, dtype=torch.float32).reshape(3, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    expected = torch.nn.functional.normalize(
        hidden[torch.arange(3), torch.tensor([1, 3, 3])], dim=1
    )
    assert torch.equal(last_token_normalized(hidden, mask), expected)


def test_native_wrapper_and_dynamic_onnx(v5_tiny, tmp_path):
    import onnxruntime as ort
    import torch

    from uranus_research_encoder.v5_onnx_export import (
        EmbeddingGraph,
        last_token_normalized,
        merge_retrieval,
    )

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with torch.inference_mode():
        original = last_token_normalized(
            v5_tiny(ids, attention_mask=mask, use_cache=False).last_hidden_state, mask
        )
        merge_retrieval(v5_tiny)
        wrapper = EmbeddingGraph(v5_tiny).eval()
        actual = wrapper(ids, mask)
        torch.testing.assert_close(actual, original, atol=1e-5, rtol=1e-5)
    # Legacy exporter here exercises a different path from the real dynamo export.
    path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        wrapper,
        (ids, mask),
        str(path),
        dynamo=False,
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "text_embeds": {0: "batch"},
        },
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for batch_ids, batch_mask in [
        (ids, mask),
        (torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.long)),
        (
            torch.tensor([[0, 0, 1], [2, 3, 4], [0, 5, 6]]),
            torch.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]),
        ),
    ]:
        with torch.inference_mode():
            expected = wrapper(batch_ids, batch_mask).numpy()
        output = session.run(
            None, {"input_ids": batch_ids.numpy(), "attention_mask": batch_mask.numpy()}
        )[0]
        np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)
