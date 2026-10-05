"""Retain model-independent merge arithmetic tests; v5 export is disabled."""

import pytest


@pytest.mark.parametrize("exact_arithmetic", [True, False], ids=["dyadic", "random-float32"])
def test_embedding_merge_across_row_blocks_matches_peft(exact_arithmetic):
    import torch
    from peft import LoraConfig
    from peft.tuners.lora.layer import Embedding

    from uranus_research_encoder.merged_export import merge_verified

    torch.manual_seed(42)
    task = "retrieval"
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
