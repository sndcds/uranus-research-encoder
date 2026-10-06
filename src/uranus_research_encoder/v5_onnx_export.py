"""FP32 v5 export only. Inference runtime must never import this module."""

import torch
from peft.tuners.lora.layer import Linear, LoraLayer

from .merged_export import merge_verified

ATTENTION_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def expected_targets():
    return {
        f"layers.{i}.{'self_attn' if target in ATTENTION_TARGETS else 'mlp'}.{target}"
        for i in range(28)
        for target in TARGETS
    }


def merge_retrieval(model):
    expected = expected_targets()
    layers = {
        name: layer
        for name, layer in model.named_modules(remove_duplicate=False)
        if isinstance(layer, LoraLayer)
    }
    if set(layers) != expected:
        raise ValueError(
            f"target_mismatch missing={sorted(expected - layers.keys())} "
            f"unexpected={sorted(layers.keys() - expected)}"
        )
    if set(model.peft_config) != {"retrieval"}:
        raise ValueError("retrieval_only_required")
    for layer in layers.values():
        if (
            type(layer) is not Linear
            or layer.active_adapters != ["retrieval"]
            or set(layer.lora_A) != {"retrieval"}
            or set(layer.lora_B) != {"retrieval"}
            or layer.r["retrieval"] != 32
            or layer.scaling["retrieval"] != 1
            or layer.fan_in_fan_out
            or layer.training
            or any(p.dtype != torch.float32 for p in layer.parameters())
        ):
            raise ValueError("unsupported_v5_lora")
    rows = merge_verified(model, "retrieval")
    # The Transformers adapter mixin remains a class method, but forward no longer
    # depends on PEFT state or an adapter selection. Remove obsolete metadata.
    model._hf_peft_config_loaded = False
    del model.peft_config
    if any(isinstance(m, LoraLayer) for m in model.modules()):
        raise ValueError("active_lora_after_merge")
    if any("lora_" in n for n, _ in model.named_parameters()):
        raise ValueError("adapter_parameter_after_merge")
    return {
        "expected_target_count": 196,
        "observed_target_count": len(rows),
        "adapter_tensor_count": 392,
        "exact_matches": len(rows),
        "within_tolerance_matches": len(rows),
        "max_difference": 0,
        "missing_targets": [],
        "unexpected_targets": [],
        "active_lora_layers": 0,
        "inference_adapter_selection_required": False,
        "targets": rows,
        "pass": True,
    }


def last_token_normalized(hidden, mask):
    positions = torch.arange(mask.shape[1], device=mask.device)
    last = torch.where(mask.bool(), positions, -1).max(dim=1).values
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


class EmbeddingGraph(torch.nn.Module):
    """Native causal eager attention, last valid position, one FP32 L2 operation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        allowed = (positions[:, None] >= positions[None, :])[None, None, :, :] & attention_mask[
            :, None, None, :
        ].bool()
        mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        hidden = self.model(
            input_ids=input_ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        return last_token_normalized(hidden, attention_mask)
