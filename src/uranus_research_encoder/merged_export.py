"""Offline export helpers; never imported by the inference runtime.

Only ordinary float32 Linear/Embedding LoRA is supported. Linear matrices are
independently reconstructed before PEFT merges them, then checked bit for bit.
Embedding tables use the explicit same formula in bounded row blocks.
"""

import hashlib

import torch
from peft.tuners.lora.layer import Embedding, Linear, LoraLayer


def tensor_digest(tensor) -> str:
    return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy())).hexdigest()


def merge_verified(model, adapter: str) -> list[dict]:
    """Audit and merge all targets, then remove all adapter wrappers.

    Transformers' native adapter mixin does not expose merge_and_unload.
    Replacing each verified wrapper by its base layer provides the equivalent
    explicit operation without constructing a second PeftModel wrapper.
    """
    audit = []
    with torch.no_grad():
        for name, layer in list(model.named_modules()):
            if not isinstance(layer, LoraLayer):
                continue
            if type(layer) not in (Linear, Embedding) or layer.merged or layer.lora_variant:
                raise ValueError("unsupported_lora_layer")
            base = layer.get_base_layer()
            if base.weight.dtype != torch.float32 or base.weight.device.type != "cpu":
                raise ValueError("float32_cpu_required")
            if isinstance(layer, Linear):
                if set(layer.lora_A) != {adapter} or layer.lora_bias[adapter]:
                    raise ValueError("unsupported_adapter")
                delta = layer.lora_B[adapter].weight @ layer.lora_A[adapter].weight
                if layer.fan_in_fan_out:
                    delta = delta.T
            else:
                if set(layer.lora_embedding_A) != {adapter}:
                    raise ValueError("unsupported_adapter")
                # The vocabulary table alone is ~1 GiB. PEFT safe_merge creates
                # full-size copies; audit and merge embedding rows in blocks.
                base_digest = tensor_digest(base.weight)
                delta_hash, expected_hash = hashlib.sha256(), hashlib.sha256()
                for start in range(0, base.weight.shape[0], 1024):
                    weight = base.weight[start : start + 1024]
                    delta = (
                        layer.lora_embedding_B[adapter]
                        @ layer.lora_embedding_A[adapter][:, start : start + 1024]
                    ).T
                    delta = (delta * layer.scaling[adapter]).contiguous()
                    expected = weight + delta
                    if not torch.isfinite(expected).all():
                        raise ValueError("nonfinite_merged_weight")
                    delta_hash.update(memoryview(delta.numpy()))
                    expected_hash.update(memoryview(expected.numpy()))
                    weight.copy_(expected)
                merged_digest = tensor_digest(base.weight)
                if merged_digest != expected_hash.hexdigest():
                    raise ValueError("merged_weight_mismatch")
                audit.append(
                    {
                        "name": name,
                        "shape": list(base.weight.shape),
                        "base_sha256": base_digest,
                        "delta_sha256": delta_hash.hexdigest(),
                        "merged_sha256": merged_digest,
                        "exact_weight_match": True,
                        "formula": "base + scaling * (B @ A).T; blocks of 1024 rows",
                        "method": "explicit_embedding_merge",
                    }
                )
                model.set_submodule(name, base)
                continue
            delta = delta * layer.scaling[adapter]
            expected = base.weight.detach() + delta
            if not torch.isfinite(expected).all():
                raise ValueError("nonfinite_merged_weight")
            expected_digest = tensor_digest(expected)
            row = {
                "name": name,
                "shape": list(base.weight.shape),
                "base_sha256": tensor_digest(base.weight),
                "delta_sha256": tensor_digest(delta),
                "merged_sha256": expected_digest,
                "formula": "base + scaling * transpose_if_required(B @ A)",
                "method": "peft_safe_merge",
            }
            del delta, expected
            layer.merge(safe_merge=True, adapter_names=[adapter])
            if tensor_digest(base.weight) != expected_digest:
                raise ValueError("merged_weight_mismatch")
            row["exact_weight_match"] = True
            audit.append(row)
            model.set_submodule(name, base)
    if not audit or any(isinstance(layer, LoraLayer) for layer in model.modules()):
        raise ValueError("incomplete_merge")
    model.requires_grad_(False)
    model.eval()
    return audit


class TokenEmbeddings(torch.nn.Module):
    """Expose token states only; preserve the native forward and eager attention."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Native create_bidirectional_mask accepts an already additive 4-D mask.
        # This avoids tracing the library's Python/vmap mask factory. The exact
        # native-vs-wrapper equivalence is checked before every real export.
        mask = 1 - attention_mask[:, None, None, :].to(torch.float32)
        mask = mask * torch.finfo(torch.float32).min
        return self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
