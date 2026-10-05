"""Pinned native v5 artifacts, shared by offline loading and explicit prefetch."""

from typing import Literal

BackendName = Literal["torch", "onnx"]
TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
TORCH_FILES = TOKENIZER_FILES + [
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "adapters/retrieval/adapter_config.json",
    "adapters/retrieval/adapter_model.safetensors",
]
MODEL_ALLOW_PATTERNS = TORCH_FILES


def artifact_patterns(backend: str) -> list[str]:
    if backend != "torch":
        raise ValueError("onnx_not_supported_for_jina_v5")
    return list(TORCH_FILES)
