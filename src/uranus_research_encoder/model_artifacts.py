"""Reviewed pinned Jina artifact manifests shared by provisioning and loading."""

from typing import Literal

BackendName = Literal["torch", "onnx"]

# Shared AutoTokenizer assets. Neither manifest admits repository Python code.
TOKENIZER_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
]

# Native single-file/sharded weights and exactly the two retrieval adapters.
TORCH_FILES = TOKENIZER_FILES + [
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "retrieval_query/adapter_config.json",
    "retrieval_query/adapter_model.safetensors",
    "retrieval_passage/adapter_config.json",
    "retrieval_passage/adapter_model.safetensors",
]

# All five task banks are embedded in this upstream graph's external data.
# No separate classification/separation/text_matching adapter files are needed.
ONNX_SHA256 = {
    "onnx/model.onnx": "836322cc5d78f0047dbb336d0049b087fd81e164505c347a313cd44a772f34cc",
    "onnx/model.onnx_data": "022b369faa8015add1d036a72927ed64f8c8196435bc37a4fc9d36fc8ebc5cc2",
}
ONNX_FILES = TOKENIZER_FILES + list(ONNX_SHA256)
ARTIFACT_MANIFESTS = {"torch": TORCH_FILES, "onnx": ONNX_FILES}

# Compatibility alias for existing provisioning/cache regression tests.
MODEL_ALLOW_PATTERNS = TORCH_FILES


def artifact_patterns(backend: Literal["torch", "onnx", "all"]) -> list[str]:
    if backend == "all":
        return list(dict.fromkeys(TORCH_FILES + ONNX_FILES))
    return ARTIFACT_MANIFESTS[backend]
