"""Reviewed native Jina runtime files shared by provisioning and offline loading."""

# Patterns cover single-file or sharded native weights and AutoTokenizer assets.
# Only the two retrieval adapters are used. Never include remote Python code,
# ONNX, other weight formats, unrelated adapters, or repository documentation.
MODEL_ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "retrieval_query/adapter_config.json",
    "retrieval_query/adapter_model.safetensors",
    "retrieval_passage/adapter_config.json",
    "retrieval_passage/adapter_model.safetensors",
]
