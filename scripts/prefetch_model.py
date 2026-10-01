#!/usr/bin/env python3
"""Explicit, network-enabled provisioning step; never called by the service."""

import argparse
import os
from pathlib import Path

from uranus_research_encoder.version import MODEL_REPOSITORY, MODEL_REVISION

# Native safetensors and the two retrieval adapters only. No repository Python code.
PATTERNS = [
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
    "README.md",
    "LICENSE*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("JINA_NONCOMMERCIAL") != "1":
        parser.error("set JINA_NONCOMMERCIAL=1 to acknowledge the model license")
    if not args.model_root.is_absolute():
        parser.error("--model-root must be absolute")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        cache_dir=str(args.model_root),
        allow_patterns=PATTERNS,
    )
    if Path(snapshot).name != MODEL_REVISION:
        raise RuntimeError("unexpected model revision")
    print(f"Prefetched {MODEL_REPOSITORY} at {MODEL_REVISION}")


if __name__ == "__main__":
    main()
