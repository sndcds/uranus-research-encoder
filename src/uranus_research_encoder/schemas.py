"""Reviewable JSON Schema artifacts; regenerate with python -m ...schemas."""

import argparse
import json
from pathlib import Path

from .contracts import (
    Chunk,
    ChunkRequest,
    ChunkResponse,
    EmbedRequest,
    EmbedResponse,
    EvidenceContext,
    Section,
)

MODELS = (EmbedRequest, EmbedResponse, ChunkRequest, ChunkResponse, EvidenceContext, Section, Chunk)


def schemas() -> dict[str, str]:
    return {
        model.__name__: json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        for model in MODELS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in schemas().items():
        (args.output / f"{name}.json").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
