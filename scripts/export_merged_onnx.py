#!/usr/bin/env python3
"""The historical ONNX exporter is not compatible with v5 Qwen3 retrieval."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.parse_args()
    parser.error("onnx_not_supported_for_jina_v5; no files were read or written")


if __name__ == "__main__":
    main()
