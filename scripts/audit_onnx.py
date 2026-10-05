#!/usr/bin/env python3
"""No Jina v5 ONNX graph has been approved for an integrity/parity audit."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root")
    parser.add_argument("--output")
    parser.parse_args()
    parser.error("onnx_not_supported_for_jina_v5; historical v3 audits remain in validation/")


if __name__ == "__main__":
    main()
