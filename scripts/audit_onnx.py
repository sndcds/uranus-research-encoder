#!/usr/bin/env python3
"""Read-only offline inspection of the pinned graph and native adapter identity."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from safetensors import safe_open

from uranus_research_encoder.model import cached_snapshot, offline_environment
from uranus_research_encoder.model_artifacts import ONNX_SHA256
from uranus_research_encoder.version import MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    offline_environment()
    snapshot = cached_snapshot(args.model_root, "onnx")
    graph = onnx.load(snapshot / "onnx/model.onnx", load_external_data=False)
    hashes = {}
    for name, expected in ONNX_SHA256.items():
        with (snapshot / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("artifact_digest_mismatch")
        hashes[name] = actual
    tensors = {}
    for tensor in graph.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            external = {entry.key: entry.value for entry in tensor.external_data}
            if external["location"] != "model.onnx_data":
                raise ValueError("unreviewed_external_path")
            tensors[tensor.name] = np.memmap(
                snapshot / "onnx/model.onnx_data",
                mode="r",
                dtype=onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type),
                offset=int(external.get("offset", 0)),
                shape=tuple(tensor.dims),
            )
        else:
            tensors[tensor.name] = onnx.numpy_helper.to_array(tensor)
    adapters = {}
    for bank, task in enumerate(("retrieval_query", "retrieval_passage")):
        differences = []
        with safe_open(snapshot / task / "adapter_model.safetensors", framework="pt") as native:
            for name in native.keys():
                left = native.get_tensor(name).float().numpy()
                if "lora_embedding_" in name:
                    key = name.replace("lora_embedding_", "parametrizations.weight.0.lora_")
                    right = tensors[key][bank].T
                elif "self_attn" in name:
                    projection = name.split("self_attn.")[1][0]
                    part = {"q": 0, "k": 1, "v": 2}[projection]
                    key = name.replace(f"self_attn.{projection}_proj", "mixer.Wqkv")
                    key = key.replace(".lora_", ".parametrizations.weight.0.lora_").removesuffix(
                        ".weight"
                    )
                    right = tensors[key][bank]
                    if ".lora_B." in name:
                        right = right[part * 1024 : (part + 1) * 1024]
                else:
                    key = name.replace(".lora_", ".parametrizations.weight.0.lora_").removesuffix(
                        ".weight"
                    )
                    right = tensors[key][bank]
                differences.append(float(np.max(np.abs(left - right))))
        adapters[task] = {
            "bank": bank,
            "tensors": len(differences),
            "max_absolute_difference": max(differences),
            "exact_tensors": sum(value == 0 for value in differences),
        }
    differences = []
    with safe_open(snapshot / "model.safetensors", framework="pt") as native:
        for name in native.keys():
            key = name.removesuffix(".weight") + ".parametrizations.weight.original"
            right = tensors[key] if key in tensors else tensors[name]
            # Stream large embedding matrices instead of materializing two copies.
            sliced = native.get_slice(name)
            maximum = 0.0
            for start in range(0, right.shape[0], 1024):
                left = sliced[start : start + 1024].float().numpy()
                maximum = max(maximum, float(np.max(np.abs(left - right[start : start + 1024]))))
            differences.append(maximum)
    report = {
        "model_revision": MODEL_REVISION,
        "onnx_parser_version": onnx.__version__,
        "sha256": hashes,
        "producer": [graph.producer_name, graph.producer_version],
        "ir_version": graph.ir_version,
        "opsets": {item.domain: item.version for item in graph.opset_import},
        "inputs": {item.name: onnx.helper.printable_type(item.type) for item in graph.graph.input},
        "outputs": {
            item.name: onnx.helper.printable_type(item.type) for item in graph.graph.output
        },
        "nodes": len(graph.graph.node),
        "initializers": len(graph.graph.initializer),
        "operator_counts": dict(Counter(node.op_type for node in graph.graph.node)),
        "task_gathers": sum(
            node.op_type == "Gather" and "task_id" in node.input for node in graph.graph.node
        ),
        "external_initializers": sum(
            tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer
        ),
        "adapters": adapters,
        "base_weights": {
            "tensors": len(differences),
            "max_absolute_difference": max(differences),
            "exact_tensors": sum(value == 0 for value in differences),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"adapters": adapters, "base_weights": report["base_weights"]}, indent=2))


if __name__ == "__main__":
    main()
