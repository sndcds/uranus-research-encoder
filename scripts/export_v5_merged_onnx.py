#!/usr/bin/env python3
"""Offline pinned v5 FP32 export; refuses overwrite and any failed merge check."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--plan-dir", type=Path, default=Path("validation/jina-v5-merged-v1"))
    p.add_argument(
        "--source-commit", help="Commit of the read-only source bundle on an isolated host"
    )
    a = p.parse_args()
    from uranus_research_encoder.model import TorchBackend, cached_snapshot, offline_environment

    offline_environment()
    import onnx
    import torch
    from safetensors import safe_open

    from uranus_research_encoder.v5_onnx_export import EmbeddingGraph, merge_retrieval

    plan = json.loads((a.plan_dir / "plan.json").read_text())
    provenance = json.loads((a.plan_dir / "native-provenance.json").read_text())
    for name, field in [
        ("selection.json", "selection_sha256"),
        ("native-provenance.json", "native_provenance_sha256"),
    ]:
        if sha(a.plan_dir / name) != plan[field]:
            raise ValueError("preregistration_changed")
    for name, v in provenance["versions"].items():
        if importlib.metadata.version(name) != v:
            raise ValueError("export_version_mismatch")
    snapshot = cached_snapshot(a.model_root, "torch")
    for artifact in provenance["artifacts"]:
        if sha(snapshot / artifact["path"]) != artifact["sha256"]:
            raise ValueError("source_hash_mismatch")
    with safe_open(snapshot / "adapters/retrieval/adapter_model.safetensors", framework="pt") as f:
        if len(f.keys()) != 392 or any(f.get_slice(k).get_dtype() != "BF16" for k in f.keys()):
            raise ValueError("adapter_tensor_contract")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write(a.output_dir / "native-provenance.json", provenance)
    backend = TorchBackend(a.model_root)
    backend.load()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    audit = merge_retrieval(backend._model)
    write(a.output_dir / "merge-audit.json", audit)
    model = EmbeddingGraph(backend._model).eval()
    encoded = backend._tokenizer(
        ["Query: Kunst", "Document: Musik und Kunst im Park."], padding=True, return_tensors="pt"
    )
    print("merge verified; exporting dynamic embedding graph", flush=True)
    torch.onnx.export(
        model,
        (encoded["input_ids"], encoded["attention_mask"]),
        str(a.output_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes=(
            {0: torch.export.Dim("batch"), 1: torch.export.Dim("sequence")},
            {0: torch.export.Dim("batch"), 1: torch.export.Dim("sequence")},
        ),
        external_data=True,
        optimize=False,
    )
    del model, backend
    graph = onnx.load(a.output_dir / "model.onnx", load_external_data=False)
    if any(
        t.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16)
        for t in graph.graph.initializer
    ):
        raise ValueError("non_fp32_weights")
    onnx.checker.check_model(str(a.output_dir / "model.onnx"))
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copyfile(snapshot / name, a.output_dir / name)
    manifest = {
        "schema": "jina-v5-merged-onnx-v1",
        "backend": "v5-onnx-merged",
        "model_repository": provenance["model_repository"],
        "model_revision": provenance["model_revision"],
        "contract": provenance["contract"],
        "export_commit": a.source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "export_source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): sha(p)
            for p in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]
                / "src/uranus_research_encoder/v5_onnx_export.py",
                Path(__file__).resolve().parents[1]
                / "src/uranus_research_encoder/merged_export.py",
                Path(__file__).resolve().parents[1] / "src/uranus_research_encoder/model.py",
            ]
        },
        "plan_sha256": sha(a.plan_dir / "plan.json"),
        "design": "B",
        "opset": 18,
        "versions": provenance["versions"],
        "optimization": "disabled",
        "artifacts": {p.name: sha(p) for p in sorted(a.output_dir.iterdir()) if p.is_file()},
    }
    write(a.output_dir / "manifest.json", manifest)
    print("export complete", flush=True)


if __name__ == "__main__":
    main()
