#!/usr/bin/env python3
"""Export isolated, task-free float32 graphs from an existing pinned native cache."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from uranus_research_encoder.merged_artifacts import (
    EXPORT_CONTRACT,
    EXPORTER_VERSION,
    OPSET,
    TASKS,
    digest,
)
from uranus_research_encoder.model import cached_snapshot, offline_environment
from uranus_research_encoder.model_artifacts import TOKENIZER_FILES
from uranus_research_encoder.version import DIMENSIONS, MODEL_REPOSITORY, MODEL_REVISION


def export_graph(model, destination: Path) -> None:
    import onnx
    import torch

    from uranus_research_encoder.merged_export import TokenEmbeddings

    ids = torch.tensor([[0, 4, 5, 2], [0, 5, 2, 1]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1, max=16)
    sequence = torch.export.Dim("sequence", min=2, max=8192)
    with torch.inference_mode():
        wrapper = TokenEmbeddings(model).eval()
        if not torch.equal(
            model(input_ids=ids, attention_mask=mask).last_hidden_state, wrapper(ids, mask)
        ):
            raise ValueError("attention_mask_wrapper_mismatch")
        torch.onnx.export(
            wrapper,
            (ids, mask),
            str(destination),
            dynamo=True,
            external_data=True,
            opset_version=OPSET,
            optimize=False,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            dynamic_shapes=({0: batch, 1: sequence}, {0: batch, 1: sequence}),
        )
    graph = onnx.load(destination, load_external_data=False)
    if [item.name for item in graph.graph.input] != ["input_ids", "attention_mask"]:
        raise ValueError("export_input_mismatch")
    if [item.name for item in graph.graph.output] != ["text_embeds"]:
        raise ValueError("export_output_mismatch")
    for tensor in graph.graph.initializer:
        if "lora_" in tensor.name:
            raise ValueError("unmerged_export")
        for entry in tensor.external_data:
            if entry.key == "location" and (
                Path(entry.value).name != entry.value
                or not (destination.parent / entry.value).is_file()
            ):
                raise ValueError("unsafe_external_data")
    onnx.checker.check_model(str(destination))


def worker(args, snapshot: Path) -> None:
    import numpy as np
    import torch
    from compare_backends import MAX_ABSOLUTE_DIFFERENCE, MIN_COSINE_SIMILARITY, TEXTS
    from transformers import AutoModel, AutoTokenizer

    from uranus_research_encoder.merged_export import merge_verified

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("incomplete_weights")
    if model.config.model_type != "jina_embeddings_v3" or model.config.hidden_size != DIMENSIONS:
        raise ValueError("model_mismatch")
    adapter = TASKS[args.worker]
    loading = model.load_adapter(
        str(snapshot / adapter), adapter_name=adapter, adapter_kwargs={"local_files_only": True}
    )
    if any(f".{adapter}." in key for key in loading.missing_keys) or loading.mismatched_keys:
        raise ValueError("incomplete_adapter")
    model.set_adapter(adapter)
    model.to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )

    def vectors():
        result = []
        with torch.inference_mode():
            for text in TEXTS:
                inputs = tokenizer(text, return_tensors="pt", truncation=False)
                hidden = model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"][..., None]
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                result.append(torch.nn.functional.normalize(pooled, dim=1)[0].numpy().copy())
        return np.asarray(result, dtype=np.float64)

    reference = vectors()
    audit = merge_verified(model, adapter)
    merged = vectors()
    metrics = {
        "max_absolute_difference": float(np.max(np.abs(reference - merged))),
        "min_cosine_similarity": float(
            np.min(
                np.sum(reference * merged, axis=1)
                / (np.linalg.norm(reference, axis=1) * np.linalg.norm(merged, axis=1))
            )
        ),
        "max_norm_error": float(np.max(np.abs(np.linalg.norm(merged, axis=1) - 1))),
    }
    passed = bool(
        np.isfinite(merged).all()
        and metrics["max_absolute_difference"] <= MAX_ABSOLUTE_DIFFERENCE
        and metrics["min_cosine_similarity"] >= MIN_COSINE_SIMILARITY
        and metrics["max_norm_error"] <= 1e-6
    )
    task_dir = args.output_dir / f"retrieval-{args.worker}"
    task_dir.mkdir()
    (task_dir / "merge-audit.json").write_text(
        json.dumps(
            {
                "task": args.worker,
                "source_adapter": adapter,
                "weights": audit,
                "native_merge_parity": metrics,
                "passed": passed,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit("Native merge parity failed; stopping before export")
    export_graph(model, task_dir / "model.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", choices=TASKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("JINA_NONCOMMERCIAL") != "1":
        parser.error("set JINA_NONCOMMERCIAL=1 to acknowledge the model license")
    offline_environment()
    for package, expected in {
        "torch": "2.11.0",
        "transformers": "5.17.0",
        "peft": "0.21.1",
        "onnx": "1.23.1",
        "onnxscript": "0.6.2",
    }.items():
        if version(package).split("+")[0] != expected:
            parser.error(f"export contract requires {package}=={expected}")
    source = args.model_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("output must be isolated from the source cache")
    snapshot = cached_snapshot(source, "torch")
    # Resolved symlink targets also cannot overlap the output directory.
    source_files = [
        p for p in snapshot.rglob("*") if p.is_file() and p.relative_to(snapshot).parts[0] != "onnx"
    ]
    if any(p.resolve().is_relative_to(output) for p in source_files):
        parser.error("output overlaps source blobs")
    if args.worker:
        worker(args, snapshot)
        return
    if output.exists() and not args.force:
        parser.error("output exists; use --force to replace a previous export")
    if output.exists():
        old = json.loads((output / "manifest.json").read_text())
        if old.get("contract") != EXPORT_CONTRACT:
            parser.error("--force only replaces an identified merged export")
    source_hashes = {p.relative_to(snapshot).as_posix(): digest(p) for p in source_files}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Incomplete exports are never published as a loadable artifact directory.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for kind in TASKS:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model-root",
                    str(source),
                    "--output-dir",
                    str(stage),
                    "--worker",
                    kind,
                ],
                check=True,
                stdout=sys.stderr,  # Keep stdout exclusively the deterministic manifest JSON.
            )
        (stage / "tokenizer").mkdir()
        for name in TOKENIZER_FILES:
            if (snapshot / name).is_file():
                shutil.copyfile(snapshot / name, stage / "tokenizer" / name)
        if source_hashes != {p.relative_to(snapshot).as_posix(): digest(p) for p in source_files}:
            raise ValueError("source_changed_during_export")
        manifest = {
            "contract": EXPORT_CONTRACT,
            "exporter_version": EXPORTER_VERSION,
            "source": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "source_sha256": source_hashes,
            "versions": {
                name: version(name)
                for name in ("torch", "transformers", "peft", "onnx", "onnxscript")
            },
            "opset": OPSET,
            "dtype": "float32",
            "exporter": "torch.onnx dynamo; optimize=False",
            "tasks": {
                kind: {
                    "task": kind,
                    "source_adapter": adapter,
                    "graph": f"retrieval-{kind}/model.onnx",
                }
                for kind, adapter in TASKS.items()
            },
            "files": {
                p.relative_to(stage).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
            "release_parity": "not yet validated; run compare_backends.py",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        # Publish artifacts readable by the unprivileged inference image. mkdtemp
        # intentionally kept the staging directory private until export completed.
        for path in stage.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        stage.chmod(0o755)
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
        print(json.dumps(manifest, sort_keys=True, indent=2))
    except BaseException:
        # Keep failure diagnostics for review, but never an apparently complete manifest.
        print(f"Export failed; diagnostic staging directory: {stage}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
