#!/usr/bin/env python3
"""Fresh full-corpus vectors through the unchanged public backend classes, offline."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
from check_v5_onnx_parity import sha

from uranus_research_encoder.model import TorchBackend, cached_snapshot
from uranus_research_encoder.v5_onnx_artifacts import verify_manifest, verify_native_snapshot
from uranus_research_encoder.v5_onnx_backend import V5MergedOnnxBackend


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["native", "onnx"])
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.plan_dir / "full-corpus-plan.json").read_text())
    path = a.plan_dir / "full-corpus-inputs.json"
    require(sha(path) == plan["inputs_sha256"], "STOP: frozen_inputs_changed")
    source = Path(__file__).resolve().parents[1]
    for name, expected in plan["encoder_source_sha256"].items():
        require(sha(source / name) == expected, "STOP: encoder_source_changed:" + name)
    for name, expected in plan["versions"].items():
        require(importlib.metadata.version(name) == expected, "STOP: version_changed:" + name)
    verify_manifest(a.graph_dir, plan["manifest_sha256"])
    verify_native_snapshot(cached_snapshot(a.model_root, "torch"))
    inputs = json.loads(path.read_text())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    destination = a.output_dir / f"full-corpus-{a.backend}.npz"
    context_path = a.output_dir / f"full-corpus-{a.backend}-context.json"
    require(not destination.exists() and not context_path.exists(), "immutable_output_exists")
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) >= 8, "eight_cpus_required")
    os.sched_setaffinity(0, affinity[:8])
    started = time.perf_counter()
    if a.backend == "native":
        import torch

        torch.set_num_interop_threads(1)
        backend = TorchBackend(a.model_root)
        backend.load()
        torch.set_num_threads(8)
    else:
        backend = V5MergedOnnxBackend(
            a.graph_dir,
            manifest_sha256=plan["manifest_sha256"],
            intra_op_threads=8,
            inter_op_threads=1,
        )
        backend.load()
    points = inputs["point_ids"]
    texts = list(dict.fromkeys(inputs["chunks"][p]["chunk"]["text"] for p in points))
    # Exact string memoization only. All 2080 original point IDs are expanded below;
    # no chunks are merged, excluded, retokenized, or regenerated.
    vectors, tokens = [], []
    for i, text in enumerate(texts):
        vectors.append(backend.embed([text], "passage")[0])
        tokens.append(backend.count(text, "passage"))
        if (i + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "backend": a.backend,
                        "unique_passages": i + 1,
                        "total_unique_passages": len(texts),
                    }
                ),
                flush=True,
            )
    lookup = {text: i for i, text in enumerate(texts)}
    expansion = [lookup[inputs["chunks"][pid]["chunk"]["text"]] for pid in points]
    queries = inputs["queries"]
    qvectors, qtokens = [], []
    for i, query in enumerate(queries):
        qvectors.append(backend.embed([query["text"]], "query")[0])
        qtokens.append(backend.count(query["text"], "query"))
        if (i + 1) % 20 == 0:
            print(json.dumps({"backend": a.backend, "queries": i + 1}), flush=True)
    dvectors = np.asarray(vectors, dtype=np.float32)[expansion]
    dtokens = np.asarray(tokens, dtype=np.int64)[expansion]
    require(
        dtokens.tolist() == [inputs["chunks"][pid]["chunk"]["token_count"] for pid in points],
        "STOP: benchmark_token_counts_changed",
    )
    np.savez_compressed(
        destination,
        documents=dvectors,
        queries=np.asarray(qvectors, dtype=np.float32),
        document_tokens=dtokens,
        query_tokens=np.asarray(qtokens, dtype=np.int64),
    )
    context = {
        "schema": "jina-v5-full-corpus-execution-v1",
        "backend": a.backend,
        "plan_sha256": sha(a.plan_dir / "full-corpus-plan.json"),
        "inputs_sha256": sha(path),
        "manifest_sha256": plan["manifest_sha256"],
        "encoder_source_sha256": plan["encoder_source_sha256"],
        "vectors_sha256": sha(destination),
        "versions": plan["versions"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": 8,
        "inter_op_threads": 1,
        "events": len(inputs["documents"]),
        "chunks": len(points),
        "unique_passages_embedded": len(texts),
        "queries_embedded": len(queries),
        "max_chunk_tokens": int(dtokens.max()),
        "elapsed_seconds_including_load": time.perf_counter() - started,
        "timing_scope": "Corpus generation only; not a controlled performance measurement",
    }
    with context_path.open("x") as f:
        json.dump(context, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {"backend": a.backend, "status": "complete", "vectors_sha256": sha(destination)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
