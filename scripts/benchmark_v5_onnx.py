#!/usr/bin/env python3
"""Isolated process benchmark; requires all preregistered parity gates to pass."""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=["native", "onnx"])
    p.add_argument("--threads", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument(
        "--workload",
        action="append",
        choices=["query", "passage", "batch3"],
        help="Optional workload subset, e.g. query-only thread scaling",
    )
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--parity-report", type=Path, required=True)
    p.add_argument("--contract-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("benchmark_output_already_exists")
    plan = json.loads((a.plan_dir / "plan.json").read_text())
    report = json.loads(a.parity_report.read_text())
    if not report["pass"] or report["plan_sha256"] != sha(a.plan_dir / "plan.json"):
        raise ValueError("parity_gate_not_passed")
    selection = json.loads((a.plan_dir / "selection.json").read_text())
    if sha(a.plan_dir / "selection.json") != plan["selection_sha256"]:
        raise ValueError("selection_changed")
    manifest = json.loads((a.graph_dir / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        if sha(a.graph_dir / name) != digest:
            raise ValueError("artifact_changed")
    context = json.loads((a.parity_report.parent / "onnx-context.json").read_text())
    if context["graph_manifest_sha256"] != sha(a.graph_dir / "manifest.json"):
        raise ValueError("untested_graph")
    contract = json.loads(a.contract_report.read_text())
    if contract.get("pass") is not True or contract.get("manifest_sha256") != sha(
        a.graph_dir / "manifest.json"
    ):
        raise ValueError("contract_gate_not_passed")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        raise ValueError("eight_cpus_required")
    os.sched_setaffinity(0, cpus[:8])
    from uranus_research_encoder.model import TorchBackend, offline_environment

    offline_environment()
    from transformers import AutoTokenizer

    started = time.perf_counter()
    if a.backend == "native":
        import torch

        torch.set_num_interop_threads(1)
        backend = TorchBackend(a.model_root)
        backend.load()
        torch.set_num_threads(a.threads)
        tokenizer = backend._tokenizer

        def embed(rows):
            # Same mixed-batch semantics as the current public encoder: serialize
            # each role; a homogeneous request keeps one forward per input.
            return np.asarray([backend.embed([r["text"]], r["kind"])[0] for r in rows])
    else:
        import onnxruntime as ort

        tokenizer = AutoTokenizer.from_pretrained(
            a.graph_dir, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "right"
        options = ort.SessionOptions()
        options.intra_op_num_threads = a.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(a.graph_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

        def embed(rows):
            encoded = tokenizer(
                [TorchBackend._input(r["text"], r["kind"]) for r in rows],
                padding=True,
                truncation=False,
                return_tensors="np",
            )
            return session.run(
                ["text_embeds"],
                {
                    k: np.asarray(encoded[k], dtype=np.int64)
                    for k in ["input_ids", "attention_mask"]
                },
            )[0]

    load_seconds = time.perf_counter() - started
    queries = [
        r
        for r in selection["inputs"]
        if r["kind"] == "query" and not r["id"].startswith("synthetic")
    ]
    docs = [
        r
        for r in selection["inputs"]
        if r["kind"] == "passage" and not r["id"].startswith("synthetic")
    ]
    # All real frozen queries and a deterministic cycle over real selected chunks.
    n = plan["performance"]["samples_per_workload"]
    workloads = {
        "query": [[queries[i % len(queries)]] for i in range(n)],
        "passage": [[docs[i % len(docs)]] for i in range(n)],
        "batch3": [[docs[(3 * i + j) % len(docs)] for j in range(3)] for i in range(n)],
    }
    start = time.perf_counter()
    embed(workloads["query"][0])
    first = time.perf_counter() - start
    result = {
        "schema": "jina-v5-runtime-performance-v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "onnxruntime")},
        "runner_sha256": sha(Path(__file__)),
        "backend": a.backend,
        "threads": a.threads,
        "inter_op_threads": 1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host": platform.node(),
        "platform": platform.platform(),
        "load_seconds": load_seconds,
        "first_inference_seconds": first,
        "parity_report_sha256": sha(a.parity_report),
        "contract_report_sha256": sha(a.contract_report),
        "plan_sha256": sha(a.plan_dir / "plan.json"),
        "manifest_sha256": sha(a.graph_dir / "manifest.json"),
        "workloads": {},
        "scope": (
            "Shared AI host isolated CPU embedding calls including tokenizer; "
            "no HTTP, Qdrant or retrieval transport. No production capacity claim."
        ),
        "native_batching": "existing one-text-per-forward; ONNX uses padded batch",
    }
    for kind, batches in workloads.items():
        if a.workload and kind not in a.workload:
            continue
        for i in range(plan["performance"]["warmup_per_workload"]):
            embed(batches[i])
        samples = []
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        for rows in batches:
            start = time.perf_counter()
            vectors = embed(rows)
            samples.append(time.perf_counter() - start)
            if vectors.shape != (len(rows), 1024) or not np.isfinite(vectors).all():
                raise ValueError("invalid_benchmark_vector")
        wall = time.perf_counter() - wall_start
        result["workloads"][kind] = {
            "samples": len(samples),
            "warmup": plan["performance"]["warmup_per_workload"],
            "mean_ms": 1000 * statistics.mean(samples),
            "median_ms": 1000 * statistics.median(samples),
            "p95_ms": 1000 * float(np.percentile(samples, 95)),
            "p99_ms": 1000 * float(np.percentile(samples, 99)) if len(samples) >= 100 else None,
            "texts_per_second": sum(map(len, batches)) / wall,
            "wall_seconds": wall,
            "cpu_seconds": time.process_time() - cpu_start,
            "peak_rss_kib_process_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "latency_seconds": samples,
            "input_ids": [[r["id"] for r in rows] for rows in batches],
        }
        print(a.backend, a.threads, kind, result["workloads"][kind]["mean_ms"], flush=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
