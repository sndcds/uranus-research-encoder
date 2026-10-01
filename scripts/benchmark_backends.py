#!/usr/bin/env python3
"""Opt-in CPU benchmark with warmups, process isolation, JSON, and a readable table."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np

from uranus_research_encoder.merged_artifacts import digest
from uranus_research_encoder.merged_batching import embed_padded
from uranus_research_encoder.merged_onnx_backend import MergedOnnxBackend
from uranus_research_encoder.model import TorchBackend
from uranus_research_encoder.onnx_backend import OnnxBackend
from uranus_research_encoder.version import MODEL_REVISION

SEED_TEXT = (
    "Beim Communitytreffen in Flensburg gibt es kostenlose Kulturangebote und offene "
    "Technikworkshops. Gemeinsam reparieren wir Geräte, lernen neue Menschen kennen "
    "und planen Veranstaltungen für die Nachbarschaft. Die Räume sind barrierefrei. "
)


def representative_text(backend, target: int) -> str:
    # Use the actual pinned tokenizer; preserve text input through both backends.
    words = (SEED_TEXT * (target // 10 + 1)).split()
    low, high = 1, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if backend.count(" ".join(words[:middle])) <= target:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


def worker(args) -> None:
    if args.worker == "torch":
        import torch

        torch.set_num_interop_threads(1)
    backend = (
        TorchBackend(args.model_root)
        if args.worker == "torch"
        else MergedOnnxBackend(
            args.merged_root,
            intra_op_threads=args.threads,
            optimization=args.merged_optimization,
        )
        if args.worker == "onnx-merged"
        else OnnxBackend(args.model_root, intra_op_threads=args.threads)
    )
    start = time.perf_counter()
    backend.load()
    if args.worker == "torch":
        # Benchmark-only override: never change TorchBackend's production default.
        torch.set_num_threads(args.threads)
    embed = partial(embed_padded, backend) if args.padded_batching else backend.embed
    result = {
        "backend": args.worker,
        "runtime": backend.runtime,
        "load_seconds": time.perf_counter() - start,
        "intra_op_threads": args.threads,
        "inter_op_threads": 1
        if args.worker != "torch"
        else __import__("torch").get_num_interop_threads(),
        "cases": [],
        "inference_batching": "padded groups of 4" if args.padded_batching else "sequential",
    }
    if args.worker == "onnx-merged":
        result["manifest_sha256"] = digest(args.merged_root / "manifest.json")
        result["graph_optimization"] = args.merged_optimization
    # Model-load measurement is process-cold; the OS page cache is not flushed.
    for length in args.lengths:
        text = representative_text(backend, length)
        count = backend.count(text)
        for kind in ("query", "passage"):
            for batch_size in args.batch_sizes:
                texts = [text] * batch_size
                for _ in range(args.warmups):
                    embed(texts, kind)
                durations = []
                for _ in range(args.samples):
                    start = time.perf_counter()
                    embed(texts, kind)
                    durations.append(time.perf_counter() - start)
                result["cases"].append(
                    {
                        "kind": kind,
                        "target_tokens": length,
                        "actual_tokens": count,
                        "batch_size": batch_size,
                        "warmups": args.warmups,
                        "samples": len(durations),
                        "seconds": durations,
                        "p50_ms": float(np.percentile(durations, 50) * 1000),
                        "p95_ms": float(np.percentile(durations, 95) * 1000),
                        "texts_per_second": batch_size * len(durations) / sum(durations),
                        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    }
                )
                # Checkpoint after each case, without logging input text or vectors.
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merged-root", type=Path)
    parser.add_argument("--merged-optimization", choices=("disabled", "basic"), default="disabled")
    parser.add_argument(
        "--padded-batching",
        action="store_true",
        help="separate merged-only experiment: pad groups of up to four texts",
    )
    parser.add_argument(
        "--backends", nargs="+", choices=("torch", "onnx", "onnx-merged"), default=["torch", "onnx"]
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        choices=(32, 128, 480, 1024, 4096, 8192),
        default=[32, 128, 480, 1024],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=(1, 4, 16), default=[1, 4])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--torch-threads",
        type=int,
        choices=range(1, 9),
        default=1,
        help="benchmark-only Torch intra-op override; production remains one thread",
    )
    parser.add_argument(
        "--thread-sweep",
        nargs="+",
        type=int,
        choices=(1, 2, 4, 8),
        help="intra-op sweep for onnx-merged; --threads controls upstream ONNX",
    )
    parser.add_argument(
        "--worker", choices=("torch", "onnx", "onnx-merged"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if os.environ.get("JINA_NONCOMMERCIAL") != "1":
        parser.error("set JINA_NONCOMMERCIAL=1 to acknowledge the model license")
    if not args.model_root.is_absolute():
        parser.error("--model-root must be absolute")
    if args.warmups < 2 or args.samples < 5:
        parser.error("at least two warmups and five measured samples are required")
    if ("onnx-merged" in args.backends or args.worker == "onnx-merged") and not args.merged_root:
        parser.error("--merged-root is required for onnx-merged")
    if args.padded_batching and (
        (not args.worker and args.backends != ["onnx-merged"])
        or args.worker not in (None, "onnx-merged")
    ):
        parser.error("--padded-batching requires only onnx-merged")
    if args.worker:
        worker(args)
        return
    report = {
        "model_revision": MODEL_REVISION,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        },
        "load_measurement": "fresh process; operating-system file cache not flushed",
        "batch_behavior": "padded groups of four (benchmark-only)"
        if args.padded_batching
        else "all service backends process texts sequentially per request",
        "thread_comparison": f"Torch benchmark uses {args.torch_threads} intra-op threads; "
        "production Torch remains one thread. ONNX settings are separately labeled",
        "rss_measurement": "process lifetime high-water mark; Linux ru_maxrss",
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="jina-benchmark-") as directory:
        runs = [
            (name, threads)
            for name in args.backends
            for threads in (
                (args.thread_sweep or [args.threads])
                if name == "onnx-merged"
                else [args.threads]
                if name == "onnx"
                else [args.torch_threads]
            )
        ]
        for name, threads in runs:
            output = Path(directory) / f"{name}-{threads}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-root",
                str(args.model_root),
                "--output",
                str(output),
                "--threads",
                str(threads),
                "--merged-optimization",
                args.merged_optimization,
                "--worker",
                name,
                "--warmups",
                str(args.warmups),
                "--samples",
                str(args.samples),
                "--lengths",
                *map(str, args.lengths),
                "--batch-sizes",
                *map(str, args.batch_sizes),
                *(["--merged-root", str(args.merged_root)] if args.merged_root else []),
                *(["--padded-batching"] if args.padded_batching else []),
            ]
            process = subprocess.run(
                command,
                check=False,
                env=os.environ | ({"USE_TORCH": "0"} if name != "torch" else {}),
            )
            result = (
                json.loads(output.read_text())
                if output.exists()
                else {"backend": name, "intra_op_threads": threads, "cases": []}
            )
            result["requested_intra_op_threads"] = threads
            result["exit_code"] = process.returncode
            result["expected_cases"] = len(args.lengths) * 2 * len(args.batch_sizes)
            result["complete"] = (
                process.returncode == 0 and len(result["cases"]) == result["expected_cases"]
            )
            report["results"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not result["complete"]:
                # In particular, do not keep allocating real-model workers after
                # an OOM kill. Preserve completed samples and stop the experiment.
                break
    print("backend     threads kind     tokens batch    p50 ms    p95 ms   texts/s  peak RSS MiB")
    for result in report["results"]:
        for case in result["cases"]:
            print(
                f"{result['backend']:11} {result['intra_op_threads']:7} "
                f"{case['kind']:8} {case['actual_tokens']:6} "
                f"{case['batch_size']:5} {case['p50_ms']:9.2f} {case['p95_ms']:9.2f} "
                f"{case['texts_per_second']:9.3f} {case['peak_rss_mib']:12.1f}"
            )
    if any(not result["complete"] for result in report["results"]):
        raise SystemExit("benchmark incomplete; see recorded worker exit codes")


if __name__ == "__main__":
    main()
