#!/usr/bin/env python3
"""Opt-in, offline real-model comparison. Run each backend in a separate process."""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from uranus_research_encoder.merged_artifacts import digest
from uranus_research_encoder.merged_onnx_backend import MergedOnnxBackend
from uranus_research_encoder.model import TorchBackend
from uranus_research_encoder.onnx_backend import OnnxBackend
from uranus_research_encoder.version import EMBEDDING_VERSION, MODEL_REVISION

QUERIES = [
    "Was ist heute kulturell besonders spannend?",
    "Communitytreffen in Flensburg",
    "kostenlose Workshops und Technikveranstaltungen",
    "cultural events this evening",
    "community technology workshop",
]
PASSAGES = [
    "Heute Abend spielt ein Jazztrio im Kulturhof. Ab 19 Uhr gibt es Live-Musik, "
    "Gespräche und eine Ausstellung regionaler Künstlerinnen. Der Eintritt kostet zehn Euro.",
    "Communitytreffen in Flensburg: Im Aktivitetshuset treffen sich am Samstag Menschen "
    "aus der Nachbarschaft. Gemeinsam planen wir Kulturangebote, tauschen Ideen aus und "
    "lernen neue Initiativen kennen. Die Teilnahme ist kostenlos.",
    "Kostenloser Technikworkshop: In unserer offenen Werkstatt reparieren wir Fahrräder, "
    "lernen Löten und probieren kleine Computerprojekte aus. Vorkenntnisse sind nicht nötig.",
    "Tonight the independent cinema presents a documentary followed by a discussion "
    "with the director. Doors open at 18:30; the screening starts at 19:00.",
    "Join our community technology workshop to learn practical electronics and share "
    "repair skills. Volunteers provide tools and help beginners. Admission is free.",
    "Der Wochenmarkt bietet am Mittwoch frisches Gemüse, Brot und regionale Lebensmittel. "
    "Die Stände stehen von acht bis dreizehn Uhr auf dem Marktplatz.",
    "Ein gemeinsamer Spaziergang führt durch den Wald und an der Förde entlang. "
    "Bitte wetterfeste Kleidung mitbringen. Treffpunkt ist der Parkplatz am Strand.",
]
LONG_TEXT = "\n\n".join(
    f"Programmpunkt {i + 1}: {PASSAGES[i % len(PASSAGES)]} "
    "Anmeldung und Informationen zur Barrierefreiheit gibt es beim Kulturteam. "
    "Bei schlechtem Wetter findet die Veranstaltung in den Innenräumen statt."
    for i in range(14)
)
TEXTS = QUERIES + PASSAGES + [LONG_TEXT]

# Frozen after the pinned float32 run: worst component error 4.78e-7 and
# cosine loss 5.54e-12. Small rounded margins, not a different-space tolerance.
MAX_ABSOLUTE_DIFFERENCE = 1e-6
MIN_COSINE_SIMILARITY = 1 - 1e-11


def acceptance(report: dict) -> bool:
    report["thresholds"] = {
        "max_absolute_difference": MAX_ABSOLUTE_DIFFERENCE,
        "min_cosine_similarity": MIN_COSINE_SIMILARITY,
        "norm_absolute_tolerance": 1e-6,
        "ranking": "identical full ordering",
    }
    report["passed"] = bool(
        report["token_counts_match"]
        and report["rankings_identical"]
        and report["task_specific"]
        and report["stable"]
        and report["max_absolute_difference"] <= MAX_ABSOLUTE_DIFFERENCE
        and report["min_cosine_similarity"] >= MIN_COSINE_SIMILARITY
        and all(
            abs(row[name] - 1) <= 1e-6
            for row in report["metrics"]
            for name in ("torch_norm", "onnx_norm")
        )
    )
    return report["passed"]


def worker(args) -> None:
    backend = (
        TorchBackend(args.model_root)
        if args.worker == "torch"
        else MergedOnnxBackend(args.merged_root, intra_op_threads=args.threads)
        if args.worker == "onnx-merged"
        else OnnxBackend(args.model_root, intra_op_threads=args.threads)
    )
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started
    print(f"{args.worker}: loaded in {load_seconds:.3f}s", flush=True)
    result = {
        "backend": args.worker,
        "runtime": backend.runtime,
        "load_seconds": load_seconds,
        "token_counts": [backend.count(text) for text in TEXTS],
        "intra_op_threads": 1 if args.worker == "torch" else args.threads,
    }
    if args.worker == "onnx-merged":
        result["manifest_sha256"] = digest(args.merged_root / "manifest.json")
    for kind in ("query", "passage"):
        vectors = backend.embed(TEXTS, kind)
        result[kind] = vectors
        # Stability and request-batching invariance are part of the contract.
        result[kind + "_stable"] = vectors[0] == backend.embed([TEXTS[0]], kind)[0]
        result[kind + "_batch_stable"] = vectors[:2] == backend.embed(TEXTS[:2], kind)
        print(f"{args.worker}: {kind} corpus and stability checks complete", flush=True)
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.output.write_text(json.dumps(result))


def compare(reference: dict, candidate: dict) -> dict:
    candidate_name = candidate.get("backend", "onnx")
    metrics = []
    for kind in ("query", "passage"):
        left, right = np.asarray(reference[kind]), np.asarray(candidate[kind])
        if left.shape != (len(TEXTS), 1024) or right.shape != left.shape:
            raise ValueError("dimension_mismatch")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("nonfinite_embeddings")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            metrics.append(
                {
                    "text_index": index,
                    "kind": kind,
                    "dimensions": len(b),
                    "tokens": reference["token_counts"][index],
                    "torch_norm": float(np.linalg.norm(a)),
                    "onnx_norm": float(np.linalg.norm(b)),
                    "cosine_similarity": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                    "max_absolute_difference": float(np.max(np.abs(a - b))),
                }
            )
    rankings = {}
    for name, result in (("torch", reference), (candidate_name, candidate)):
        query = np.asarray(result["query"][: len(QUERIES)])
        passage = np.asarray(result["passage"][len(QUERIES) : len(QUERIES) + len(PASSAGES)])
        scores = query @ passage.T
        rankings[name] = np.argsort(-scores, axis=1, kind="stable").tolist()
    return {
        "candidate_backend": candidate.get("backend", "onnx"),
        "manifest_sha256": candidate.get("manifest_sha256"),
        "model_revision": MODEL_REVISION,
        "embedding_version": EMBEDDING_VERSION,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "token_counts_match": reference["token_counts"] == candidate["token_counts"],
        "metrics": metrics,
        "min_cosine_similarity": min(row["cosine_similarity"] for row in metrics),
        "max_absolute_difference": max(row["max_absolute_difference"] for row in metrics),
        "rankings": rankings,
        "rankings_identical": rankings["torch"] == rankings[candidate_name],
        "task_specific": all(
            a != b
            for result in (reference, candidate)
            for a, b in zip(result["query"], result["passage"], strict=True)
        ),
        "stable": all(
            result[kind + suffix]
            for result in (reference, candidate)
            for kind in ("query", "passage")
            for suffix in ("_stable", "_batch_stable")
        ),
        "runtime_measurements": {
            name: {key: result[key] for key in ("runtime", "load_seconds", "peak_rss_mib")}
            for name, result in (("torch", reference), (candidate_name, candidate))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--merged-root", type=Path)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--worker", choices=("torch", "onnx", "onnx-merged"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if os.environ.get("JINA_NONCOMMERCIAL") != "1":
        parser.error("set JINA_NONCOMMERCIAL=1 to acknowledge the model license")
    if not args.model_root.is_absolute():
        parser.error("--model-root must be absolute")
    if args.worker:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="jina-parity-") as directory:
        results = []
        for backend in ["torch", "onnx", "onnx-merged"] if args.merged_root else ["torch", "onnx"]:
            output = Path(directory) / f"{backend}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model-root",
                    str(args.model_root),
                    "--output",
                    str(output),
                    "--threads",
                    str(args.threads),
                    "--worker",
                    backend,
                    *(["--merged-root", str(args.merged_root)] if args.merged_root else []),
                ],
                check=True,
                env=os.environ | ({"USE_TORCH": "0"} if backend != "torch" else {}),
            )
            results.append(json.loads(output.read_text()))
        reports = [compare(results[0], candidate) for candidate in results[1:]]
    for report in reports:
        acceptance(report)
    report = (
        reports[0]
        if len(reports) == 1
        else {"comparisons": reports, "passed": all(row["passed"] for row in reports)}
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("kind     text  tokens  cosine similarity  max abs difference")
    for row in reports[-1]["metrics"]:
        print(
            f"{row['kind']:8} {row['text_index']:4} {row['tokens']:7} "
            f"{row['cosine_similarity']:.12f} {row['max_absolute_difference']:.9g}"
        )
    for comparison in reports:
        print(
            f"{comparison['candidate_backend']}: passed={comparison['passed']}; "
            f"rankings={comparison['rankings_identical']}; stable={comparison['stable']}"
        )
    if not report["passed"]:
        raise SystemExit("Torch/ONNX parity failed; do not change the production backend")


if __name__ == "__main__":
    main()
