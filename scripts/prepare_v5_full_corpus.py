#!/usr/bin/env python3
"""Verify original benchmark bytes and freeze judgment-free full-corpus inputs.

Run with the existing Research Python environment and its src on PYTHONPATH.
Only pure snapshot/document/eligibility validators are called; no benchmark runner,
service, scoring policy, judgment evaluator, or Phase 2D module is executed.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import UUID, uuid5

RESEARCH_COMMIT = "c643acbc30a0f2f8e80a15b8b2aa824076ca9d38"
PINS = {
    "benchmark/snapshots/public-events-20261005/events.jsonl": (
        "2de49bc71942926e953f4de76d4b5dbcfe1780a6a2a20ee2b0feb6fa97cbf7f1"
    ),
    "benchmark/snapshots/public-events-20261005/manifest.json": (
        "45b8d6c0ed0220d678f8027b5258674d2585d71122c14d7f9488ad94abb116e1"
    ),
    "benchmark/ground-truth-v1.jsonl": (
        "a72b04ca44655ca4e8ba456e8d9698ff6ebbe47e1dde54f075780c9e6e4a8b53"
    ),
    "benchmark/query-proposals-v1.json": (
        "6cf266259b3c07d92d991cc4bd760406bba8d0e62ff78c87d78d48795f1854f2"
    ),
    "benchmark/results/chunks-v5-20261006_8cpu_001.json": (
        "cfcfdfaa9c2ed30400f86affe16036b6967a66b19055737de133d07665885ce4"
    ),
    "benchmark/results/v5-20261006_8cpu_001.json": (
        "e0ef32e5dba8d51d3a8086d73557e2f2b3418ee729b6e4863cf1c0a9f9f2760b"
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "immutable_input_exists")
    root = a.research_root
    for name, expected in PINS.items():
        require(sha(root / name) == expected, "STOP: benchmark_hash_mismatch:" + name)
    source_hashes = {}
    for name in (
        "ground_truth.py",
        "ground_truth_snapshot.py",
        "controlled_runner.py",
        "controlled_evaluation.py",
        "encoder_contracts.py",
        "semantic_manifest.py",
        "research/vector_documents.py",
    ):
        relative = "src/uranus_research_service/" + name
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", RESEARCH_COMMIT + ":" + relative]
        )
        require(original == (root / relative).read_bytes(), "STOP: research_source_changed")
        source_hashes[relative] = sha(root / relative)
    from uranus_research_service.controlled_runner import NAMESPACE, documents, validate_chunk
    from uranus_research_service.encoder_contracts import Chunk
    from uranus_research_service.ground_truth import Eligibility
    from uranus_research_service.ground_truth_snapshot import load_snapshot
    from uranus_research_service.semantic_manifest import digest

    base = root / "benchmark"
    manifest, events = load_snapshot(
        base / "snapshots/public-events-20261005/events.jsonl",
        base / "snapshots/public-events-20261005/manifest.json",
    )
    result = json.loads((base / "results/v5-20261006_8cpu_001.json").read_text())
    chunks = json.loads((base / "results/chunks-v5-20261006_8cpu_001.json").read_text())
    raw_cases = [
        json.loads(line) for line in (base / "ground-truth-v1.jsonl").read_text().splitlines()
    ]
    proposals = {q["id"]: q for q in json.loads((base / "query-proposals-v1.json").read_text())}
    require(len(events) == 611 and len(chunks) == 2080 and len(raw_cases) == 120, "corpus_size")
    require(result["build_id"] == "20261006_8cpu_001", "build_identity")
    require(result["source_snapshot_hash"] == manifest.source_snapshot_hash, "snapshot_identity")
    require(result["query_set_hash"] == PINS["benchmark/ground-truth-v1.jsonl"], "query_identity")
    require(
        result["model"]["model_revision"] == "dd76d535f5447ca3897a9c893fb1e612ead98192", "revision"
    )
    require(digest(result["manifest"]) == result["manifest_digest"], "manifest_digest")
    docs = documents(events)
    require(
        {eid: digest(doc) for eid, doc in docs.items()} == result["documents"], "document_texts"
    )
    ordered, point_ids = [], []
    for eid, doc in docs.items():
        mapping = result["chunk_mapping"][eid]
        require(mapping["document_hash"] == events[UUID(eid)].document_hash, "document_hash")
        for index, reference in enumerate(mapping["chunks"]):
            pid = reference["point_id"]
            payload = chunks[pid]
            chunk = Chunk.model_validate(payload["chunk"])
            validate_chunk(doc, chunk, index)
            require(
                pid == str(uuid5(NAMESPACE, "v5:" + eid + ":" + digest(payload["chunk"]))),
                "point_id",
            )
            require(
                payload["entity_id"] == eid
                and payload["document_hash"] == mapping["document_hash"],
                "payload",
            )
            require(
                payload["embedding_version"] == result["model"]["embedding_version"],
                "embedding_contract",
            )
            require(
                reference
                == {
                    "point_id": pid,
                    "content_hash": chunk.content_hash,
                    "tokens": chunk.token_count,
                    "characters": len(chunk.text),
                },
                "chunk_mapping",
            )
            point_ids.append(pid)
            ordered.append(payload)
    require(len(set(point_ids)) == 2080 and set(point_ids) == set(chunks), "chunk_coverage")
    require(digest(ordered) == result["manifest"]["corpus_hash"], "corpus_hash")
    historical = {c["case_id"]: c for c in result["cases"]}
    profiles, queries = {}, []
    for raw in raw_cases:
        # Original query metadata only; judgment fields are never consumed.
        cid, text = raw["id"], raw["query"]
        eligibility = Eligibility.model_validate(raw.get("eligibility", {}))
        require(text == proposals[cid]["query"], "query_text")
        eligible_ids = sorted(
            str(eid) for eid, row in events.items() if eligibility.accepts(row.event)
        )
        occurrences = sorted(
            str(o.id)
            for row in events.values()
            for o in row.event.occurrences
            if eligibility.accepts(row.event.model_copy(update={"occurrences": [o]}))
        )
        old = historical[cid]
        require(
            eligible_ids == old["eligible_ids"] and occurrences == old["eligible_occurrence_ids"],
            "eligibility",
        )
        require(digest(eligible_ids) == old["eligibility_hash"], "eligibility_hash")
        profile = {"eligible_ids": eligible_ids, "eligible_occurrence_ids": occurrences}
        key = digest(profile)
        allowed = [
            pid
            for pid in point_ids
            if chunks[pid]["entity_id"] in eligible_ids
            and any(
                c["scope"] == "event" or c["occurrence_id"] in occurrences
                for c in chunks[pid]["chunk"]["contexts"]
            )
        ]
        require(len(allowed) == old["candidate_chunks"], "candidate_count")
        require(
            {chunks[pid]["entity_id"] for pid in allowed}
            == {h["event_id"] for h in old["ranking"]},
            "event_coverage",
        )
        profile |= {"candidate_count": len(allowed), "eligible_point_ids_sha256": digest(allowed)}
        profiles[key] = profile
        queries.append(
            {
                "id": cid,
                "text": text,
                "eligibility": eligibility.model_dump(mode="json"),
                "profile": key,
            }
        )
    require(len({q["id"] for q in queries}) == 120, "query_coverage")
    output = {
        "schema": "jina-v5-full-corpus-inputs-v1",
        "benchmark_build": result["build_id"],
        "source_sha256": PINS,
        "research_commit": RESEARCH_COMMIT,
        "research_source_sha256": source_hashes,
        "snapshot_hash": manifest.source_snapshot_hash,
        "corpus_hash": result["manifest"]["corpus_hash"],
        "model": result["model"],
        "documents": docs,
        "point_ids": point_ids,
        "chunks": chunks,
        "queries": queries,
        "eligibility_profiles": profiles,
        "judgments_used": False,
        "phase2d_used": False,
    }
    with a.output.open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "events": len(docs),
                "chunks": len(chunks),
                "queries": len(queries),
                "profiles": len(profiles),
                "query_chunk_comparisons": sum(
                    profiles[q["profile"]]["candidate_count"] for q in queries
                ),
                "input_sha256": sha(a.output),
            }
        )
    )


if __name__ == "__main__":
    main()
