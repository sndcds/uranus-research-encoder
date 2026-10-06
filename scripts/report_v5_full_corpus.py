#!/usr/bin/env python3
"""Compare full frozen eligible rankings, with no judgments or relevance metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_v5_onnx_parity import aggregate, compare, sha


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def eligible_points(inputs, profile):
    events, occurrences = set(profile["eligible_ids"]), set(profile["eligible_occurrence_ids"])
    return [
        pid
        for pid in inputs["point_ids"]
        if inputs["chunks"][pid]["entity_id"] in events
        and any(
            c["scope"] == "event" or c["occurrence_id"] in occurrences
            for c in inputs["chunks"][pid]["chunk"]["contexts"]
        )
    ]


def cosine_scores(documents, query):
    # Fixed row-wise FP64 reductions preserve exact ties for identical vectors.
    # This is mathematical cosine, not a replay of Qdrant's SIMD rounding.
    d, q = np.asarray(documents, dtype=np.float64), np.asarray(query, dtype=np.float64)
    return np.sum(d * q, axis=1) / (np.sqrt(np.sum(d * d, axis=1)) * np.sqrt(np.sum(q * q)))


def differences(left, right, native_scores, onnx_scores):
    nr, ort = (
        {pid: i + 1 for i, pid in enumerate(left)},
        {pid: i + 1 for i, pid in enumerate(right)},
    )
    require(set(nr) == set(ort), "ranking_candidate_mismatch")
    return [
        {
            "id": pid,
            "native_rank": rank,
            "onnx_rank": ort[pid],
            "native_score": native_scores[pid],
            "onnx_score": onnx_scores[pid],
        }
        for pid, rank in nr.items()
        if rank != ort[pid]
    ]


def ties(order, scores, changed):
    groups = {}
    for pid in order:
        groups.setdefault(scores[pid], []).append(pid)
    return [
        {"score": score, "ordered_ids": ids}
        for score, ids in groups.items()
        if len(ids) > 1 and set(ids).intersection(changed)
    ]


def compare_rankings(points, chunks, ns, oscores):
    docs = [{"id": pid, "event_id": chunks[pid]["entity_id"]} for pid in points]
    nscore, oscore = [dict(zip(points, map(float, s), strict=True)) for s in (ns, oscores)]
    nc, oc = [
        sorted(points, key=lambda pid, scores=s: (-scores[pid], pid)) for s in (nscore, oscore)
    ]
    ne, oe = aggregate(ns, docs), aggregate(oscores, docs)
    ni, oi = [[h["event_id"] for h in ranking] for ranking in (ne, oe)]
    nh, oh = [{h["event_id"]: h for h in ranking} for ranking in (ne, oe)]
    nes, oes = [{k: v["score"] for k, v in h.items()} for h in (nh, oh)]
    chunk_diffs = differences(nc, oc, nscore, oscore)
    event_diffs = differences(ni, oi, nes, oes)
    winner_diffs = []
    for eid in ni:
        a, b = nh[eid]["point_id"], oh[eid]["point_id"]
        if a != b:
            winner_diffs.append(
                {
                    "event_id": eid,
                    "native_winner": a,
                    "onnx_winner": b,
                    "candidate_scores": [
                        {"point_id": p, "native": nscore[p], "onnx": oscore[p]} for p in (a, b)
                    ],
                    "native_exact_tie": nscore[a] == nscore[b],
                    "onnx_exact_tie": oscore[a] == oscore[b],
                    "same_text": chunks[a]["chunk"]["text"] == chunks[b]["chunk"]["text"],
                }
            )
    changed_chunks = {r["id"] for r in chunk_diffs}
    changed_events = {r["id"] for r in event_diffs}
    delta = np.abs(ns - oscores)
    max_index = int(delta.argmax()) if len(delta) else None
    return {
        "candidate_chunks": len(points),
        "candidate_events": len(ni),
        "top10_identical": ni[:10] == oi[:10],
        "top20_identical": ni[:20] == oi[:20],
        "full_ranking_identical": ni == oi,
        "chunk_ranking_identical": nc == oc,
        "chunk_winners_identical": not winner_diffs,
        "max_event_rank_difference": max(
            (abs(r["native_rank"] - r["onnx_rank"]) for r in event_diffs), default=0
        ),
        "max_chunk_rank_difference": max(
            (abs(r["native_rank"] - r["onnx_rank"]) for r in chunk_diffs), default=0
        ),
        "max_score_difference": float(delta.max()) if len(delta) else 0.0,
        "max_score_difference_point": None
        if max_index is None
        else {
            "point_id": points[max_index],
            "native": float(ns[max_index]),
            "onnx": float(oscores[max_index]),
        },
        "native_top20": ni[:20],
        "onnx_top20": oi[:20],
        "ordering_sha256": {
            "native_chunks": digest(nc),
            "onnx_chunks": digest(oc),
            "native_events": digest(ni),
            "onnx_events": digest(oi),
            "native_winners": digest({eid: h["point_id"] for eid, h in nh.items()}),
            "onnx_winners": digest({eid: h["point_id"] for eid, h in oh.items()}),
        },
        "event_rank_differences": event_diffs,
        "chunk_rank_differences": chunk_diffs,
        "winner_differences": winner_diffs,
        "exact_ties_involving_changed_ids": {
            "native_chunks": ties(nc, nscore, changed_chunks),
            "onnx_chunks": ties(oc, oscore, changed_chunks),
            "native_events": ties(ni, nes, changed_events),
            "onnx_events": ties(oi, oes, changed_events),
        },
        "exact_chunk_tie_count": {
            "native": len(nc) - len(set(nscore.values())),
            "onnx": len(oc) - len(set(oscore.values())),
        },
        "exact_event_tie_count": {
            "native": len(ni) - len(set(nes.values())),
            "onnx": len(oi) - len(set(oes.values())),
        },
    }


def evaluate(plan_dir, vector_dir, output):
    require(not output.exists(), "immutable_report_exists")
    plan_path, input_path = plan_dir / "full-corpus-plan.json", plan_dir / "full-corpus-inputs.json"
    plan, inputs = json.loads(plan_path.read_text()), json.loads(input_path.read_text())
    require(sha(input_path) == plan["inputs_sha256"], "input_hash")
    parent = json.loads((plan_dir / "plan.json").read_text())
    require(sha(plan_dir / "plan.json") == plan["parent_plan_sha256"], "parent_plan_changed")
    require(
        parent["tolerances"] == plan["tolerances"]
        and parent["ranking"]["max_score_difference"] == plan["max_score_difference"],
        "threshold_changed",
    )
    points, queries = inputs["point_ids"], inputs["queries"]
    require(
        len(points) == 2080
        and len(set(points)) == 2080
        and len(queries) == 120
        and len(inputs["documents"]) == 611,
        "incomplete_inputs",
    )
    arrays, contexts = {}, {}
    for backend in ("native", "onnx"):
        path = vector_dir / f"full-corpus-{backend}.npz"
        ctx = json.loads((vector_dir / f"full-corpus-{backend}-context.json").read_text())
        require(
            ctx["plan_sha256"] == sha(plan_path)
            and ctx["inputs_sha256"] == sha(input_path)
            and ctx["manifest_sha256"] == plan["manifest_sha256"],
            "context_identity",
        )
        require(
            ctx["encoder_source_sha256"] == plan["encoder_source_sha256"]
            and ctx["versions"] == plan["versions"],
            "execution_source_changed",
        )
        require(ctx["vectors_sha256"] == sha(path) and ctx["backend"] == backend, "vector_identity")
        require(
            ctx["threads"] == 8 and ctx["inter_op_threads"] == 1 and len(ctx["cpu_affinity"]) == 8,
            "execution_profile",
        )
        with np.load(path, allow_pickle=False) as archive:
            data = {
                key: archive[key]
                for key in ("documents", "queries", "document_tokens", "query_tokens")
            }
        for name, count in (("documents", 2080), ("queries", 120)):
            require(
                data[name].shape == (count, 1024)
                and data[name].dtype == np.float32
                and np.isfinite(data[name]).all(),
                "incomplete_vectors",
            )
        require(
            data["document_tokens"].tolist()
            == [inputs["chunks"][pid]["chunk"]["token_count"] for pid in points],
            "chunk_tokens",
        )
        require(data["query_tokens"].shape == (120,), "query_tokens")
        arrays[backend], contexts[backend] = data, ctx
    n, o = arrays["native"], arrays["onnx"]
    require(np.array_equal(n["query_tokens"], o["query_tokens"]), "query_tokenization")
    vectors = []
    for name, rows in (("documents", points), ("queries", [q["id"] for q in queries])):
        for i, identifier in enumerate(rows):
            vectors.append(
                {
                    "id": identifier,
                    "kind": "passage" if name == "documents" else "query",
                    "token_count": int(
                        n["document_tokens" if name == "documents" else "query_tokens"][i]
                    ),
                    **compare(n[name][i], o[name][i], plan["tolerances"]),
                }
            )
    index = {pid: i for i, pid in enumerate(points)}
    rankings, nscores, oscores, masks = [], [], [], []
    for qi, query in enumerate(queries):
        profile = inputs["eligibility_profiles"][query["profile"]]
        allowed = eligible_points(inputs, profile)
        require(
            len(allowed) == profile["candidate_count"]
            and digest(allowed) == profile["eligible_point_ids_sha256"],
            "candidate_contract",
        )
        ns = cosine_scores(n["documents"], n["queries"][qi])
        ort = cosine_scores(o["documents"], o["queries"][qi])
        allowed_indices = [index[pid] for pid in allowed]
        mask = np.zeros(len(points), dtype=np.bool_)
        mask[allowed_indices] = True
        nscores.append(ns)
        oscores.append(ort)
        masks.append(mask)
        row = {
            "id": query["id"],
            "eligibility_profile": query["profile"],
            **compare_rankings(
                allowed, inputs["chunks"], ns[allowed_indices], ort[allowed_indices]
            ),
        }
        row["pass"] = (
            all(
                row[k]
                for k in (
                    "top10_identical",
                    "top20_identical",
                    "full_ranking_identical",
                    "chunk_ranking_identical",
                    "chunk_winners_identical",
                )
            )
            and row["max_score_difference"] <= plan["max_score_difference"]
        )
        rankings.append(row)
    score_path = vector_dir / "full-corpus-scores.npz"
    require(not score_path.exists(), "immutable_score_artifact_exists")
    np.savez_compressed(
        score_path,
        native=nscores,
        onnx=oscores,
        eligible=masks,
        point_ids=np.asarray(points),
        query_ids=np.asarray([q["id"] for q in queries]),
    )
    summary = {
        "queries_checked": len(rankings),
        "events_checked": len(inputs["documents"]),
        "chunks_checked": len(points),
        "query_chunk_comparisons": sum(r["candidate_chunks"] for r in rankings),
        "unfiltered_scores_per_runtime": len(points) * len(queries),
        "query_event_comparisons": sum(r["candidate_events"] for r in rankings),
        "vector_checks": len(vectors),
        "vector_pass": sum(r["pass"] for r in vectors),
        **{
            key + "_count": sum(r[key] for r in rankings)
            for key in (
                "top10_identical",
                "top20_identical",
                "full_ranking_identical",
                "chunk_ranking_identical",
                "chunk_winners_identical",
            )
        },
        "max_event_rank_difference": max(r["max_event_rank_difference"] for r in rankings),
        "max_chunk_rank_difference": max(r["max_chunk_rank_difference"] for r in rankings),
        "max_score_difference": max(r["max_score_difference"] for r in rankings),
        "max_vector_component_difference": max(r["max_absolute_difference"] for r in vectors),
        "failed_queries": [r["id"] for r in rankings if not r["pass"]],
    }
    require(
        summary["query_chunk_comparisons"]
        == plan["required_counts"]["eligible_query_chunk_comparisons"],
        "incomplete_comparison",
    )
    report = {
        "schema": "jina-v5-full-corpus-parity-v1",
        "benchmark_build": inputs["benchmark_build"],
        "plan_sha256": sha(plan_path),
        "inputs_sha256": sha(input_path),
        "manifest_sha256": plan["manifest_sha256"],
        "reporter_sha256": sha(Path(__file__)),
        "numpy_version": np.__version__,
        "scoring": plan["scoring"],
        "chunk_tie_break": plan["chunk_tie_break"],
        "event_tie_break": plan["event_tie_break"],
        "winner_tie_break": plan["winner_tie_break"],
        "artifact_sha256": {f.name: sha(f) for f in vector_dir.glob("full-corpus-*.npz")},
        "execution": contexts,
        "summary": summary,
        "vectors": vectors,
        "queries": rankings,
        "pass": all(r["pass"] for r in vectors) and all(r["pass"] for r in rankings),
    }
    with output.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"pass": report["pass"], **summary}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--vector-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = evaluate(a.plan_dir, a.vector_dir, a.output)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
