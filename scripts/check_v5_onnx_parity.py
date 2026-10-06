#!/usr/bin/env python3
"""Execute frozen FP32 gates in separate native/ORT processes, no service access."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def compare(a, b, tolerance):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float(a @ b / (an * bn))
    result = {
        "vector_dimension": len(a),
        "native_norm": an,
        "onnx_norm": bn,
        "max_absolute_difference": float(diff.max()),
        "mean_absolute_difference": float(diff.mean()),
        "cosine_similarity": cosine,
    }
    result["pass"] = bool(
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and len(a) == 1024
        and diff.max() <= tolerance["max_absolute_difference"]
        and diff.mean() <= tolerance["mean_absolute_difference"]
        and cosine >= tolerance["min_cosine_similarity"]
        and abs(an - 1) <= tolerance["norm_absolute_error"]
        and abs(bn - 1) <= tolerance["norm_absolute_error"]
    )
    return result


def aggregate(scores, docs):
    winners = {}
    for score, d in zip(scores, docs, strict=True):
        hit = {"score": float(score), "point_id": d["id"], "event_id": d["event_id"]}
        old = winners.get(hit["event_id"])
        if old is None or (-hit["score"], hit["point_id"]) < (-old["score"], old["point_id"]):
            winners[hit["event_id"]] = hit
    return sorted(winners.values(), key=lambda h: (-h["score"], h["event_id"]))


def evaluate(a):
    plan = json.loads((a.plan_dir / "plan.json").read_text())
    selected = json.loads((a.plan_dir / "selection.json").read_text())
    if sha(a.plan_dir / "selection.json") != plan["selection_sha256"]:
        raise ValueError("selection_changed")
    rows = selected["inputs"]
    tolerance = plan["tolerances"]
    if a.mode == "report":
        native, ort = [np.load(a.output_dir / f"{m}.npz") for m in ["native", "onnx"]]
        vectors = [
            {
                **{k: r[k] for k in ["id", "kind"]},
                "token_count": int(native["tokens"][i]),
                **compare(native["single"][i], ort["single"][i], tolerance),
            }
            for i, r in enumerate(rows)
        ]
        batches = []
        for mode, data in [("native", native), ("onnx", ort)]:
            for side in plan["batch"]["padding_sides"]:
                for i, r in enumerate(rows):
                    batches.append(
                        {
                            "id": r["id"],
                            "runtime": mode,
                            "padding_side": side,
                            **compare(data["single"][i], data[side][i], tolerance),
                        }
                    )
        for side in plan["batch"]["padding_sides"]:
            for i, r in enumerate(rows):
                batches.append(
                    {
                        "id": r["id"],
                        "runtime": "native_vs_onnx",
                        "padding_side": side,
                        **compare(native[side][i], ort[side][i], tolerance),
                    }
                )
        di = [i for i, r in enumerate(rows) if r["kind"] == "passage"]
        docs = [rows[i] for i in di]
        rankings = []
        for i, q in enumerate(rows):
            if q["kind"] != "query":
                continue
            # Identical FP64 dot products of the two FP32 normalized embeddings.
            ns = native["single"][di].astype("float64") @ native["single"][i].astype("float64")
            oscores = ort["single"][di].astype("float64") @ ort["single"][i].astype("float64")
            nr, onr = aggregate(ns, docs), aggregate(oscores, docs)
            nids, oids = [[h["event_id"] for h in r] for r in [nr, onr]]
            shifts = [oids.index(e) - k for k, e in enumerate(nids)]
            winners = {h["event_id"]: h["point_id"] for h in onr}
            mismatches = [
                {"event_id": h["event_id"], "native": h["point_id"], "onnx": winners[h["event_id"]]}
                for h in nr
                if h["point_id"] != winners[h["event_id"]]
            ]
            score_diff = float(np.abs(ns - oscores).max())
            row = {
                "id": q["id"],
                "full_order_identical": nids == oids,
                "top10_set_identical": set(nids[:10]) == set(oids[:10]),
                "top10_order_identical": nids[:10] == oids[:10],
                "spearman": 1
                - 6 * sum(s * s for s in shifts) / (len(shifts) * (len(shifts) ** 2 - 1)),
                "max_rank_shift": max(abs(s) for s in shifts),
                "max_score_difference": score_diff,
                "chunk_winner_differences": mismatches,
                "rank_differences": [
                    {"event_id": e, "native_rank": k + 1, "onnx_rank": oids.index(e) + 1}
                    for k, e in enumerate(nids)
                    if shifts[k]
                ],
                "native_ranking": nr,
                "onnx_ranking": onr,
            }
            row["pass"] = (
                nids == oids
                and not mismatches
                and score_diff <= plan["ranking"]["max_score_difference"]
            )
            rankings.append(row)
        report = {
            "schema": "jina-v5-onnx-parity-report-v1",
            "plan_sha256": sha(a.plan_dir / "plan.json"),
            "selection_sha256": sha(a.plan_dir / "selection.json"),
            "artifact_sha256": {
                f"{m}.npz": sha(a.output_dir / f"{m}.npz") for m in ["native", "onnx"]
            },
            "scope": selected["scope"],
            "vectors": vectors,
            "batches": batches,
            "rankings": rankings,
            "summary": {
                "vector_pass": sum(r["pass"] for r in vectors),
                "vector_total": len(vectors),
                "batch_pass": sum(r["pass"] for r in batches),
                "batch_total": len(batches),
                "ranking_pass": sum(r["pass"] for r in rankings),
                "ranking_total": len(rankings),
                "top10_identical": sum(r["top10_order_identical"] for r in rankings),
            },
            "pass": all(r["pass"] for r in vectors + batches + rankings),
        }
        write(a.output_dir / "parity.json", report)
        print(report["summary"], "PASS" if report["pass"] else "FAIL", flush=True)
        if not report["pass"]:
            raise SystemExit(1)
        return

    from uranus_research_encoder.model import TorchBackend, offline_environment

    offline_environment()
    from transformers import AutoTokenizer

    texts = [TorchBackend._input(r["text"], r["kind"]) for r in rows]
    if a.mode == "native":
        import torch

        from uranus_research_encoder.model import cached_snapshot
        from uranus_research_encoder.v5_onnx_artifacts import verify_native_snapshot
        from uranus_research_encoder.v5_onnx_export import last_token_normalized

        verify_native_snapshot(cached_snapshot(a.model_root, "torch"))
        backend = TorchBackend(a.model_root)
        backend.load()
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        tokenizer = backend._tokenizer

        def forward(encoded):
            with torch.inference_mode():
                hidden = backend._model(
                    **{k: torch.tensor(v) for k, v in encoded.items()}, use_cache=False
                ).last_hidden_state
                return last_token_normalized(
                    hidden, torch.tensor(encoded["attention_mask"])
                ).numpy()
    else:
        import onnxruntime as ort

        manifest = json.loads((a.graph_dir / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            if sha(a.graph_dir / name) != digest:
                raise ValueError("artifact_hash_mismatch")
        tokenizer = AutoTokenizer.from_pretrained(
            a.graph_dir, local_files_only=True, trust_remote_code=False
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(a.graph_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        def forward(encoded):
            return session.run(
                ["text_embeds"],
                {
                    k: np.asarray(v, dtype=np.int64)
                    for k, v in encoded.items()
                    if k in ["input_ids", "attention_mask"]
                },
            )[0]

    tokenizer.padding_side = "right"
    single, tokens = [], []
    for i, (text, row) in enumerate(zip(texts, rows, strict=True)):
        encoded = tokenizer(text, return_tensors="np", truncation=False)
        tokens.append(encoded["input_ids"].shape[1])
        # Compare the unchanged public native path for the primary reference.
        value = (
            np.asarray(backend.embed([row["text"]], row["kind"]), dtype=np.float32)
            if a.mode == "native"
            else forward(encoded)
        )
        single.append(value[0])
        if i % 20 == 0:
            print(a.mode, "single", i, len(texts), flush=True)
    arrays = {"single": np.asarray(single), "tokens": np.asarray(tokens)}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.output_dir / f"{a.mode}-single.npz", **arrays)
    # Interleave shortest and longest inputs for mixed batches, including roles.
    order = sorted(range(len(texts)), key=lambda i: (tokens[i], rows[i]["id"]))
    order = [
        x
        for pair in zip(
            order[: (len(order) + 1) // 2], reversed(order[(len(order) + 1) // 2 :]), strict=False
        )
        for x in pair
    ] + ([order[len(order) // 2]] if len(order) % 2 else [])
    assert sorted(order) == list(range(len(texts)))
    for side in plan["batch"]["padding_sides"]:
        tokenizer.padding_side = side
        values = np.empty_like(arrays["single"])
        for start in range(0, len(order), plan["batch"]["size"]):
            idx = order[start : start + plan["batch"]["size"]]
            encoded = tokenizer(
                [texts[i] for i in idx], padding=True, return_tensors="np", truncation=False
            )
            values[idx] = forward(encoded)
            if start % 30 == 0:
                print(a.mode, side, start, len(texts), flush=True)
        arrays[side] = values
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.output_dir / f"{a.mode}.npz", **arrays)
    write(
        a.output_dir / f"{a.mode}-context.json",
        {
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "threads": 8,
            "inter_op_threads": 1,
            "graph_manifest_sha256": sha(a.graph_dir / "manifest.json")
            if a.mode == "onnx"
            else None,
            "plan_sha256": sha(a.plan_dir / "plan.json"),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["native", "onnx", "report"])
    p.add_argument("--model-root", type=Path, default=Path("/tmp/encoder-v5-offline-cache"))
    p.add_argument("--graph-dir", type=Path, default=Path("models/jina-v5-merged-v1"))
    p.add_argument("--plan-dir", type=Path, default=Path("validation/jina-v5-merged-v1"))
    p.add_argument("--output-dir", type=Path, default=Path("models/jina-v5-parity-v1"))
    evaluate(p.parse_args())
