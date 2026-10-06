#!/usr/bin/env python3
"""Real in-process HTTP contract smoke; run in an ONNX-only environment."""

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from uranus_research_encoder.app import create_app
from uranus_research_encoder.config import Settings
from uranus_research_encoder.v5_onnx_artifacts import sha256
from uranus_research_encoder.version import EMBEDDING_VERSION, MODEL_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--plan-dir", type=Path, default=Path("validation/jina-v5-merged-v1"))
    p.add_argument("--reference-vectors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("contract_report_already_exists")
    plan = json.loads((a.plan_dir / "plan.json").read_text())
    if sha256(a.plan_dir / "selection.json") != plan["selection_sha256"]:
        raise ValueError("selection_changed")
    rows = json.loads((a.plan_dir / "selection.json").read_text())["inputs"]
    reference = np.load(a.reference_vectors)["single"].astype("float64")
    key = secrets.token_urlsafe(40)
    settings = Settings(
        api_key=key,
        jina_noncommercial=True,
        backend="v5-onnx-merged",
        merged_onnx_root=a.graph_dir.resolve(),
        v5_onnx_manifest_sha256=sha256(a.graph_dir / "manifest.json"),
        onnx_intra_op_threads=8,
        onnx_inter_op_threads=1,
    )
    auth = {"Authorization": "Bearer " + key}
    checks = []
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready", headers=auth)
        assert ready.status_code == 200, ready.text
        version = client.get("/version", headers=auth).json()
        assert version["backend"] == "v5-onnx-merged"
        assert version["runtime"].startswith("onnxruntime-")
        assert version["model_revision"] == MODEL_REVISION
        assert version["embedding_version"] == EMBEDDING_VERSION
        assert ready.json()["backend"] == version["backend"]
        for kind in ("query", "passage"):
            ids = [i for i, r in enumerate(rows) if r["kind"] == kind][:3]
            response = client.post(
                "/embed",
                headers=auth,
                json={"model": "jina-v5", "kind": kind, "texts": [rows[i]["text"] for i in ids]},
            )
            assert response.status_code == 200, response.text
            body = response.json()
            actual = np.asarray(body["vectors"], dtype="float64")
            assert actual.shape == (3, 1024)
            assert body["embedding_version"] == EMBEDDING_VERSION
            assert body["metrics"]["text_count"] == 3
            assert body["metrics"]["token_count"] == sum(
                np.load(a.reference_vectors)["tokens"][ids]
            )
            for i, vector in zip(ids, actual, strict=True):
                diff = np.abs(vector - reference[i])
                norm = float(np.linalg.norm(vector))
                cosine = float(vector @ reference[i] / (norm * np.linalg.norm(reference[i])))
                passed = bool(
                    diff.max() <= plan["tolerances"]["max_absolute_difference"]
                    and diff.mean() <= plan["tolerances"]["mean_absolute_difference"]
                    and cosine >= plan["tolerances"]["min_cosine_similarity"]
                    and abs(norm - 1) <= plan["tolerances"]["norm_absolute_error"]
                )
                checks.append(
                    {
                        "id": rows[i]["id"],
                        "kind": kind,
                        "max_difference": float(diff.max()),
                        "mean_difference": float(diff.mean()),
                        "cosine": cosine,
                        "norm": norm,
                        "pass": passed,
                    }
                )
        assert (
            client.post(
                "/embed",
                headers=auth,
                json={"model": "jina-v5", "kind": "document", "texts": ["test"]},
            ).status_code
            == 422
        )
    report = {
        "schema": "jina-v5-onnx-http-contract-v1",
        "manifest_sha256": sha256(a.graph_dir / "manifest.json"),
        "reference_vectors_sha256": sha256(a.reference_vectors),
        "plan_sha256": sha256(a.plan_dir / "plan.json"),
        "version": version,
        "ready": ready.json(),
        "checks": checks,
        "torch_imported": "torch" in sys.modules,
        "peft_imported": "peft" in sys.modules,
        "pass": all(c["pass"] for c in checks)
        and "torch" not in sys.modules
        and "peft" not in sys.modules,
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print("HTTP contract PASS" if report["pass"] else "HTTP contract FAIL", flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
