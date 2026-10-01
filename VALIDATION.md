# Implementation and validation report

Implemented locally in `sndcds/uranus-research-encoder`. No production deployment,
uranus-admin modification, Qdrant modification, or planner modification was performed.
The ONNX follow-up used explicitly authorized read-only SSH access to copy the pinned
native artifacts into an isolated local validation cache. The live cache was unchanged.
No experimental pilot implementation was used. The existing AGPL-3.0 LICENSE was retained.

See [ONNX_VALIDATION.md](ONNX_VALIDATION.md) and `validation/` for the follow-up's
pinned graph audit, real-model parity, CPU measurements, and ONNX image verification.
Historical native-backend build checks below remain labeled by their original image.

## Repository structure

```text
pyproject.toml, uv.lock, .python-version
README.md, VALIDATION.md, LICENSE, .gitignore, .dockerignore
src/uranus_research_encoder/
  __init__.py, __main__.py, app.py, config.py, auth.py, contracts.py
  chunking.py, model.py, onnx_backend.py, model_artifacts.py, errors.py, version.py, schemas.py
tests/
  contract, context, chunking, authentication, API, configuration, limits
  concurrency, failure, logging, version, schema and native-loader tests
  explicitly marked real-model integration test
  fixtures/uranus_admin_chunks_v1.json
  fixtures/http_examples_fake_backend.json
contracts/
  EmbedRequest.json, EmbedResponse.json, ChunkRequest.json, ChunkResponse.json
  EvidenceContext.json, Section.json, Chunk.json
scripts/prefetch_model.py, scripts/audit_onnx.py
scripts/compare_backends.py, scripts/benchmark_backends.py
deploy/Dockerfile, deploy/compose.example.yml
.github/workflows/ci.yml
```

## Compatibility identifiers

- Service: `0.1.0`
- Contract: `uranus-research-encoder-v1`
- Model: `jina-v3`
- Repository: `jinaai/jina-embeddings-v3-hf`
- Revision: `d18862d9a48706220815554fac3ebb4dfa46fc28`
- Dimensions: `1024`
- Chunk version: `sections-480-overlap64-v2`

Exact embedding version:

```text
d18862d9a48706220815554fac3ebb4dfa46fc28:native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2
```

The reviewed admin contract revision is
`5cb7fb91a7900d443c4fa08b1901db1b5cdfa01f`. Its current contextual Section types,
chunk kinds, request serialization, response validation and version expectation were
inspected read-only. There is no runtime import of uranus-admin.

## Native implementation and filtered-cache verification (before ONNX follow-up)

| Check | Result |
| --- | --- |
| `uv sync --locked` | Installed successfully after a transient download retry |
| `uv sync --locked --offline` | Passed against the resolved local cache |
| `uv run --no-sync ruff format --check .` | Passed: 32 files already formatted |
| `uv run --no-sync ruff check .` | Passed: all checks passed |
| `uv run --no-sync pytest -q` | **151 passed, 1 integration test deselected, 10 warnings** (6.56s) |
| `pytest -q -m integration` | **1 skipped**: real model cache not configured |
| Seven generated JSON schemas against committed bytes | Passed in suite |
| Communitytreffen contextual `/chunks` | HTTP **200**, three chunks, contexts preserved |
| Small generated native Jina model + actual PEFT adapters | Offline load, adapter tensor integrity, task separation, float32 normalization, deterministic results passed |
| Incomplete/missing local model | Rejected without network fallback |
| YAML parsing and Compose loopback/read-only settings | Passed |
| Prefetch and server CLI help | Passed |
| `git diff --check` | Passed |

The ten test warnings are upstream: Starlette deprecates its httpx TestClient
backend, and PEFT warns when the intentionally separate second adapter is added in
nine generated-model cases. No real model weights were downloaded. TestClient's
thread/event-loop startup stalled inside the execution sandbox; API tests completed
outside that sandbox using only local fake/generated models.

## Filtered snapshot regression

The production cache is intentionally a filtered Hugging Face snapshot, containing
native safetensors, tokenizer/config files, and only the query/passage retrieval
adapters. `MODEL_ALLOW_PATTERNS` in `src/uranus_research_encoder/model_artifacts.py`
is the single reviewed allowlist imported by provisioning and runtime validation.
Runtime retains the pinned revision, `local_files_only=True`, disabled remote code,
the safetensors requirement, and adapter integrity checks.

The regression fixture generates small native Jina weights and both PEFT adapters
in a temporary Hub-style cache with blobs, snapshot symlinks, and
`trees/<revision>.json`. That cached tree also lists absent `.gitattributes`,
`custom_st.py`, ONNX, unrelated adapters, and other excluded artifacts. An unfiltered
offline `snapshot_download` reproduces `IncompleteSnapshotError`; the backend loads
the same cache successfully with the shared allowlist, for both single-file and
sharded safetensors. Both cases failed at `JinaBackend.load()` before the fix.
Negative cases remove required config, model, tokenizer, or adapter files and still
fail offline. Network connections are blocked in these tests; the real Hub,
Transformers, and PEFT loaders are exercised without mocking them.

No existing model cache was modified and no model artifacts were downloaded for
this fix. Preserve the entire provisioned cache directory when transferring it,
including tree metadata; excluded repository files need not be downloaded.
The three required checks above were rerun for this fix, with `UV_CACHE_DIR` set to
`/tmp/uranus-encoder-uv-cache` because the default uv cache is read-only in the
sandbox. The full suite passed outside the sandbox after TestClient startup stalled
inside it; all ten native-loader cases also passed inside the sandbox. The opt-in
real-cache integration test was not run for this fix.

## Docker verification

The normal production Dockerfile build pulled the pinned base images and reached
`uv sync --locked`, but the build container could not resolve the PyTorch download
host. This environment therefore did **not** establish a clean networked build.

An offline verification build **succeeded**, producing local image
`uranus-research-encoder:local-offline-verified`, image ID prefix `42df09c3ba91`.
A temporary context outside the repository used the final production Dockerfile with
only a dependency-cache COPY into the builder and `--offline` added to its two uv
sync commands. It used the exact committed lockfile and final application source,
with `docker build --network none`. The final runtime stage was unchanged. The cache
contains public package artifacts already downloaded for tests; no proxy values,
credentials, model weights or test fixtures were included in the image.

Automatic approval review rejected an earlier proposal to forward environment proxy
values as Docker build arguments because of potential credential exposure in build
metadata/cache. That proposal was not executed; the offline build supplied a safer
verification path without requesting broader access.

Container smoke checks also **passed** using two temporary containers with
`--network none`, UID 10001, a read-only root, `/tmp` tmpfs, all capabilities dropped,
`no-new-privileges`, 128 pids, two CPUs and 2 GiB memory. Neither container published
a host port. With no model mount, `/health` returned 200 and `/ready` returned 503.
With small generated native weights mounted read-only, `/ready`, contextual `/chunks`
and `/embed` returned 200, and vectors were finite, normalized and 1024-dimensional.
Root filesystem writes were rejected; `/tmp` writes worked; no GCC was installed.
Every captured log line parsed as JSON and contained no test key. Both containers
were stopped and removed after verification. These synthetic-weight checks do not
replace the pending real-Jina integration test.

## Security properties

- Bearer authentication before parsing protected request bodies; constant-time key
  comparison; file-based keys preferred; no automatic dotenv reads.
- All requested contexts, UUID identities and kinds validated; unknown fields/kinds
  rejected; mixed contextual/plain documents rejected; deterministic context unions.
- Exact tokenizer counts, bounded overlap and chunks; no silent truncation.
- Strict body/text/document/section/chunk/response limits; immediate admission rejection
  on saturation; bounded adapter/inference serialization and server concurrency.
- Native Transformers only, remote code disabled, pinned local snapshot and retrieval
  adapters, explicit offline loaders, no runtime provisioning or downloads.
- Explicit non-commercial license acknowledgement; startup errors leave liveness
  available and readiness failed. Readiness performs no embedding.
- Finite normalized 1024-dimensional float32 vectors; deterministic CPU computation.
- Generic bounded errors; fixed structured service events; sanitized third-party logs;
  no request text, vectors or Authorization values in logs.
- Docs opt-in, no CORS, no redirects. Version metadata exposes no deployment secrets,
  environment values, hostnames or paths.
- Non-root container, read-only-compatible root, tmpfs, no capabilities, no privileged
  or host-network mode; example host binding restricted to `127.0.0.1:6335`.

## HTTP examples

These are captured successful API-test responses using the **deterministic fake
backend**, not measurements from production Jina weights. In particular the chunk
`token_count` values below belong to the fake tokenizer. The exact JSON is also in
`tests/fixtures/http_examples_fake_backend.json`; the chunk response is regression-tested.

### /health — HTTP 200

```json
{
  "status": "ok"
}
```

### /ready — HTTP 200

```json
{
  "status": "ready",
  "backend": "fake",
  "runtime": "deterministic-test-backend",
  "model_revision": "d18862d9a48706220815554fac3ebb4dfa46fc28",
  "model": "jina-v3",
  "dimensions": 1024,
  "embedding_version": "d18862d9a48706220815554fac3ebb4dfa46fc28:native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2",
  "contract_version": "uranus-research-encoder-v1"
}
```

### /version — HTTP 200

```json
{
  "service_version": "0.1.0",
  "backend": "fake",
  "runtime": "deterministic-test-backend",
  "contract_version": "uranus-research-encoder-v1",
  "model": "jina-v3",
  "model_repository": "jinaai/jina-embeddings-v3-hf",
  "model_revision": "d18862d9a48706220815554fac3ebb4dfa46fc28",
  "dimensions": 1024,
  "embedding_version": "d18862d9a48706220815554fac3ebb4dfa46fc28:native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2",
  "chunk_version": "sections-480-overlap64-v2"
}
```

### /chunks — HTTP 200

```json
{
  "embedding_version": "d18862d9a48706220815554fac3ebb4dfa46fc28:native-transformers5.17.0-retrieval-normalized-f32:sections-480-overlap64-v2",
  "documents": [
    {
      "entity_id": "019d5f3a-de7e-780a-9fa0-dc24e5545e2e",
      "chunks": [
        {
          "chunk_index": 0,
          "chunk_kind": "content",
          "contexts": [
            {
              "scope": "event",
              "venue_id": null,
              "space_id": null,
              "occurrence_id": null
            }
          ],
          "text": "Titel: Communitytreffen...",
          "content_hash": "173cbc1aaf69c33d4474194b8a44f7564262327c46eb6ac84df6199b8de31dea",
          "token_count": 28
        },
        {
          "chunk_index": 1,
          "chunk_kind": "tickets",
          "contexts": [
            {
              "scope": "event",
              "venue_id": null,
              "space_id": null,
              "occurrence_id": null
            }
          ],
          "text": "Preisart: kostenlos...",
          "content_hash": "2f803ed201ea36cd2b3cbacebb22c4cbee6df35559448c436e80036727ff320d",
          "token_count": 24
        },
        {
          "chunk_index": 2,
          "chunk_kind": "location_context",
          "contexts": [
            {
              "scope": "venue",
              "venue_id": "019d9b4e-b7ca-7b1f-8d19-56ca89e33ce7",
              "space_id": null,
              "occurrence_id": null
            }
          ],
          "text": "Ort: Aktivitetshuset\n\nGemeinden / Kommunen: Flensburg",
          "content_hash": "f3104d3390acc6fbf28d6b513f2a1027b7cf795c6682c0e81e8e4b2ff1c8d9ad",
          "token_count": 55
        }
      ]
    }
  ]
}
```

## Known limitations and remaining rollout work

1. Provision the pinned real Jina snapshot and both retrieval adapters using the
   explicit prefetch tool; preserve the entire cache tree and verify operator-controlled
   provenance. Runtime pin checking does not cryptographically attest modified local
   bytes that have been placed under the expected snapshot name.
2. Run the marked integration test with the real cache, then verify real `/ready`,
   `/version`, contextual `/chunks`, and query/passage embeddings in a separate
   authorized staging environment. Production numerical quality is not established
   by generated-weight tests.
3. Benchmark target-machine CPU latency and peak memory, especially at 8192 input
   tokens. The Compose 8 GiB memory value is an example; eager attention for long
   inputs can require more. Inference has no hard interruption timeout. Keep one
   server worker and tune admission/resource limits using measured load.
4. Verify the standard Docker build on a builder with approved package-registry
   connectivity. Bit-identical vectors across different CPU architectures/software
   stacks are not promised; the lockfile and versions must be reviewed together.
5. Confirm non-commercial Jina authorization, provision a real high-entropy key with
   UID 10001 read access, and review the loopback/SSH-tunnel boundary before any
   separately authorized deployment. No deployment was performed here.

No static type checker is configured; linting, schema checks, unit/API tests and
native offline loader tests are the implemented CI gates.
