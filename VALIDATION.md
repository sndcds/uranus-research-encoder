# Implementation and validation report

## Jina v5 migration — 2026-10-05

This section describes v5 only. All earlier reports below and the ONNX reports are
historical v3 evidence and remain unchanged. The migration does not deploy a service,
modify consumers, delete model caches, or reindex any production data.

### Code and compatibility review

Baseline commit: `48ce71550d5dad8c697facd3767aef8b4be97cc1`.
Service 0.2.0; unchanged JSON structure/contract `uranus-research-encoder-v1`;
model `jina-v5`, repository `jinaai/jina-embeddings-v5-text-small`, revision
`dd76d535f5447ca3897a9c893fb1e612ead98192`. See `version.py` for the full frozen
embedding identifier, including native Qwen3, Torch/Transformers/PEFT, CPU eager,
retrieval prefixes, last-token pooling, L2 float32, 1024 dimensions and chunk version.

The official pinned config subclass and encoding wrapper were reviewed as text,
not executed. Native Qwen3 + Transformers' PEFT integration supplies the single
retrieval adapter without `trust_remote_code=True`. Query inputs use `Query: `;
passage inputs use `Document: `. Last-token pooling is followed by exactly one
float32 L2 normalization. The actual model context is 32,768 tokens, including the
prefix and special tokens. No tokenizer truncation is enabled.

`chunking.py` is byte-identical to the baseline. The 480/64 algorithm, contexts,
ordering, hashes and `sections-480-overlap64-v2` remain unchanged. Counts necessarily
use the new tokenizer and the passage prefix for chunks; therefore token boundaries
can differ. Prefixes are not inserted in returned chunk texts/content hashes.
Dependencies retain their original pins; only the project version changes in uv.lock.

### Local tests (no real model)

- Baseline: **204 passed, 3 deselected** in 30.10 s.
- Final repository suite: **218 passed, 2 deselected** in 6.22 s.
- Focused native-loader/provisioning/version suite: **37 passed** in 9.85 s.
- Ruff formatting/lint, `uv lock --check --offline` and `uv sync --locked --offline`: passed.
- One existing Starlette/httpx deprecation warning remains; no unrelated dependency upgrade was made.
- The sandbox blocked the ASGI TestClient event loop; full API tests were executed
  outside that sandbox. Synthetic loader tests still prohibit socket connections.
- Existing auth, admission/body/response limits, concurrency, logging/error,
  schema-byte comparisons and chunking regressions pass. Synthetic models use
  minimal native Qwen3 weights and one retrieval LoRA, not downloaded weights.
- HTTP fixtures explicitly remain **fake-backend examples**, not model-quality data.

The retired v3 ONNX loader/export tests are replaced by explicit fail-closed tests:
configuration, load, count, embeddings, manifests, provisioning and export reject
unsupported v5 use before artifact IO. Existing generic merge arithmetic and
historical parity-gate arithmetic tests remain; neither establishes v5 ONNX parity.

### Real model and offline provisioning

`ENCODER_INTEGRATION_MODEL_ROOT=/tmp/encoder-v5-offline-cache JINA_NONCOMMERCIAL=1
uv run --no-sync pytest -q -m integration` passed: **1 passed, 1 skipped,
218 deselected** in 6.77 s. The skip is explicitly unsupported v5 ONNX parity.
The real model test blocks `socket.socket.connect` and verifies the pinned revision,
all parameters float32, exactly one retrieval adapter, query/passage distinction,
repeated and request-batch invariance, finite unit vectors of length 1024, and
real-tokenizer chunk budgets. This is functional acceptance, not a quality or
production performance benchmark.

The isolated cache contains exactly six files: config, tokenizer JSON/config,
native safetensors and retrieval adapter config/weights. No remote Python,
other adapters, vocabulary/merge fallback files or alternate weights are present.
[Artifact sizes and hashes](validation/jina-v5-artifacts.json) document the actual
files used. The main model SHA256 matches the pinned Hub LFS digest.

The initial public prefetch hit Xet/HTTPS DNS problems. Config/tokenizer/adapter
artifacts came from the official pinned Hub repository. The main public weights
were copied read-only from the previously authorized AI-host cache at the same
revision and verified against the Hub digest. The final canonical prefetch then
completed successfully against the populated separate cache, fetching six entries.
No remote code, server configuration, production service, or old cache was changed.
The model cache is temporary and is not committed.

### Docker and real HTTP contract

The final image was built locally with:

```sh
docker build --network=host -f deploy/Dockerfile -t uranus-research-encoder:v5-validation .
```

Image: `sha256:f395740ba8179b229fabb5c2f3e05bb09ae22c9386a9700d8c5c1d3c0799b49c`.
The first build failed on Docker build-network DNS; host networking was needed
only during the build. Runtime smoke containers used **`--network none`**, read-only
root/cache/key mounts, UID/GID 10001, dropped capabilities, no-new-privileges,
6 GiB memory, two CPUs, 128 pids and a bounded `/tmp` tmpfs. No host port was published.

With an empty cache: `/health` 200, authenticated `/ready` 503 with generic
`not_ready`, and `/version` 200 with v5 metadata. With the real six-file cache:
`/health` 200, unauthenticated `/ready` 401, authenticated `/ready` and `/version`
200; query and passage embeddings were normalized, finite, 1024-dimensional,
distinct and repeatable. The long accessibility document produced three stable
chunks within 480 tokens, with correct content hashes. The previous model request
literal was rejected with 422. All test containers were stopped and removed.

[Actual HTTP results](validation/jina-v5-http-smoke.json) and the
[executed smoke probe](validation/jina-v5-http-smoke.py) are stored repository artifacts.
The probe also asserts the actual UID, read-only root flag and loopback-only
interfaces. Run it only inside an equivalently isolated test container:

```sh
docker exec -i encoder-v5-validation-real python - < validation/jina-v5-http-smoke.py
```

### Reproduction commands and remaining work

Executed from the encoder checkout (the model cache path is local test configuration):

```sh
uv sync --locked --offline
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -q
ENCODER_INTEGRATION_MODEL_ROOT=/tmp/encoder-v5-offline-cache JINA_NONCOMMERCIAL=1 \
  uv run --no-sync pytest -q -m integration
JINA_NONCOMMERCIAL=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 \
  uv run --no-sync python scripts/prefetch_model.py --model-root /tmp/encoder-v5-native-cache
```

No production rollout, CPU/RAM capacity benchmark, v5 ONNX export/parity, consumer
migration or external Qdrant reindex was performed. These remain separate work.
There are no v5 retrieval-quality or performance improvement claims. Docker and
real-model functional validation are complete; neither proves 32K eager-attention
requests fit the example memory limit.

### Changed files and purpose

| File | Purpose |
| --- | --- |
| `MERGED_ONNX_VALIDATION.md` | Label previous merged-ONNX results as historical v3 evidence. |
| `ONNX_VALIDATION.md` | Label previous upstream-ONNX results as historical v3 evidence. |
| `README.md` | Document native v5 semantics, offline artifacts, unchanged chunking, setup, limits and license. |
| `VALIDATION.md` | Record actual v5 checks separately from unchanged historical results. |
| `contracts/ChunkRequest.json` | Regenerate the request model literal as jina-v5. |
| `contracts/EmbedRequest.json` | Regenerate the request model literal as jina-v5. |
| `deploy/Dockerfile` | Ship only the validated Torch target; retain non-root/offline/read-only support. |
| `deploy/compose.example.yml` | Use the v5 model identifier in the existing example. |
| `pyproject.toml` | Bump service/package to 0.2.0 without changing dependency pins. |
| `scripts/audit_onnx.py` | Reject unavailable v5 graph audits explicitly. |
| `scripts/benchmark_backends.py` | Default to Torch and reject unsupported ONNX runs. |
| `scripts/compare_backends.py` | Disable v5 ONNX parity execution; preserve historical gate arithmetic. |
| `scripts/export_merged_onnx.py` | Reject the incompatible exporter before any file IO or overwrite. |
| `scripts/prefetch_model.py` | Provision only the canonical pinned native v5 manifest. |
| `src/uranus_research_encoder/app.py` | Count the requested embedding kind through the backend abstraction. |
| `src/uranus_research_encoder/config.py` | Accept v5 identity and reject unsupported ONNX configuration. |
| `src/uranus_research_encoder/contracts.py` | Change the model literal while preserving the JSON structure. |
| `src/uranus_research_encoder/merged_artifacts.py` | Reject unvalidated ONNX manifests; retain the generic digest helper. |
| `src/uranus_research_encoder/merged_onnx_backend.py` | Preserve class imports but fail closed for every v5 operation. |
| `src/uranus_research_encoder/model.py` | Implement validated native Qwen3 loading, one LoRA, prefixes, exact counts and last-token pooling. |
| `src/uranus_research_encoder/model_artifacts.py` | Share the minimal non-executable v5 artifact allowlist. |
| `src/uranus_research_encoder/onnx_backend.py` | Preserve class imports but fail closed for every v5 operation. |
| `src/uranus_research_encoder/version.py` | Pin v5 revision and explicitly identify the new embedding space/runtime. |
| `tests/conftest.py` | Keep deterministic fake vectors; update context limit and optional count kind. |
| `tests/fixtures/http_examples_fake_backend.json` | Update explicitly synthetic HTTP metadata examples. |
| `tests/fixtures/uranus_admin_chunks_v1.json` | Update the synthetic client request model literal. |
| `tests/test_benchmark_runner.py` | Retain worker failure handling tests against the supported Torch mode. |
| `tests/test_concurrency.py` | Retain concurrency regression coverage using the v5 request contract. |
| `tests/test_config.py` | Check v5 defaults and rejection of unsupported backends. |
| `tests/test_contracts.py` | Retain strict request validation using the v5 literal. |
| `tests/test_embed.py` | Check v5 requests, kind-aware counts and rejection of the previous model. |
| `tests/test_failures.py` | Retain readiness, generic error and invalid-vector regression coverage. |
| `tests/test_health.py` | Assert v5 readiness metadata while retaining security/liveness tests. |
| `tests/test_integration.py` | Exercise actual pinned weights offline, all-float32 parameters and retrieval semantics. |
| `tests/test_limits.py` | Retain HTTP limits and test the actual 32K context boundary contract. |
| `tests/test_logging.py` | Retain sanitized logging tests with v5 requests. |
| `tests/test_merged_onnx.py` | Retain model-independent merge arithmetic; retire invalid v3 graph acceptance tests. |
| `tests/test_model.py` | Generate tiny Qwen3/LoRA fixtures; test loading, completeness, prefixes, pooling, counts and vectors. |
| `tests/test_onnx.py` | Test fail-closed load/count/embed, readiness, manifests, provisioning and non-destructive export rejection. |
| `tests/test_onnx_parity.py` | Explicitly skip real parity until v5 ONNX is supported. |
| `tests/test_parity_gate.py` | Identify retained v3 report tests as historical arithmetic checks. |
| `tests/test_prefetch.py` | Test canonical revision/manifest and rejection of unsupported provisioning modes. |
| `tests/test_regression.py` | Retain chunk/client regressions and compare committed HTTP metadata examples. |
| `tests/test_version.py` | Freeze v5 identity/embedding version and package/service consistency. |
| `uv.lock` | Update the project version only. |
| `validation/jina-v5-artifacts.json` | Record hashes and sizes of the actual six real-model artifacts. |
| `validation/jina-v5-http-smoke.json` | Record actual isolated-container HTTP acceptance results. |
| `validation/jina-v5-http-smoke.py` | Reproduce the real HTTP/isolation checks without exposing the mounted test key. |

---

## Historical Jina v3 report (unchanged)


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
