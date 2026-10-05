# Uranus Research Encoder

A standalone, stateless internal HTTP service for the Kulturbytes / Uranus research
stack. It turns generic Sections into deterministic contextual chunks and produces
normalized 1024-dimensional Jina retrieval embeddings. It has no database, queue,
external inference API, or runtime model download.

See [VALIDATION.md](VALIDATION.md) for executed checks, Docker verification, captured
HTTP examples, and remaining real-model rollout checks.

```text
uranus-admin
    |
    | HTTP over loopback / SSH tunnel
    v
uranus-research-encoder :6335
    |
    v
local Jina model files (read-only)
```

The encoder does not know about PostgreSQL, Qdrant, the research planner, or domain
entities beyond `entity_id`, `Section`, and `EvidenceContext`. This boundary is
intentional. This repository does not install or deploy anything onto existing hosts.

## Versions and model

| Identifier | Value |
| --- | --- |
| Service | `0.2.0` |
| HTTP contract | `uranus-research-encoder-v1` |
| Logical model | `jina-v5` |
| Repository | `jinaai/jina-embeddings-v5-text-small` |
| Weights revision | `dd76d535f5447ca3897a9c893fb1e612ead98192` |
| Dimensions | `1024` |
| Chunk version | `sections-480-overlap64-v2` |
| Model license | `CC-BY-NC-4.0` |

The exact, centrally defined and tested embedding version is:

```text
dd76d535f5447ca3897a9c893fb1e612ead98192:native-qwen3-torch2.11.0-transformers5.17.0-peft0.21.1-cpu-eager-retrieval-query-document-last-token-l2-f32-d1024:sections-480-overlap64-v2
```

`GET /version` is the machine-readable deployment contract. Clients should compare
its contract and embedding versions before indexing. Request, embedding, or chunk
schema changes require explicit version and client compatibility review. Never update
the model, Transformers version, pooling, adapters or chunk rules while retaining an
old embedding version. Reindexing requirements must be assessed when these change.

## Architecture

- `contracts.py` defines strict Pydantic v2 request/response types and hard limits.
- `chunking.py` implements the reviewed Section v2 semantics using exact counts.
- `model.py` defines the backend protocol and the native `TorchBackend` reference.
- `onnx_backend.py` and `merged_onnx_backend.py` retain imports but reject v5 use.
- `model_artifacts.py` defines their shared canonical artifact manifests.
- `auth.py` authenticates before reading/parsing protected request bodies and applies
  streaming size, body-read timeout, and admission limits.
- `app.py` composes the HTTP service and accepts a backend for tests.
- `version.py` is the central source of compatibility metadata.
- `schemas.py` generates the committed reviewable schemas in `contracts/`.
- `scripts/prefetch_model.py` is an explicit network-enabled provisioning tool.
- `deploy/` contains a production image definition and portable example Compose file.

The selected 677M model uses **Qwen3**, a 1024-dimensional hidden state and a
**32,768-token** model context. The tokenizer advertises a larger length; the loader
uses the reviewed model configuration limit instead. Existing locked dependencies
are retained: Transformers 5.17.0, Torch 2.11.0 and PEFT 0.21.1. The official minimums
are Transformers 4.57.0, Torch 2.8.0 and PEFT 0.15.2.

The [pinned model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-small/blob/dd76d535f5447ca3897a9c893fb1e612ead98192/README.md),
[configuration](https://huggingface.co/jinaai/jina-embeddings-v5-text-small/blob/dd76d535f5447ca3897a9c893fb1e612ead98192/configuration_jina_embeddings_v5.py)
and [encoding wrapper](https://huggingface.co/jinaai/jina-embeddings-v5-text-small/blob/dd76d535f5447ca3897a9c893fb1e612ead98192/modeling_jina_embeddings_v5.py)
were reviewed as source text. The upstream configuration only subclasses Qwen3Config
with a custom model type. The service validates that upstream identity, constructs a
native `Qwen3Config`, and loads an explicit `Qwen3Model`; it **never executes the
repository's Python code**. `AutoTokenizer` receives that native configuration with
`trust_remote_code=False`. The upstream generic AutoModel example requires remote
code, but our equivalent native path does not.

Both external kinds use the **same retrieval LoRA adapter** from `adapters/retrieval`:
`query` prepends `Query: `; `passage` prepends `Document: `. Clients supply raw text,
not these internal prefixes. The backend pools the last valid token and applies
exactly one L2 normalization in float32. No mean pooling, dimension reduction,
secondary normalization, alternative adapter or fallback model is used. A finite,
nonzero unit vector of exactly 1024 dimensions is checked before returning it.

The Torch reference uses CPU float32, eager attention, deterministic Torch operations,
one Torch compute thread, and one text per forward pass. A lock covers adapter selection
and inference together. Identical inputs yield identical results on the same locked
software and hardware; cross-platform floating-point bit identity is not promised.
GPU operation is intentionally not configured. Long input inference is expensive;
production throughput and memory need measurement on the intended machine.

## API and authentication

Except for `GET /health`, requests require `Authorization: Bearer <key>`.
Keys must contain 32–4096 printable ASCII characters; use a high-entropy random key.
The secret file takes precedence over the environment variable. One final newline is
allowed in the file. Comparison uses `hmac.compare_digest`. Never put keys in URLs,
command-line arguments, repository files, access logs, or issue reports.

| Method/path | Behavior |
| --- | --- |
| `GET /health` | Exactly `{"status":"ok"}`; no model access or inference |
| `GET /ready` | Configuration, loaded model, tokenizer, pinned revision; 503 if unavailable |
| `GET /version` | Backend/runtime, service, contract, repository, revision, dimensions, embedding and chunk versions |
| `POST /embed` | `{"model":"jina-v5","texts":["..."],"kind":"query"}`; also accepts `passage` |
| `POST /chunks` | Generic documents with UUID `entity_id` and `sections` |

`/embed` returns `embedding_version`, `vectors`, and text/token **counts** in `metrics`.
Every vector must have 1024 finite numbers and unit norm. Text is never silently
truncated: requests above the model's 32,768-token window, including the retrieval prefix and special tokens,
fail validation. JSON represents the numeric values of float32 outputs.

`/chunks` returns `embedding_version` and ordered `documents`, each with `entity_id`
and ordered `chunks`. Each chunk has a contiguous zero-based `chunk_index`,
`chunk_kind`, `contexts`, `text`, SHA256 `content_hash`, and exact `token_count`.
The [Communitytreffen request fixture](tests/fixtures/uranus_admin_chunks_v1.json)
exercises the contextual client contract.

Errors are fixed, bounded JSON such as `{"error":"invalid_request"}`. Expected status
codes: 401 for failed authentication, 422 for invalid JSON/contracts/token limits,
413 for oversized bodies, 415 for compressed input, 408 for slow body reception,
503 for unavailable model or busy capacity, and 500 for internal/backend errors.
No response exposes validation internals, paths, secrets, text through error messages,
or exception details. `/version` remains usable after a model-load failure when
configuration is valid. Invalid configuration leaves liveness available and protected
routes closed with 503. Startup loads the model before accepting traffic; liveness
remains available after a completed load attempt fails. Fixing configuration/cache
requires restarting the process.

There is no CORS or slash redirection. OpenAPI/Swagger are disabled by default;
`ENCODER_ENABLE_DOCS=1` enables development docs and schema routes, which still require
a bearer header. No unauthenticated browser documentation is exposed.

## Sections, contexts, and deterministic chunks

Sections have exactly `kind`, optional `context`, and `text`. Kinds are:
`content`, `participation`, `accessibility`, `tickets`, `additional`, `facilities`,
`location_context`, `activities`, and `categories`. Unknown fields/kinds are rejected.
Empty/whitespace-only strings and invalid Unicode surrogates are rejected.

Contexts have exactly `scope`, `venue_id`, `space_id`, and `occurrence_id`. Absent IDs
are null. UUIDs are serialized canonically without changing identity.

| Scope | Required | Optional | Must be null |
| --- | --- | --- | --- |
| `event` | — | — | all three IDs |
| `venue` | `venue_id` | — | `space_id`, `occurrence_id` |
| `space` | `space_id` | `venue_id` | `occurrence_id` |
| `occurrence` | `occurrence_id` | `venue_id`, `space_id` | — |

Scope is never inferred. Contextual and plain Sections cannot coexist in one document.
Each contextual Section is split independently. Equal text **and kind** can share a
chunk, with the union of complete distinct contexts sorted by canonical model JSON.
First occurrence determines chunk order; input order determines document order.
Different scopes or optional location IDs are never collapsed. Plain Sections follow
the admin v2 behavior: a whole document fitting within 480 tokens is combined with
blank lines and labeled `content`; larger documents split each Section by its kind.

Each emitted chunk is measured with the actual model tokenizer, including its `Document: ` prefix and special
tokens, and contains at most 480 tokens. A deterministic prefix search prefers paragraph
or sentence ends in the last 35% of the candidate. It may underfill a chunk because
BPE counts are not monotonic; every chosen piece is rechecked. Split pieces trim outer
whitespace as in the reviewed v2 contract. Overlap is the longest suffix found by the
v2 search with a standalone token count of at most 64 **including the passage prefix and special tokens**;
this is a budget, not a guarantee of exactly 64 content tokens. Splits always advance.
No approximate character-to-token conversion or tokenizer truncation is used.

## Resource limits

| Resource | Limit |
| --- | --- |
| Request body, streamed or Content-Length | 4 MiB |
| Body reception | 15 seconds |
| Texts per embedding request | 64 |
| Characters per text/Section | 200,000 |
| Documents per chunk request | 4 |
| Sections per document | 1,000 |
| Total characters per document | 200,000 |
| Total text characters per request | 800,000 |
| Chunks per document | 1,000 |
| Serialized success response | 16 MiB |
| Concurrent admitted work requests | default 2, configurable 1–8 |
| Concurrent model forwards | 1 |
| Server connections/tasks | 32; TCP backlog 32 |

Admission is immediate: saturated requests receive 503 without joining a queue. The
model lock can have at most the remaining admitted requests waiting. A slot includes
body reception, computation, and response. There is no unbounded executor submission.
Do not run additional Uvicorn workers: each would load a separate model and have its
own limits. There is no hard cancellation of a running model forward; clients can time
out while admitted computation completes. Container memory/pid/CPU limits are the
outer resource boundary. The 8 GiB Compose memory setting is an example, not a measured
capacity guarantee for 32,768-token eager-attention requests; tune it before rollout.

## Configuration and local development

Python **3.13.15**, uv **0.12.5**, and the committed `uv.lock` are the supported baseline.
Linux uses locked CPU-only inference packages. No CUDA libraries are needed.
`uv sync --locked` installs the reference backend and test tools. The locked ONNX
packages remain for historical report/merge arithmetic tests, not as supported
serving backends. For a minimal installation, use
`uv sync --locked --no-default-groups --group runtime-torch`.

```sh
uv sync --locked
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -q
```

Normal tests use a deterministic fake backend, small generated native Qwen3 weights and retrieval LoRA weights. They exercise real offline loaders
without downloading Jina weights. The `integration` and `benchmark` markers are
excluded by default. No static type checker is configured.

To run locally, set explicit environment variables or use a service manager:

```sh
export ENCODER_API_KEY_FILE=/absolute/path/embedding.key
export ENCODER_MODEL_ROOT=/absolute/path/model-cache
export ENCODER_MODEL=jina-v5
export ENCODER_BACKEND=torch
export JINA_NONCOMMERCIAL=1
export ENCODER_MAX_CONCURRENT_REQUESTS=2
uv run --no-sync python -m uranus_research_encoder
```

`ENCODER_BACKEND=torch` is the only supported configuration for v5. Requests for
`onnx` or `onnx-merged` fail configuration/readiness, and direct backend calls,
manifest validation, export and parity tools reject use before model IO. The previous
graphs implement a different architecture, task selection and pooling contract;
labeling them as v5 would be incorrect. ONNX export/parity requires separate work.
The old ONNX validation documents and `validation/` measurements are **historical v3
evidence only**, not current setup instructions or v5 performance claims.

Both `/ready` and `/version` expose backend/runtime, model revision, dimensions,
embedding version and contract version without inference. The JSON structure remains
unchanged, so `uranus-research-encoder-v1` is retained. The changed model literal and
embedding space are intentional incompatibilities, represented by service 0.2.0 and
new model/revision/embedding identifiers. Consumers must deliberately migrate and
reindex externally; this service performs neither operation.

The CLI binds to **127.0.0.1:6335** by default and installs sanitized JSON logging.
Use this entrypoint in operations: direct Uvicorn defaults enable access logging.
`ENCODER_API_KEY` is supported when no key file is configured. Model/key paths must be
absolute; startup and inference do not depend on the current working directory.
`ENCODER_MODEL_REVISION`, if supplied, must equal the pin. No `.env` file is discovered
or read by the service. For Compose examples, use `--env-file /dev/null` to suppress
Compose's separate automatic `.env` behavior too.

## Model provisioning and offline operation

Prefetch only on a separately authorized, network-enabled provisioning machine:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/prefetch_model.py \
  --model-root /absolute/path/model-cache --backend torch
```

This tool accepts no repository or revision override and supports only `--backend torch`.
Use a **separate v5 cache path**; never replace or delete an existing v3 cache.
The allowlist contains `config.json`, `tokenizer.json`, `tokenizer_config.json`,
native safetensors (including sharded-checkpoint support), and exactly
`adapters/retrieval/{adapter_config.json,adapter_model.safetensors}`. It excludes
repository Python, other task adapters, alternate weights, redundant vocabulary/merges
(the fast-tokenizer JSON is required) and sentence-transformers wrappers. Provisioning and runtime share `model_artifacts.py` as their canonical
allowlist. Excluded entries in the cached full repository tree need not be present.

Transfer the **entire provisioned cache directory**, including `blobs`, snapshot
symlinks, and cached `trees` metadata; do not download the complete repository.
Mount that tree read-only at `/models`. Files must be readable by container UID/GID
10001, including the separately mounted key.
Treat the provisioned cache as trusted immutable artifacts; it is not a substitute for
operator provenance/integrity controls. A matching snapshot directory name alone is
not a cryptographic attestation of locally modified bytes.

Runtime explicitly sets Hugging Face offline flags and calls every loader with local
files only. `trust_remote_code=False` is mandatory. Missing or incomplete model files,
missing adapters, configuration failures or revision mismatches fail readiness; no
alternative model is downloaded. No prefetch runs during image build or service startup.
Network-level egress denial can additionally be applied by the operator.

After provisioning, explicitly test the actual cache offline:

```sh
JINA_NONCOMMERCIAL=1 ENCODER_INTEGRATION_MODEL_ROOT=/absolute/path/model-cache \
  uv run --no-sync pytest -q -m integration
```

Without that variable the integration test reports a skip; a supplied invalid cache
fails. Before rollout, also run the actual Communitytreffen request with the real
loaded tokenizer, verify all three version identifiers, and measure target-host
latency/memory. Fake-model HTTP tests do not establish real-model numerical quality.

## Container and deployment boundary

The Dockerfile exposes `runtime-torch` and the default `runtime` target. ONNX targets
are not shipped for v5. Building an image does not provision model artifacts.

```sh
docker build -f deploy/Dockerfile -t uranus-research-encoder:0.2.0 .
```

The multi-stage image pins Python and uv by version and digest, installs with
`uv sync --locked`, and omits compilers and uv from the final runtime stage. It runs
as UID/GID 10001, supports a read-only root with only `/tmp` writable, and listens on
`0.0.0.0:8080` **inside** the container. The example host mapping is strictly
`127.0.0.1:6335:8080`.

For a new authorized deployment, review `deploy/compose.example.yml`, supply absolute
`ENCODER_HOST_MODEL_ROOT` and `ENCODER_HOST_KEY_FILE` environment paths, and run Compose
with `--env-file /dev/null`. The example drops all capabilities, enables
`no-new-privileges`, uses `/tmp` tmpfs, and bounds pids, CPU and memory. It does not use
privileged mode or host networking. Compose is a portable example, not production
host configuration. A systemd unit is unnecessary for this container deployment.

The healthcheck measures liveness only. An operator/client must check authenticated
`/ready` and `/version` before sending work. Configure any SSH forward on an authorized
client with a loopback local bind. This repository supplies no production SSH commands
and performs no host changes. For container-only offline smoke checks, use
`docker run --network none` and probe through `docker exec`; the server needs no egress.

## Compatibility artifacts

Read-only compatibility inspection used local `sndcds/uranus-admin` revision
`5cb7fb91a7900d443c4fa08b1901db1b5cdfa01f`, specifically the public types/transport in
`backend/app/research/{evidence_context,chunk_kinds,vector_documents,vector_models,vector_transport}.py`.
These files were not changed. No pilot encoder implementation was used.

The fixture is a synthetic copy of the requested client payload shape, not production
data. The committed schemas cover `EmbedRequest`, `EmbedResponse`, `ChunkRequest`,
`ChunkResponse`, `EvidenceContext`, `Section`, and `Chunk`. Cross-field invariants
(context scopes, mixed-context rejection, totals, unit norm) are enforced by Pydantic
validators; JSON Schema alone does not express all of them.

Regenerate schemas intentionally after a reviewed model change:

```sh
uv run --no-sync python -m uranus_research_encoder.schemas --output contracts
```

The schema drift test compares exact generated bytes to committed artifacts. CI runs
locked dependency installation, formatting, lint, and the full default test suite.
Normal CI never downloads multi-gigabyte model weights.

## Licensing

The existing repository [LICENSE](LICENSE) is AGPL-3.0; it applies to the service code.
Model weights are separate Jina AI materials under **CC-BY-NC-4.0**. Preserve Jina AI
attribution and the pinned model card when distributing/provisioning weights. See the
[model license declaration](https://huggingface.co/jinaai/jina-embeddings-v5-text-small/blob/dd76d535f5447ca3897a9c893fb1e612ead98192/README.md)
and [CC-BY-NC-4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/).

Operators must explicitly set `JINA_NONCOMMERCIAL=1` to acknowledge non-commercial
Jina use before either prefetch or runtime loading. That flag does not grant permission
for commercial use. Review the published license terms before commercial use; this documentation makes no legal assessment.
Dependencies retain their own licenses; the service license does not relicense weights.

## Migration boundary and token-count change

The preceding model used two retrieval adapters and masked mean pooling; v5 uses a
shared retrieval adapter, kind-specific prefixes and last-token pooling. Its vectors
are a different embedding space. Never extend an old collection with these vectors
under an old identifier. Rollback means running the previous service against its
unchanged previous cache; no cache deletion or automatic consumer switch is supplied.

`chunking.py`, MAX_TOKENS=480, OVERLAP=64, Section/context merging, ordering and
content hashing remain unchanged at `sections-480-overlap64-v2`. The necessary
migration difference is the tokenizer and its model-input count: `/embed` counts the
requested kind's prefix; `/chunks` and one-argument `Backend.count(text)` use passage
semantics. Texts and hashes do not contain the added prefix. Chunk boundaries may
therefore differ from the previous tokenizer despite the unchanged algorithm. There
is no normalization of user text beyond the existing chunking rules.

The actual 32K model limit does not imply that eager CPU inference fits the example
container budget at that limit. Measure target-host memory and latency before rollout.
No v5 retrieval-quality improvement or production throughput is claimed by this
repository migration.
