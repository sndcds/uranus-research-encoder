# Pinned ONNX investigation

## Phase 1 findings, before backend implementation

Audit target: `jinaai/jina-embeddings-v3-hf` at
`d18862d9a48706220815554fac3ebb4dfa46fc28`. The existing native Transformers
implementation, including its two retrieval adapters, is the semantic reference.
This investigation does not authorize deployment or changes to the live cache.

Sources:

- [Pinned repository tree](https://huggingface.co/api/models/jinaai/jina-embeddings-v3-hf/tree/d18862d9a48706220815554fac3ebb4dfa46fc28?recursive=true)
- [Pinned model configuration](https://huggingface.co/jinaai/jina-embeddings-v3-hf/blob/d18862d9a48706220815554fac3ebb4dfa46fc28/config.json)
- [Pinned native usage documentation](https://huggingface.co/jinaai/jina-embeddings-v3-hf/blob/d18862d9a48706220815554fac3ebb4dfa46fc28/README.md)
- [Pinned ONNX artifacts](https://huggingface.co/jinaai/jina-embeddings-v3-hf/tree/d18862d9a48706220815554fac3ebb4dfa46fc28/onnx)

### Artifact inventory

| Path | Bytes | SHA-256 from pinned LFS metadata |
| --- | ---: | --- |
| `onnx/model.onnx` | 1,513,159 | `836322cc5d78f0047dbb336d0049b087fd81e164505c347a313cd44a772f34cc` |
| `onnx/model.onnx_data` | 2,291,339,168 | `022b369faa8015add1d036a72927ed64f8c8196435bc37a4fc9d36fc8ebc5cc2` |
| `onnx/model_fp16.onnx` | 1,147,151,362 | `329c3ea03a1815cc98f6b97efcafb9000c6c780c2d89d40d4f541b9a88434c38` |

The complete float32 graph was parsed using ONNX 1.23.1 with external-data loading
disabled; its downloaded bytes match the pinned SHA-256. It has IR version 10,
standard-domain opset 16, 3,734 nodes, 3,003 initializers, no local functions, and
no custom operator domains. Its producer metadata says PyTorch 2.2.1, not the
service's native Transformers 5.17.0 implementation. 493 initializers reference
the sibling `model.onnx_data` file. That file is mandatory.

The FP16 variant's graph nodes and declared input/output metadata were inspected
using the first and last 2 MiB HTTP ranges, without downloading all its weights.
It has the same public interface and task-selection mechanism, with 3,736 nodes
including two extra output casts. Its name does not establish float32 parity;
it is not the candidate for this float32 migration. A subsequent protobuf metadata
walk used 299 additional small ranges, skipping tensor payloads. All 3,003
initializers contain inline raw data (811 float16 and 2,192 int64), with **zero
external-data references**. It needs no companion file. Total bytes inspected were
5,419,008; the whole-file FP16 checksum was not revalidated. The checked-in
`validation/onnx-fp16-metadata.json` records that distinction.

### Actual graph interface and task semantics

| Direction | Name | Type | Shape |
| --- | --- | --- | --- |
| Input | `input_ids` | int64 | batch × sequence |
| Input | `attention_mask` | int64 | batch × sequence |
| Input | `task_id` | int64 | scalar |
| Output | `text_embeds` | float32 | batch × sequence × 1024 |
| Output | `13049` | float32 | batch × 1024 |

`task_id` feeds 198 `Gather` nodes selecting axis zero of five-bank LoRA tensors.
The graph applies the selected low-rank update to embeddings, attention, feed
forward layers, and the pooler. These are not two separately merged graphs and
not a task-independent generic representation. The task applies to the whole
batch. The other three adapter banks are embedded in the upstream external data;
their separate repository adapter files are unnecessary.

Task mapping was checked against actual weights, not inferred from filenames:
all 294 tensors in the pinned native `retrieval_query/adapter_model.safetensors`
match ONNX bank **0** exactly after float32 conversion; all 294 tensors in
`retrieval_passage/adapter_model.safetensors` match bank **1** exactly. Comparison
accounts for embedding-matrix transposes and the Q/K/V split of `Wqkv`. Maximum
absolute difference is **0** for all 588 comparisons. This agrees with the
ordered `lora_adaptations` entries in the pinned configuration. Other banks differ.

### Pooling, normalization, and equivalence limits

`text_embeds` is the final encoder layer-normalization output. The other output
is `Gather` of token zero followed by a learned `Gemm` and `Tanh`; it is **not**
attention-mask mean pooling and must not be used as the service embedding.
There is no sentence-level L2 normalization on either output. A candidate backend
must request `text_embeds`, perform float32 attention-mask mean pooling, and then
apply the reference L2 normalization externally.

The graph's standard ONNX operators do not require Python remote code or
`trust_remote_code=True`. Tokenization must still use pinned local assets with
remote code disabled. Actual CPUExecutionProvider session loading and numerical
equivalence are separate validation gates.

The task-weight comparison proves the retrieval bank mapping, not complete
native-model equivalence. This graph came from the older implementation with
different operator decomposition. Base-weight identity, rotary behavior, token
outputs, pooled embeddings, ranking, and long-input behavior still require
comparison against the exact native reference before the production default can
change. No custom two-graph export is justified solely by this inspection:
upstream structurally supports both retrieval tasks. Torch remains the default
until real-model validation establishes compatibility.

### Validation environment boundary

The existing server snapshot is read only. It contains the native Jina files but
no Jina ONNX artifacts. Required native files are copied into a new git-ignored
local audit directory; pinned ONNX files are provisioned only into that isolated
directory. No service process, deployment, live cache, or uranus-admin files are
modified. Model artifacts are never committed or downloaded by normal CI.

## Completed float32 compatibility validation

`scripts/audit_onnx.py` verifies graph/data digests, enumerates the graph, and compares
all native base tensors and both retrieval adapters against the external data.
The results in `validation/onnx-artifact-audit.json` show **294/294 base tensors**
and **588/588 retrieval-adapter tensors** matching exactly after native BF16 storage
is converted to float32. No custom export is necessary for task correctness.

`scripts/compare_backends.py` runs the exact native reference and candidate in
separate fresh processes, offline. The corpus includes all five requested German
and English queries, seven realistic cultural/community passages, and a 912-token
multi-section programme. Every text is embedded as both query and passage:
**26 comparisons**, all 1024-dimensional, finite, normalized, and stable under
repetition and request batching. Token counts match. The full report is
`validation/onnx-parity.json`.

| Observed measure | Result |
| --- | --- |
| Maximum absolute component difference | `4.777684807777405e-7` |
| Minimum cosine similarity | greater than `0.999999999994` |
| Ranking: five queries over seven passages | Identical full order for every query |
| Task behavior | Distinct query/passage representations, both matching the reference |
| Reference peak RSS in parity run | 3,349.79 MiB |
| ONNX peak RSS in parity run | 3,852.32 MiB |

The frozen acceptance gate requires maximum component error **≤ 1e-6**, cosine
similarity **≥ 1 − 1e-11**, norm error **≤ 1e-6**, exact ranking order, equal token
counts, task separation, and repeat/batch stability. The component and cosine limits
round up the observed worst errors (4.78e-7 and about 5.54e-12) with small margins;
they do not permit a different embedding space. Normal-CI negative controls reject
material errors, generic task behavior, ranking changes, and unnormalized vectors.
The measured report passes this gate. This evidence supports retaining the existing
`embedding_version`; no model, adapter, tokenizer, pooling, or chunk contract changed.
It is not a claim of bit-identical output across runtimes or every possible input.

Reproduce on an existing cache provisioned for both backends:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/audit_onnx.py \
  --model-root /absolute/path/cache --output artifact-audit.json
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/compare_backends.py \
  --model-root /absolute/path/cache --output parity.json
JINA_NONCOMMERCIAL=1 ENCODER_INTEGRATION_MODEL_ROOT=/absolute/path/cache \
  uv run --no-sync pytest -q -m integration tests/test_onnx_parity.py
```

The script was executed against the real pinned artifacts; the integration test
invokes the same script and requires an explicitly supplied cache. Neither path
downloads files. Normal CI selects only generated local ONNX/native artifacts.

## CPU benchmark conditions and limits

Machine: Intel Core i5-8265U (four physical cores/eight logical CPUs). Each benchmark
ran in a separate temporary container with **one CPU, 4 GiB RAM, no swap, no network**,
read-only root/source/model mounts, and a writable results directory only. Torch
2.11.0+cpu/Transformers 5.17.0 and ONNX Runtime 1.30.0 both used one intra-op thread;
ONNX used sequential execution, one inter-op thread, full graph optimization, no
spinning, no CPU arena, and no memory-pattern retention. Torch retained its reference
inter-op setting (four in these containers). Both perform one text per forward.

The upstream ONNX graph materializes a vocabulary-sized LoRA weight update before
looking up tokens. An initial host run with the default arena was killed during
inference; disabling arena/pattern retention allowed complete parity validation.
This is why the candidate explicitly selects those memory settings. No graph was
rewritten, exported, quantized, or approximated at startup.

Every completed benchmark case has one warmup and three measured samples. p50/p95
are exploratory empirical quantiles on a shared workstation, not production tail
latency guarantees. Model-load times are fresh-process measurements; the operating
system's file cache was not flushed. RSS is the process lifetime high-water mark at
the completed case, not the unobservable peak of a killed warmup.

Fresh-process model loading in the length sweep took **18.79 s for Torch** and
**17.29 s for ONNX**, including offline validation and ONNX artifact hashing.

`validation/onnx-benchmark-lengths.json` records completed query/passage measurements
at 32, 128, 478 (target 480), and 1023 (target 1024) tokens. Both backends hit the
container's **confirmed OOM limit during the 4096-token query warmup**, exiting 137.
There are no latency/throughput results for that case or the subsequent 4096-token
passage case. 8192 tokens were not attempted after those memory failures. This
establishes a limitation under the tested 4 GiB budget, not a universal model limit;
larger-memory target-host validation remains necessary.

| Backend | Kind | Tokens | Batch | p50 ms | p95 ms | Texts/s | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| torch | query | 32 | 1 | 579.5 | 672.2 | 1.640 | 3457.4 |
| torch | passage | 32 | 1 | 486.1 | 957.8 | 1.519 | 3457.4 |
| torch | query | 128 | 1 | 1616.6 | 2298.0 | 0.540 | 3457.4 |
| torch | passage | 128 | 1 | 1816.6 | 1869.4 | 0.582 | 3457.4 |
| torch | query | 478 | 1 | 7203.5 | 8328.6 | 0.135 | 3457.4 |
| torch | passage | 478 | 1 | 8453.0 | 8526.5 | 0.123 | 3457.4 |
| torch | query | 1023 | 1 | 19752.9 | 20545.9 | 0.051 | 3457.4 |
| torch | passage | 1023 | 1 | 22684.4 | 24318.4 | 0.043 | 3457.4 |
| onnx | query | 32 | 1 | 6695.8 | 7159.4 | 0.148 | 4062.4 |
| onnx | passage | 32 | 1 | 7785.4 | 8409.0 | 0.125 | 4076.6 |
| onnx | query | 128 | 1 | 9054.2 | 9346.2 | 0.115 | 4076.6 |
| onnx | passage | 128 | 1 | 8371.3 | 8418.8 | 0.119 | 4077.3 |
| onnx | query | 478 | 1 | 13478.1 | 13482.0 | 0.074 | 4077.3 |
| onnx | passage | 478 | 1 | 12403.6 | 12478.6 | 0.081 | 4077.3 |
| onnx | query | 1023 | 1 | 22439.8 | 23030.2 | 0.044 | 4077.3 |
| onnx | passage | 1023 | 1 | 23024.0 | 23239.1 | 0.043 | 4077.3 |

Request batches of 4 and 16 were also measured at **32 tokens per text**, under
identical container limits. All eight cases completed successfully. This is a
request-size comparison with sequential forwards, not an optimized graph-batch
implementation. Larger request batches at every longer token length were not
measured; the tool supports that extended grid. Full samples and load measurements
are in `validation/onnx-benchmark-batches.json`.

| Backend | Kind | Batch | p50 ms | p95 ms | Texts/s | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| torch | query | 4 | 1920.5 | 2554.6 | 1.862 | 3580.9 |
| torch | query | 16 | 8538.7 | 8998.5 | 1.849 | 3580.9 |
| torch | passage | 4 | 2017.8 | 2535.0 | 1.838 | 3580.9 |
| torch | passage | 16 | 9921.7 | 10196.5 | 1.696 | 3580.9 |
| onnx | query | 4 | 30681.2 | 30838.5 | 0.138 | 4078.9 |
| onnx | query | 16 | 118322.1 | 121915.0 | 0.134 | 4079.9 |
| onnx | passage | 4 | 28873.8 | 30082.5 | 0.137 | 4079.9 |
| onnx | passage | 16 | 121493.5 | 122866.1 | 0.132 | 4080.6 |

The optional benchmark tool supports lengths 32/128/480/1024/4096/8192, request sizes
1/4/16, configurable warmups/samples, fresh-process isolation, JSON checkpoints,
readable tables, and nonzero exit on an incomplete worker. It is excluded from
normal CI and never provisions models:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/benchmark_backends.py \
  --model-root /absolute/path/cache --output benchmark.json \
  --lengths 32 128 480 1024 4096 --batch-sizes 1 4 16 --warmups 2 --samples 5
```

## Dependency and image verification

`runtime-torch` and `runtime-onnx` dependency groups separate the inference packages.
Normal development installs both and the ONNX parser used to generate tiny CI
fixtures. The ONNX image contains CPU `onnxruntime==1.30.0`, tokenizer support, and
the application; Torch, PEFT, and the `onnx` exporter/parser are absent. Python 3.13
is supported by the installed wheel and the executed sessions.

The `runtime-onnx` target was built offline after the normal Docker builder could
not resolve the package registry. A temporary build context added only a clean public
dependency-cache COPY and `--offline` to builder sync commands. Runtime stages were
unchanged. No credentials or proxy values were forwarded. The resulting image is
**436,297,544 bytes (416.09 MiB)**, ID prefix `04adcc5b9d45`. Model weights are mounted
separately and are not included in that size. The previously validated Torch reference
image is approximately 1.22 GB; the upstream ONNX model artifact itself is larger
than the native BF16 safetensors artifact (about 2.29 GB versus 1.12 GB).

A temporary network-disabled, read-only container passed application/ORT import
checks and confirmed absence of Torch, PEFT, and `onnx`. Real ONNX benchmarks also
loaded the hash-verified graph and performed inference in that read-only image.
ORT emits a device-ID persistence warning on read-only startup and falls back to
an in-memory identifier; telemetry is explicitly disabled by the backend. No request
text or embedding vectors are emitted by the service or benchmark tool.

## Checks and production decision

- `uv run --no-sync ruff format --check .`: passed (40 Python files).
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync pytest -q`: **173 passed, 2 integration tests deselected,
  10 upstream warnings** in 9.65s. Executed outside the sandbox because of the
  previously documented TestClient startup issue.
- Generated ONNX tests exercise actual CPU sessions, task selection, masked pooling,
  normalization, deterministic batches, metadata, offline partial snapshots, missing
  files, changed graph/data rejection, and token limits. Existing native integrity
  and API/chunk/EvidenceContext tests remain passing.
- The 62 contract/context/chunk tests also passed with a Python import guard rejecting
  Torch, Transformers, ONNX Runtime, PEFT, and Qdrant imports. UUID handling and chunk
  hashing require no model loading or external services. Missing-PyTorch tokenizer
  warnings in the ONNX-only environment are informational; Torch is not installed
  there to silence them.

**Torch remains the default and recommended production backend on this evidence.**
Upstream float32 ONNX is numerically sufficient for the tested retrieval contract,
but it is slower in every completed single-text and batch case and requires more
memory here. Maximum recorded benchmark RSS was 3,580.91 MiB for Torch and
4,080.64 MiB for ONNX.
No custom two-graph export was performed. A separately reviewed, offline export with
each retrieval adapter merged could be investigated to remove dynamic vocabulary
updates, but it would need fresh artifact hashes, parity, ranking, and benchmarks.
No deployment, live model-cache modification, or uranus-admin change was made.
