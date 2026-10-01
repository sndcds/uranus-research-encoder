# Experimental locally merged ONNX

Torch remains the production default. These graphs are local derived artifacts,
not the upstream Jina ONNX release. No deployment or live-cache changes are part
of this experiment. The public HTTP schemas and embedding version are unchanged.

## Export contract

`jina-v3-merged-f32-v1`, exporter `1`, opset **18**. Source:
`jinaai/jina-embeddings-v3-hf` at
`d18862d9a48706220815554fac3ebb4dfa46fc28`. Export dependencies are separate from
runtime dependencies: Torch 2.11.0 CPU, Transformers 5.17.0, PEFT 0.21.1,
ONNX 1.23.1, ONNXScript 0.6.2.

Each task loads the pinned native safetensors and only its own adapter in a
fresh process, with `local_files_only=True`, `trust_remote_code=False`, float32,
and eager attention. No downloads, remote model Python, upstream graph rewriting,
quantization, FP16, or pooling changes are involved.

The native Transformers adapter mixin does not expose `merge_and_unload`.
Instead, all ordinary linear LoRA targets are independently reconstructed as
`base + scaling * (B @ A)` (with transpose if required), then merged using
PEFT `layer.merge(safe_merge=True, adapter_names=[task])`. Expected and merged
float32 matrices must have identical SHA-256 digests. Embedding LoRA matrices use
`base + scaling * (B @ A).T` in blocks of 1024 vocabulary rows. This avoids the
multiple full vocabulary-table copies of PEFT's safe merge. Tiny-model tests
compare the embedding result directly with PEFT's `get_delta_weight` formula.
Full and blocked GEMM are [not guaranteed bit-identical across CPU kernels](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). The
cross-block fixture checks exact equality for small dyadic weights, and checks
both random float32 results against an independent float64 reference using the
standard dot-product/scaling/addition rounding bound. This does not change any
embedding parity threshold. Embedding audit hashes certify the specified blocked
formula; linear audit hashes independently certify PEFT's same-shape merge.
Every target is audited: 145 linear matrices and two embedding tables per task,
including the unused native pooler. Each adapter wrapper is then replaced by its
plain base layer. Parameters are frozen.

The first full-table merge attempt was killed with exit 137 on this host;
row-block embedding merge completed. This is an export-memory observation, not
an inference benchmark.

Before export, the entire parity corpus is evaluated through both unmerged and
merged native models. The frozen component/cosine/norm thresholds apply here
as well. Failure stops export. `TokenEmbeddings` exposes only native token states;
it supplies the equivalent additive 4D attention mask to avoid tracing the
Transformers Python mask factory. Exact native/wrapper equality is checked using
a padded two-item input before exporting.

`torch.onnx.export(dynamo=True, optimize=False, external_data=True)` preserves
dynamic batch and sequence dimensions (capture bounds batch 1–16, sequence
2–8192). ONNX checker and structural checks run after export. No optional ORT
passes are enabled in the initial backend. CPU execution is sequential, inter-op
is fixed at 1, intra-op is explicitly configurable from 1 to 8. Spinning,
CPU memory arenas, and memory patterns are disabled for the baseline.

After completing the one-thread baseline benchmark, structural inspection found
144 transposes of constant weight initializers in each export. A separate
`--merged-optimization basic` experiment enables `ORT_ENABLE_BASIC` at session
initialization. [ORT documents this level](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
as constant folding and other basic semantics-preserving rewrites. It does not
enable the extended transformer/GELU/attention fusion levels. The stored graphs
and manifest remain unchanged. The runtime setting is recorded in each new
parity/benchmark report, and must pass the same real-model parity gate before
benchmarking. The baseline default is still `disabled`; the running thread sweep
continues with that setting.

## Layout and provenance

```text
export-directory/
  manifest.json
  tokenizer/{config.json,tokenizer.json,tokenizer_config.json,...}
  retrieval-query/{model.onnx,model.onnx.data,merge-audit.json}
  retrieval-passage/{model.onnx,model.onnx.data,merge-audit.json}
```

Both graphs accept only `input_ids` and `attention_mask`, int64
`[batch, sequence]`; output `text_embeds` is float32 `[batch, sequence, 1024]`.
There is no task input or LoRA task-bank selection. Python/NumPy performs the
same masked mean pooling and float32 L2 normalization as upstream OnnxBackend.
Requests still execute one text at a time.

The operator-owned manifest identifies the source repository/revision, source
file hashes, tool versions, contract/exporter version, opset, dtype, task and
source adapter, plus SHA-256 and byte size for every generated file. Runtime
checks the contract/source identity, task-to-path mapping, required graphs and
external data, tokenizer files, paths, and every listed digest before creating
sessions. The manifest is the operator's trust root, not a signed upstream
release; replacing both manifest and files requires operator review. Local
artifact digests are never compiled into the application.

Export uses a sibling staging directory, publishes the manifest only after both
tasks complete, and verifies source hashes again before publication. Failure
leaves a diagnostic staging directory. An existing output requires `--force`,
and must identify itself as an earlier export of this contract. Source/output
overlap, including resolved source blob targets, is rejected.
Completed directories/files use modes 0755/0644 so the unprivileged runtime can
read a read-only mount. Stdout is the sorted manifest JSON; progress goes to stderr.

## Reproduction (offline)

Install development tools separately with `uv sync --locked --group export`.
The model cache must already contain the pinned native weights and both adapters.
Use an isolated output outside that cache:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/export_merged_onnx.py \
  --model-root /absolute/audit-cache \
  --output-dir /absolute/derived/merged-v1

JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/compare_backends.py \
  --model-root /absolute/audit-cache \
  --merged-root /absolute/derived/merged-v1 \
  --output validation/merged-onnx-parity.json
```

Stop if either three-way parity comparison fails. Do not relax the thresholds:
component error ≤ 1e-6, cosine ≥ 0.99999999999, norm error ≤ 1e-6, equal token
counts, identical full ranking order, task distinction, exact repeats and
request-size stability. The corpus includes five German/English queries, seven
Kulturbytes-style passages and the existing long multi-section text, all under
both tasks. Tiny generated tests are normal CI; real-cache tests require the
`integration` marker and explicit cache environment variables.

Only after parity passes, benchmark in fresh processes:

```sh
JINA_NONCOMMERCIAL=1 uv run --no-sync python scripts/benchmark_backends.py \
  --model-root /absolute/audit-cache --merged-root /absolute/derived/merged-v1 \
  --backends torch onnx onnx-merged --thread-sweep 1 2 4 8 \
  --lengths 32 128 480 1024 --batch-sizes 1 4 --warmups 2 --samples 5 \
  --output validation/merged-onnx-benchmark.json
```

Torch's production intra-op setting is 1. Compare one-thread rows directly;
ONNX 2/4/8-thread rows use more intra-op parallelism and are labeled separately.
The tool records actual token lengths, every raw timing sample, p50, p95,
texts/sec, load seconds, runtime/CPU/thread metadata and process peak RSS. The OS
file cache is not flushed. Request size 4 is sequential, not true graph batching.

The separate benchmark-only `--padded-batching` experiment groups up to four
texts with right padding. It is not enabled by the HTTP backend. After the
sequential baseline, validate it with `compare_backends.py --padded-batching`
using the same model/merged roots and chosen thread count. This checks all corpus
texts in groups, alone, and as a short/long padded pair; the extra pair is also
compared directly with the corresponding Torch vectors. Repeat output remains
exact; request-size comparisons use the unchanged component/cosine gate.
This mode also emits an independent sequential-merged comparison at the selected
thread count. Its parity-process RSS includes both modes; performance and memory
comparisons use the separately isolated benchmark workers.
Only if it passes, run `benchmark_backends.py --padded-batching --backends
onnx-merged` with the same chosen thread count and the same lengths/request sizes.
Use a separate JSON output. The timing inputs remain homogeneous within each
request to preserve the existing benchmark contract; mixed-length masking is
validated separately. Generated tests exercise real ORT batches and nonzero PAD
hidden states, so accidentally including padding in pooling fails CI.

Experimental service selection is explicit:
`ENCODER_BACKEND=onnx-merged`, `ENCODER_MERGED_ONNX_ROOT=/absolute/derived/merged-v1`,
`ENCODER_ONNX_INTRA_OP_THREADS=1`, `ENCODER_ONNX_INTER_OP_THREADS=1` and the existing
license acknowledgement. The original `onnx` backend still loads the upstream
scalar-task graph from the HF cache.
`ENCODER_MERGED_ONNX_OPTIMIZATION=basic` explicitly selects the separately gated
basic-pass experiment; leaving it unset preserves the unoptimized baseline.

## Results and release decision

Both native adapter merges passed before ONNX export:

| Task | Audited matrices | Max component difference | Minimum cosine | Max norm error |
|---|---:|---:|---:|---:|
| query | 147 | 1.396984e-7 | 0.9999999999995166 | 7.505195e-8 |
| passage | 147 | 1.201406e-7 | 0.9999999999996017 | 9.099906e-8 |

All 147 base matrix hashes match across the independent task loads. 146 merged
matrices differ between tasks; the unused `pooler.dense` is the sole unchanged
matrix. See [query audit](validation/merged-query-weight-audit.json),
[passage audit](validation/merged-passage-weight-audit.json),
[export manifest](validation/merged-export-manifest.json), and
[graph audit](validation/merged-graph-audit.json).

| Task | Graph bytes | External tensor bytes | Total MiB |
|---|---:|---:|---:|
| query | 6,840,520 | 2,233,278,464 | 2,136.34 |
| passage | 6,738,739 | 2,233,278,464 | 2,136.25 |

The host is an Intel Core i5-8265U with 8 logical CPUs and about 14 GiB RAM.
These local measurements must not be presented as new measurements on the
user's deployment server. Both task graphs remain resident in the baseline,
so duplicated float32 backbone weights are a likely memory cost to measure.
This is a shared development laptop, with other applications active; raw timing
variation and p95 matter. Fresh workers isolate each backend's process memory,
but do not isolate CPU scheduling, thermal behavior or the operating-system
file cache. A production decision still needs a repeat on the actual server.
The corpus touches only part of the vocabulary: measured RSS is not an upper
bound for every future workload or for all externally stored tensor pages.

The full three-backend semantic comparison passed; raw per-text metrics and
rankings are in [merged-onnx-parity.json](validation/merged-onnx-parity.json).
Its historical `onnx_norm`/`onnx` field names describe the candidate identified by
`candidate_backend` in each comparison.

| Candidate vs Torch | Max component difference | Minimum cosine | Max norm error | Rankings/tokens/tasks/stability |
|---|---:|---:|---:|---|
| upstream ONNX | 4.777685e-7 | 0.9999999999944622 | 7.833922e-8 | pass |
| locally merged ONNX | 4.237518e-7 | 0.9999999999954730 | 7.673905e-8 | pass |

All 26 task/text combinations are finite 1024-dimensional vectors. Full ordering
of all seven passages is identical for every query. Query and passage embeddings
differ for all 13 texts, and each separately matches its Torch adapter. Repeats
and request-size checks are exact. The largest corpus item has 912 tokens.

| Backend | Process-cold load seconds | Parity-process peak RSS MiB |
|---|---:|---:|
| Torch | 9.674 | 3867.35 |
| upstream ONNX | 17.303 | 4833.05 |
| locally merged ONNX | 29.388 | 3826.36 |

These RSS figures are from parity, not the warmed latency benchmark. Graph file
size is not resident-memory size. The manifest is immutable export-time
provenance; its release-parity reminder points to the independent report.

Latency, throughput, thread sweep and final decision are pending the benchmark.

## API references

The implementation was checked against the installed pinned package source and
[PEFT's merge semantics](https://huggingface.co/docs/peft/developer_guides/checkpoint)
and [PyTorch's export documentation](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).
Published documentation can describe newer releases; the executable contract
here is pinned and tested.
